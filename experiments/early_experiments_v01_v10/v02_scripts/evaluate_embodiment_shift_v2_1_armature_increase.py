import logging

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

import mujoco

MODEL_FILE = "dynamics_model_v7.pt"

SEEDS = [

    8001,

    8002,

    8003,

    8004,

    8005,

]

ROLLOUT_STEPS = 100

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

J4_ARM_INDEX = 3

ARMATURE_MULTIPLIERS = [

    1.0,

    1.5,

    2.0,

    3.0,

    5.0,

]

logging.getLogger(

    "robosuite"

).setLevel(

    logging.WARNING

)

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(

                INPUT_DIM,

                HIDDEN

            ),

            nn.ReLU(),

            nn.Linear(

                HIDDEN,

                HIDDEN

            ),

            nn.ReLU(),

            nn.Linear(

                HIDDEN,

                OUTPUT_DIM

            )

        )

    def forward(self, x):

        return self.network(x)

model = DynamicsModel().cpu()

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu"

)

if "model_state_dict" in checkpoint:

    state_dict = checkpoint[

        "model_state_dict"

    ]

else:

    state_dict = checkpoint

state_dict = {

    k.replace(

        "net.",

        "network.",

        1

    ): v

    for k, v in state_dict.items()

}

model.load_state_dict(

    state_dict

)

model.eval()

def extract_state(obs):

    joint_pos = np.asarray(

        obs["robot0_joint_pos"],

        dtype=np.float32

    )

    joint_vel = np.asarray(

        obs["robot0_joint_vel"],

        dtype=np.float32

    )

    eef_pos = np.asarray(

        obs["robot0_eef_pos"],

        dtype=np.float32

    )

    state = np.concatenate(

        [

            joint_pos,

            joint_vel,

            eef_pos

        ]

    ).astype(

        np.float32

    )

    assert state.shape == (17,)

    return state

def make_env():

    return suite.make(

        env_name="Lift",

        robots="Panda",

        has_renderer=False,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        use_object_obs=False,

        control_freq=20,

    )

def generate_actions(seed):

    np.random.seed(seed)

    actions = []

    for _ in range(

        ROLLOUT_STEPS

    ):

        action = np.zeros(

            7,

            dtype=np.float32

        )

        action_idx = np.random.randint(

            0,

            7

        )

        magnitude = np.random.uniform(

            0.05,

            0.20

        )

        sign = np.random.choice(

            [-1.0, 1.0]

        )

        action[action_idx] = (

            sign * magnitude

        )

        actions.append(

            action

        )

    return np.asarray(

        actions,

        dtype=np.float32

    )

def run_experiment(

    seed,

    target_armature,

    initial_qpos,

    initial_qvel,

    actions,

):

    env_a = make_env()

    env_b = make_env()

    np.random.seed(seed)

    torch.manual_seed(seed)

    obs_a = env_a.reset()

    np.random.seed(seed)

    torch.manual_seed(seed)

    obs_b = env_b.reset()

    env_a.sim.data.qpos[:] = (

        initial_qpos

    )

    env_a.sim.data.qvel[:] = (

        initial_qvel

    )

    env_b.sim.data.qpos[:] = (

        initial_qpos

    )

    env_b.sim.data.qvel[:] = (

        initial_qvel

    )

    j4_dof = int(

        env_b.robots[0]

        .arm_joint_indexes[J4_ARM_INDEX]

    )

    baseline_armature = float(

        env_b

        .sim

        .model

        .dof_armature[j4_dof]

    )

    env_b.sim.model.dof_armature[

        j4_dof

    ] = target_armature

    mujoco.mj_setConst(

        env_b.sim.model._model,

        env_b.sim.data._data

    )

    env_a.sim.forward()

    env_b.sim.forward()

    obs_a = env_a._get_observations(

        force_update=True

    )

    obs_b = env_b._get_observations(

        force_update=True

    )

    initial_state_a = extract_state(

        obs_a

    )

    initial_state_b = extract_state(

        obs_b

    )

    initial_difference = float(

        np.max(

            np.abs(

                initial_state_a

                - initial_state_b

            )

        )

    )

    one_step_errors_a = []

    one_step_errors_b = []

    rollout_errors_a = []

    rollout_errors_b = []

    actual_a = []

    actual_b = []

    predicted_a = []

    predicted_b = []

    actions_used = []

    free_state_a = (

        initial_state_a.copy()

    )

    free_state_b = (

        initial_state_b.copy()

    )

    for step in range(

        ROLLOUT_STEPS

    ):

        action = actions[step]

        actions_used.append(

            action.copy()

        )

        current_obs_a = (

            env_a._get_observations(

                force_update=True

            )

        )

        current_actual_a = (

            extract_state(

                current_obs_a

            )

        )

        current_obs_b = (

            env_b._get_observations(

                force_update=True

            )

        )

        current_actual_b = (

            extract_state(

                current_obs_b

            )

        )

        tf_input_a = np.concatenate(

            [

                current_actual_a,

                action

            ]

        ).astype(

            np.float32

        )

        with torch.no_grad():

            tf_prediction_a = model(

                torch.from_numpy(

                    tf_input_a

                ).unsqueeze(0)

            ).numpy()[0]

        tf_input_b = np.concatenate(

            [

                current_actual_b,

                action

            ]

        ).astype(

            np.float32

        )

        with torch.no_grad():

            tf_prediction_b = model(

                torch.from_numpy(

                    tf_input_b

                ).unsqueeze(0)

            ).numpy()[0]

        next_obs_a, _, _, _ = (

            env_a.step(action)

        )

        actual_next_a = (

            extract_state(

                next_obs_a

            )

        )

        next_obs_b, _, _, _ = (

            env_b.step(action)

        )

        actual_next_b = (

            extract_state(

                next_obs_b

            )

        )

        one_step_errors_a.append(

            np.abs(

                tf_prediction_a

                - actual_next_a

            )

        )

        one_step_errors_b.append(

            np.abs(

                tf_prediction_b

                - actual_next_b

            )

        )

        free_input_a = np.concatenate(

            [

                free_state_a,

                action

            ]

        ).astype(

            np.float32

        )

        with torch.no_grad():

            free_prediction_a = model(

                torch.from_numpy(

                    free_input_a

                ).unsqueeze(0)

            ).numpy()[0]

        free_input_b = np.concatenate(

            [

                free_state_b,

                action

            ]

        ).astype(

            np.float32

        )

        with torch.no_grad():

            free_prediction_b = model(

                torch.from_numpy(

                    free_input_b

                ).unsqueeze(0)

            ).numpy()[0]

        rollout_errors_a.append(

            np.abs(

                free_prediction_a

                - actual_next_a

            )

        )

        rollout_errors_b.append(

            np.abs(

                free_prediction_b

                - actual_next_b

            )

        )

        actual_a.append(

            actual_next_a.copy()

        )

        actual_b.append(

            actual_next_b.copy()

        )

        predicted_a.append(

            free_prediction_a.copy()

        )

        predicted_b.append(

            free_prediction_b.copy()

        )

        free_state_a = (

            free_prediction_a.copy()

        )

        free_state_b = (

            free_prediction_b.copy()

        )

    one_step_errors_a = np.asarray(

        one_step_errors_a,

        dtype=np.float32

    )

    one_step_errors_b = np.asarray(

        one_step_errors_b,

        dtype=np.float32

    )

    rollout_errors_a = np.asarray(

        rollout_errors_a,

        dtype=np.float32

    )

    rollout_errors_b = np.asarray(

        rollout_errors_b,

        dtype=np.float32

    )

    actual_a = np.asarray(

        actual_a,

        dtype=np.float32

    )

    actual_b = np.asarray(

        actual_b,

        dtype=np.float32

    )

    predicted_a = np.asarray(

        predicted_a,

        dtype=np.float32

    )

    predicted_b = np.asarray(

        predicted_b,

        dtype=np.float32

    )

    actions_used = np.asarray(

        actions_used,

        dtype=np.float32

    )

    a_one_step = float(

        one_step_errors_a.mean()

    )

    b_one_step = float(

        one_step_errors_b.mean()

    )

    a_rollout = float(

        rollout_errors_a.mean()

    )

    b_rollout = float(

        rollout_errors_b.mean()

    )

    a_final = float(

        rollout_errors_a[-1].mean()

    )

    b_final = float(

        rollout_errors_b[-1].mean()

    )

    a_max = float(

        rollout_errors_a.max()

    )

    b_max = float(

        rollout_errors_b.max()

    )

    b_time_error = (

        rollout_errors_b.mean(

            axis=1

        )

    )

    def first_threshold(

        threshold

    ):

        hits = np.where(

            b_time_error >= threshold

        )[0]

        if len(hits) == 0:

            return -1

        return int(

            hits[0] + 1

        )

    first_01 = first_threshold(

        0.10

    )

    first_02 = first_threshold(

        0.20

    )

    first_05 = first_threshold(

        0.50

    )

    b_one_step_dim = (

        one_step_errors_b.mean(

            axis=0

        )

    )

    b_rollout_dim = (

        rollout_errors_b.mean(

            axis=0

        )

    )

    env_a.close()

    env_b.close()

    return {

        "seed":

            seed,

        "target_armature":

            target_armature,

        "baseline_armature":

            baseline_armature,

        "initial_difference":

            initial_difference,

        "a_one_step":

            a_one_step,

        "b_one_step":

            b_one_step,

        "a_rollout":

            a_rollout,

        "b_rollout":

            b_rollout,

        "a_final":

            a_final,

        "b_final":

            b_final,

        "a_max":

            a_max,

        "b_max":

            b_max,

        "first_01":

            first_01,

        "first_02":

            first_02,

        "first_05":

            first_05,

        "b_one_step_dim":

            b_one_step_dim,

        "b_rollout_dim":

            b_rollout_dim,

        "actual_a":

            actual_a,

        "actual_b":

            actual_b,

        "predicted_a":

            predicted_a,

        "predicted_b":

            predicted_b,

        "actions":

            actions_used,

    }

print("=" * 90)

print(

    "EMBODIMENT SHIFT V2.1 — "

    "INCREASED ARMATURE"

)

print("=" * 90)

print()

print(

    f"Model   : {MODEL_FILE}"

)

print(

    f"Seeds   : {SEEDS}"

)

print(

    f"Horizon : {ROLLOUT_STEPS}"

)

print()

print(

    "Armature multipliers:",

    ARMATURE_MULTIPLIERS

)

all_results = []

baseline_reference = None

for seed in SEEDS:

    print()

    print(

        "#" * 90

    )

    print(

        f"SEED {seed}"

    )

    print(

        "#" * 90

    )

    actions = generate_actions(

        seed

    )

    reference_env = make_env()

    np.random.seed(seed)

    torch.manual_seed(seed)

    reference_env.reset()

    initial_qpos = (

        reference_env

        .sim

        .data

        .qpos

        .copy()

    )

    initial_qvel = (

        reference_env

        .sim

        .data

        .qvel

        .copy()

    )

    j4_dof_reference = int(

        reference_env

        .robots[0]

        .arm_joint_indexes[J4_ARM_INDEX]

    )

    baseline_armature = float(

        reference_env

        .sim

        .model

        .dof_armature[

            j4_dof_reference

        ]

    )

    if baseline_reference is None:

        baseline_reference = (

            baseline_armature

        )

    elif not np.isclose(

        baseline_reference,

        baseline_armature

    ):

        reference_env.close()

        raise RuntimeError(

            "Baseline armature changed "

            "between environments."

        )

    reference_env.close()

    print()

    print(

        f"Detected Panda J4 baseline "

        f"armature = "

        f"{baseline_armature:.8f}"

    )

    target_values = [

        baseline_armature * multiplier

        for multiplier in

        ARMATURE_MULTIPLIERS

    ]

    for multiplier, target in zip(

        ARMATURE_MULTIPLIERS,

        target_values

    ):

        result = run_experiment(

            seed=seed,

            target_armature=target,

            initial_qpos=(

                initial_qpos

            ),

            initial_qvel=(

                initial_qvel

            ),

            actions=actions,

        )

        result[

            "multiplier"

        ] = multiplier

        all_results.append(

            result

        )

        print()

        print(

            f"J4 armature "

            f"x{multiplier:.1f}"

        )

        print(

            f"  baseline : "

            f"{result['baseline_armature']:.8f}"

        )

        print(

            f"  target   : "

            f"{result['target_armature']:.8f}"

        )

        print()

        print(

            "  TEACHER-FORCED"

        )

        print(

            f"    A 1-step : "

            f"{result['a_one_step']:.8f}"

        )

        print(

            f"    B 1-step : "

            f"{result['b_one_step']:.8f}"

        )

        print()

        print(

            "  FREE-RUNNING"

        )

        print(

            f"    A rollout : "

            f"{result['a_rollout']:.8f}"

        )

        print(

            f"    B rollout : "

            f"{result['b_rollout']:.8f}"

        )

        print(

            f"    A final   : "

            f"{result['a_final']:.8f}"

        )

        print(

            f"    B final   : "

            f"{result['b_final']:.8f}"

        )

        print(

            f"    A max     : "

            f"{result['a_max']:.8f}"

        )

        print(

            f"    B max     : "

            f"{result['b_max']:.8f}"

        )

        print()

        print(

            "  FAILURE TIMING"

        )

        print(

            f"    B >=0.10 : "

            f"{result['first_01']}"

        )

        print(

            f"    B >=0.20 : "

            f"{result['first_02']}"

        )

        print(

            f"    B >=0.50 : "

            f"{result['first_05']}"

        )

print()

print("=" * 90)

print(

    "V2.1 ARMATURE INCREASE SUMMARY"

)

print("=" * 90)

print()

header = (

    f"{'Mult':>7} | "

    f"{'Target':>10} | "

    f"{'A 1-step':>11} | "

    f"{'B 1-step':>11} | "

    f"{'Δ1-step':>11} | "

    f"{'A Roll':>11} | "

    f"{'B Roll':>11} | "

    f"{'ΔRoll':>11}"

)

print(header)

print("-" * len(header))

for multiplier in (

    ARMATURE_MULTIPLIERS

):

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    target = np.mean(

        [

            r["target_armature"]

            for r in rows

        ]

    )

    a_one = np.mean(

        [

            r["a_one_step"]

            for r in rows

        ]

    )

    b_one = np.mean(

        [

            r["b_one_step"]

            for r in rows

        ]

    )

    a_roll = np.mean(

        [

            r["a_rollout"]

            for r in rows

        ]

    )

    b_roll = np.mean(

        [

            r["b_rollout"]

            for r in rows

        ]

    )

    d_one = (

        b_one

        - a_one

    )

    d_roll = (

        b_roll

        - a_roll

    )

    print(

        f"{multiplier:7.1f} | "

        f"{target:10.4f} | "

        f"{a_one:11.8f} | "

        f"{b_one:11.8f} | "

        f"{d_one:+11.8f} | "

        f"{a_roll:11.8f} | "

        f"{b_roll:11.8f} | "

        f"{d_roll:+11.8f}"

    )

print()

print("=" * 90)

print("FINAL / MAX ERROR")

print("=" * 90)

for multiplier in (

    ARMATURE_MULTIPLIERS

):

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    b_final = np.mean(

        [

            r["b_final"]

            for r in rows

        ]

    )

    b_max = np.mean(

        [

            r["b_max"]

            for r in rows

        ]

    )

    print()

    print(

        f"Armature x{multiplier:.1f}"

    )

    print(

        f"  B final : "

        f"{b_final:.8f}"

    )

    print(

        f"  B max   : "

        f"{b_max:.8f}"

    )

print()

print("=" * 90)

print("FAILURE THRESHOLD SUMMARY")

print("=" * 90)

for multiplier in (

    ARMATURE_MULTIPLIERS

):

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    print()

    print(

        f"Armature x{multiplier:.1f}"

    )

    for threshold, key in [

        (

            0.10,

            "first_01"

        ),

        (

            0.20,

            "first_02"

        ),

        (

            0.50,

            "first_05"

        ),

    ]:

        values = np.asarray(

            [

                r[key]

                for r in rows

            ],

            dtype=np.int32

        )

        valid = values[

            values >= 0

        ]

        print(

            f"  >= {threshold:.2f}: "

            f"{len(valid)} / "

            f"{len(values)} seeds"

        )

        if len(valid) > 0:

            print(

                f"      mean step = "

                f"{valid.mean():.2f}"

            )

dim_names = []

for i in range(7):

    dim_names.append(

        f"J{i + 1}_Pos"

    )

for i in range(7):

    dim_names.append(

        f"J{i + 1}_Vel"

    )

dim_names.extend(

    [

        "EEF_X",

        "EEF_Y",

        "EEF_Z"

    ]

)

print()

print("=" * 90)

print("TOP ROBOT B ERROR DIMENSIONS")

print("=" * 90)

for multiplier in (

    ARMATURE_MULTIPLIERS

):

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    matrix = np.asarray(

        [

            r["b_rollout_dim"]

            for r in rows

        ]

    )

    mean_dim = matrix.mean(

        axis=0

    )

    sorted_dims = np.argsort(

        mean_dim

    )[::-1]

    print()

    print(

        f"Armature x{multiplier:.1f}"

    )

    for rank, dim in enumerate(

        sorted_dims[:5],

        1

    ):

        print(

            f"  {rank}. "

            f"{dim_names[dim]:8s} "

            f"{mean_dim[dim]:.8f}"

        )

np.savez(

    "embodiment_shift_v2_1_"

    "armature_increase_results.npz",

    seeds=np.asarray(

        SEEDS,

        dtype=np.int32

    ),

    multipliers=np.asarray(

        ARMATURE_MULTIPLIERS,

        dtype=np.float32

    ),

    baseline_armature=np.float32(

        baseline_reference

    ),

    target_armatures=np.asarray(

        [

            baseline_reference * multiplier

            for multiplier

            in ARMATURE_MULTIPLIERS

        ],

        dtype=np.float32

    ),

    a_one_step=np.asarray(

        [

            r["a_one_step"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_one_step=np.asarray(

        [

            r["b_one_step"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    a_rollout=np.asarray(

        [

            r["a_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_rollout=np.asarray(

        [

            r["b_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    a_final=np.asarray(

        [

            r["a_final"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_final=np.asarray(

        [

            r["b_final"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    a_max=np.asarray(

        [

            r["a_max"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_max=np.asarray(

        [

            r["b_max"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    first_01=np.asarray(

        [

            r["first_01"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    first_02=np.asarray(

        [

            r["first_02"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    first_05=np.asarray(

        [

            r["first_05"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    b_one_step_dim=np.asarray(

        [

            r["b_one_step_dim"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_rollout_dim=np.asarray(

        [

            r["b_rollout_dim"]

            for r in all_results

        ],

        dtype=np.float32

    ),

)

for result in all_results:

    seed = result["seed"]

    multiplier = result[

        "multiplier"

    ]

    tag = (

        f"seed{seed}_"

        f"mult{multiplier:.1f}"

        .replace(

            ".",

            "_"

        )

    )

    np.savez(

        f"embodiment_v2_1_{tag}.npz",

        actual_a=result[

            "actual_a"

        ],

        actual_b=result[

            "actual_b"

        ],

        predicted_a=result[

            "predicted_a"

        ],

        predicted_b=result[

            "predicted_b"

        ],

        actions=result[

            "actions"

        ],

    )

print()

print("=" * 90)

print("SAVED")

print("=" * 90)

print(

    "embodiment_shift_v2_1_"

    "armature_increase_results.npz"

)

print(

    "Full trajectories:"

)

print(

    "embodiment_v2_1_seed*_mult*.npz"

)

print()

print("=" * 90)

print("DONE")

print("=" * 90)
