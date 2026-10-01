import logging

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

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

ARMATURE_VALUES = [

    0.00,

    0.01,

    0.05,

    0.10,

    0.25,

]

J4_ARM_INDEX = 3

logging.getLogger("robosuite").setLevel(

    logging.WARNING

)

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(INPUT_DIM, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, OUTPUT_DIM),

        )

    def forward(self, x):

        return self.network(x)

model = DynamicsModel().cpu()

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu",

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

        dtype=np.float32,

    )

    joint_vel = np.asarray(

        obs["robot0_joint_vel"],

        dtype=np.float32,

    )

    eef_pos = np.asarray(

        obs["robot0_eef_pos"],

        dtype=np.float32,

    )

    state = np.concatenate(

        [

            joint_pos,

            joint_vel,

            eef_pos,

        ]

    ).astype(np.float32)

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

            dtype=np.float32,

        )

        action_idx = np.random.randint(

            0,

            7,

        )

        magnitude = np.random.uniform(

            0.05,

            0.20,

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

        dtype=np.float32,

    )

def apply_armature(

    env,

    armature_value,

):

    j4_dof = int(

        env.robots[0].arm_joint_indexes[

            J4_ARM_INDEX

        ]

    )

    original_armature = float(

        env.sim.model.dof_armature[

            j4_dof

        ]

    )

    env.sim.model.dof_armature[

        j4_dof

    ] = armature_value

    modified_armature = float(

        env.sim.model.dof_armature[

            j4_dof

        ]

    )

    env.sim.forward()

    return (

        original_armature,

        modified_armature,

        j4_dof,

    )

def run_experiment(

    seed,

    armature_value,

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

    (

        original_armature,

        modified_armature,

        j4_dof,

    ) = apply_armature(

        env_b,

        armature_value,

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

    free_state_a = initial_state_a.copy()

    free_state_b = initial_state_b.copy()

    for step in range(

        ROLLOUT_STEPS

    ):

        action = actions[step]

        actions_used.append(

            action.copy()

        )

        actual_obs_a = env_a._get_observations(

            force_update=True

        )

        actual_obs_b = env_b._get_observations(

            force_update=True

        )

        current_actual_a = extract_state(

            actual_obs_a

        )

        current_actual_b = extract_state(

            actual_obs_b

        )

        tf_input_a = np.concatenate(

            [

                current_actual_a,

                action,

            ]

        ).astype(np.float32)

        with torch.no_grad():

            tf_prediction_a = model(

                torch.from_numpy(

                    tf_input_a

                ).unsqueeze(0)

            ).numpy()[0]

        tf_input_b = np.concatenate(

            [

                current_actual_b,

                action,

            ]

        ).astype(np.float32)

        with torch.no_grad():

            tf_prediction_b = model(

                torch.from_numpy(

                    tf_input_b

                ).unsqueeze(0)

            ).numpy()[0]

        next_obs_a, _, _, _ = (

            env_a.step(action)

        )

        next_obs_b, _, _, _ = (

            env_b.step(action)

        )

        actual_next_a = extract_state(

            next_obs_a

        )

        actual_next_b = extract_state(

            next_obs_b

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

                action,

            ]

        ).astype(np.float32)

        with torch.no_grad():

            free_prediction_a = model(

                torch.from_numpy(

                    free_input_a

                ).unsqueeze(0)

            ).numpy()[0]

        free_input_b = np.concatenate(

            [

                free_state_b,

                action,

            ]

        ).astype(np.float32)

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

        dtype=np.float32,

    )

    one_step_errors_b = np.asarray(

        one_step_errors_b,

        dtype=np.float32,

    )

    rollout_errors_a = np.asarray(

        rollout_errors_a,

        dtype=np.float32,

    )

    rollout_errors_b = np.asarray(

        rollout_errors_b,

        dtype=np.float32,

    )

    actual_a = np.asarray(

        actual_a,

        dtype=np.float32,

    )

    actual_b = np.asarray(

        actual_b,

        dtype=np.float32,

    )

    predicted_a = np.asarray(

        predicted_a,

        dtype=np.float32,

    )

    predicted_b = np.asarray(

        predicted_b,

        dtype=np.float32,

    )

    actions_used = np.asarray(

        actions_used,

        dtype=np.float32,

    )

    a_one_step_mae = float(

        one_step_errors_a.mean()

    )

    b_one_step_mae = float(

        one_step_errors_b.mean()

    )

    a_rollout_mae = float(

        rollout_errors_a.mean()

    )

    b_rollout_mae = float(

        rollout_errors_b.mean()

    )

    a_final_mae = float(

        rollout_errors_a[-1].mean()

    )

    b_final_mae = float(

        rollout_errors_b[-1].mean()

    )

    a_max_error = float(

        rollout_errors_a.max()

    )

    b_max_error = float(

        rollout_errors_b.max()

    )

    b_time_error = (

        rollout_errors_b.mean(axis=1)

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

    b_one_step_dim_mae = (

        one_step_errors_b.mean(

            axis=0

        )

    )

    b_rollout_dim_mae = (

        rollout_errors_b.mean(

            axis=0

        )

    )

    env_a.close()

    env_b.close()

    return {

        "seed":

            seed,

        "armature":

            armature_value,

        "original_armature":

            original_armature,

        "modified_armature":

            modified_armature,

        "j4_dof":

            j4_dof,

        "initial_difference":

            initial_difference,

        "a_one_step":

            a_one_step_mae,

        "b_one_step":

            b_one_step_mae,

        "a_rollout":

            a_rollout_mae,

        "b_rollout":

            b_rollout_mae,

        "a_final":

            a_final_mae,

        "b_final":

            b_final_mae,

        "a_max":

            a_max_error,

        "b_max":

            b_max_error,

        "first_01":

            first_01,

        "first_02":

            first_02,

        "first_05":

            first_05,

        "b_one_step_dim":

            b_one_step_dim_mae,

        "b_rollout_dim":

            b_rollout_dim_mae,

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

print("EMBODIMENT SHIFT V2 — ARMATURE")

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

    "Absolute J4 armature values:",

    ARMATURE_VALUES

)

all_results = []

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

    reference_env.close()

    for armature_value in (

        ARMATURE_VALUES

    ):

        result = run_experiment(

            seed=seed,

            armature_value=(

                armature_value

            ),

            initial_qpos=(

                initial_qpos

            ),

            initial_qvel=(

                initial_qvel

            ),

            actions=(

                actions

            ),

        )

        all_results.append(

            result

        )

        print()

        print(

            f"J4 armature = "

            f"{armature_value:.4f}"

        )

        print(

            f"  original armature : "

            f"{result['original_armature']:.8f}"

        )

        print(

            f"  modified armature : "

            f"{result['modified_armature']:.8f}"

        )

        print(

            f"  initial A/B diff  : "

            f"{result['initial_difference']:.10f}"

        )

        print()

        print(

            "  TEACHER-FORCED"

        )

        print(

            f"    A one-step MAE : "

            f"{result['a_one_step']:.8f}"

        )

        print(

            f"    B one-step MAE : "

            f"{result['b_one_step']:.8f}"

        )

        print()

        print(

            "  FREE-RUNNING"

        )

        print(

            f"    A rollout MAE  : "

            f"{result['a_rollout']:.8f}"

        )

        print(

            f"    B rollout MAE  : "

            f"{result['b_rollout']:.8f}"

        )

        print(

            f"    A final MAE    : "

            f"{result['a_final']:.8f}"

        )

        print(

            f"    B final MAE    : "

            f"{result['b_final']:.8f}"

        )

        print(

            f"    A max error    : "

            f"{result['a_max']:.8f}"

        )

        print(

            f"    B max error    : "

            f"{result['b_max']:.8f}"

        )

        print()

        print(

            "  FAILURE TIMING"

        )

        print(

            f"    B >= 0.10 : "

            f"{result['first_01']}"

        )

        print(

            f"    B >= 0.20 : "

            f"{result['first_02']}"

        )

        print(

            f"    B >= 0.50 : "

            f"{result['first_05']}"

        )

print()

print("=" * 90)

print("ARMATURE SWEEP SUMMARY")

print("=" * 90)

print()

header = (

    f"{'Armature':>10} | "

    f"{'A 1-step':>11} | "

    f"{'B 1-step':>11} | "

    f"{'1-step Δ':>11} | "

    f"{'A Rollout':>11} | "

    f"{'B Rollout':>11} | "

    f"{'Rollout Δ':>11}"

)

print(header)

print("-" * len(header))

aggregate = []

for armature_value in (

    ARMATURE_VALUES

):

    rows = [

        r for r in all_results

        if np.isclose(

            r["armature"],

            armature_value

        )

    ]

    a_one_step = np.mean(

        [

            r["a_one_step"]

            for r in rows

        ]

    )

    b_one_step = np.mean(

        [

            r["b_one_step"]

            for r in rows

        ]

    )

    a_rollout = np.mean(

        [

            r["a_rollout"]

            for r in rows

        ]

    )

    b_rollout = np.mean(

        [

            r["b_rollout"]

            for r in rows

        ]

    )

    one_step_delta = (

        b_one_step

        - a_one_step

    )

    rollout_delta = (

        b_rollout

        - a_rollout

    )

    aggregate.append(

        {

            "armature":

                armature_value,

            "a_one_step":

                a_one_step,

            "b_one_step":

                b_one_step,

            "one_step_delta":

                one_step_delta,

            "a_rollout":

                a_rollout,

            "b_rollout":

                b_rollout,

            "rollout_delta":

                rollout_delta,

        }

    )

    print(

        f"{armature_value:10.4f} | "

        f"{a_one_step:11.8f} | "

        f"{b_one_step:11.8f} | "

        f"{one_step_delta:+11.8f} | "

        f"{a_rollout:11.8f} | "

        f"{b_rollout:11.8f} | "

        f"{rollout_delta:+11.8f}"

    )

print()

print("=" * 90)

print("FINAL / MAX ERROR")

print("=" * 90)

for armature_value in (

    ARMATURE_VALUES

):

    rows = [

        r for r in all_results

        if np.isclose(

            r["armature"],

            armature_value

        )

    ]

    a_final = np.mean(

        [

            r["a_final"]

            for r in rows

        ]

    )

    b_final = np.mean(

        [

            r["b_final"]

            for r in rows

        ]

    )

    a_max = np.mean(

        [

            r["a_max"]

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

        f"Armature {armature_value:.4f}"

    )

    print(

        f"  A final : "

        f"{a_final:.8f}"

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

for armature_value in (

    ARMATURE_VALUES

):

    rows = [

        r for r in all_results

        if np.isclose(

            r["armature"],

            armature_value

        )

    ]

    print()

    print(

        f"Armature {armature_value:.4f}"

    )

    for threshold, key in [

        (0.10, "first_01"),

        (0.20, "first_02"),

        (0.50, "first_05"),

    ]:

        values = np.asarray(

            [

                r[key]

                for r in rows

            ],

            dtype=np.int32,

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

                f"      mean step : "

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

        "EEF_Z",

    ]

)

print()

print("=" * 90)

print("TOP ROBOT B ERROR DIMENSIONS")

print("=" * 90)

for armature_value in (

    ARMATURE_VALUES

):

    rows = [

        r for r in all_results

        if np.isclose(

            r["armature"],

            armature_value

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

        f"Armature {armature_value:.4f}"

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

    "embodiment_shift_v2_armature_results.npz",

    seeds=np.asarray(

        SEEDS,

        dtype=np.int32,

    ),

    armature_values=np.asarray(

        ARMATURE_VALUES,

        dtype=np.float32,

    ),

    a_one_step=np.asarray(

        [

            r["a_one_step"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    b_one_step=np.asarray(

        [

            r["b_one_step"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    a_rollout=np.asarray(

        [

            r["a_rollout"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    b_rollout=np.asarray(

        [

            r["b_rollout"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    a_final=np.asarray(

        [

            r["a_final"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    b_final=np.asarray(

        [

            r["b_final"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    a_max=np.asarray(

        [

            r["a_max"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    b_max=np.asarray(

        [

            r["b_max"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    first_01=np.asarray(

        [

            r["first_01"]

            for r in all_results

        ],

        dtype=np.int32,

    ),

    first_02=np.asarray(

        [

            r["first_02"]

            for r in all_results

        ],

        dtype=np.int32,

    ),

    first_05=np.asarray(

        [

            r["first_05"]

            for r in all_results

        ],

        dtype=np.int32,

    ),

    b_one_step_dim=np.asarray(

        [

            r["b_one_step_dim"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

    b_rollout_dim=np.asarray(

        [

            r["b_rollout_dim"]

            for r in all_results

        ],

        dtype=np.float32,

    ),

)

for result in all_results:

    seed = result["seed"]

    armature = result["armature"]

    tag = (

        f"seed{seed}_"

        f"arm{armature:.4f}"

        .replace(".", "_")

    )

    np.savez(

        f"embodiment_v2_{tag}.npz",

        actual_a=result["actual_a"],

        actual_b=result["actual_b"],

        predicted_a=result["predicted_a"],

        predicted_b=result["predicted_b"],

        actions=result["actions"],

    )

print()

print("=" * 90)

print("SAVED")

print("=" * 90)

print(

    "embodiment_shift_v2_armature_results.npz"

)

print(

    "Full per-seed/per-armature trajectories:"

)

print(

    "embodiment_v2_seed*_arm*.npz"

)

print()

print("=" * 90)

print("DONE")

print("=" * 90)
