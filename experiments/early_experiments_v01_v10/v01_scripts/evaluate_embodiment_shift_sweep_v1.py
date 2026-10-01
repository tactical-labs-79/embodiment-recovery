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

DAMPING_SCALES = [

    1.0,

    2.0,

    5.0,

    10.0,

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

            nn.Linear(HIDDEN, OUTPUT_DIM)

        )

    def forward(self, x):

        return self.network(x)

model = DynamicsModel().cpu()

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu"

)

if "model_state_dict" in checkpoint:

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint

state_dict = {

    k.replace("net.", "network.", 1): v

    for k, v in state_dict.items()

}

model.load_state_dict(state_dict)

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

    damping_scale,

    initial_qpos,

    initial_qvel,

    actions,

):

    env_a = make_env()

    env_b = make_env()

    np.random.seed(seed)

    torch.manual_seed(seed)

    env_a.reset()

    np.random.seed(seed)

    torch.manual_seed(seed)

    env_b.reset()

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

        env_b.robots[0].arm_joint_indexes[

            J4_ARM_INDEX

        ]

    )

    original_damping = float(

        env_b.sim.model.dof_damping[

            j4_dof

        ]

    )

    env_b.sim.model.dof_damping[

        j4_dof

    ] = (

        original_damping

        * damping_scale

    )

    modified_damping = float(

        env_b.sim.model.dof_damping[

            j4_dof

        ]

    )

    env_a.sim.forward()

    env_b.sim.forward()

    obs_a = env_a._get_observations(

        force_update=True

    )

    obs_b = env_b._get_observations(

        force_update=True

    )

    state_a = extract_state(

        obs_a

    )

    state_b = extract_state(

        obs_b

    )

    initial_difference = np.max(

        np.abs(

            state_a - state_b

        )

    )

    actual_a = []

    predicted_a = []

    actual_b = []

    predicted_b = []

    for step in range(

        ROLLOUT_STEPS

    ):

        action = actions[step]

        next_obs_a, _, _, _ = (

            env_a.step(action)

        )

        actual_next_a = extract_state(

            next_obs_a

        )

        next_obs_b, _, _, _ = (

            env_b.step(action)

        )

        actual_next_b = extract_state(

            next_obs_b

        )

        input_a = np.concatenate(

            [

                state_a,

                action

            ]

        ).astype(np.float32)

        with torch.no_grad():

            prediction_a = model(

                torch.from_numpy(

                    input_a

                ).unsqueeze(0)

            ).numpy()[0]

        input_b = np.concatenate(

            [

                state_b,

                action

            ]

        ).astype(np.float32)

        with torch.no_grad():

            prediction_b = model(

                torch.from_numpy(

                    input_b

                ).unsqueeze(0)

            ).numpy()[0]

        actual_a.append(

            actual_next_a.copy()

        )

        predicted_a.append(

            prediction_a.copy()

        )

        actual_b.append(

            actual_next_b.copy()

        )

        predicted_b.append(

            prediction_b.copy()

        )

        state_a = prediction_a.copy()

        state_b = prediction_b.copy()

    actual_a = np.asarray(

        actual_a,

        dtype=np.float32

    )

    predicted_a = np.asarray(

        predicted_a,

        dtype=np.float32

    )

    actual_b = np.asarray(

        actual_b,

        dtype=np.float32

    )

    predicted_b = np.asarray(

        predicted_b,

        dtype=np.float32

    )

    error_a = np.abs(

        predicted_a

        - actual_a

    )

    error_b = np.abs(

        predicted_b

        - actual_b

    )

    a_rollout = float(

        error_a.mean()

    )

    b_rollout = float(

        error_b.mean()

    )

    a_final = float(

        error_a[-1].mean()

    )

    b_final = float(

        error_b[-1].mean()

    )

    a_max = float(

        error_a.max()

    )

    b_max = float(

        error_b.max()

    )

    b_time_error = error_b.mean(

        axis=1

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

    b_dim_mae = error_b.mean(

        axis=0

    )

    env_a.close()

    env_b.close()

    return {

        "initial_difference":

            initial_difference,

        "original_damping":

            original_damping,

        "modified_damping":

            modified_damping,

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

        "b_dim_mae":

            b_dim_mae,

    }

print("=" * 90)

print("EMBODIMENT SHIFT SWEEP V1")

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

    "Damping scales:",

    DAMPING_SCALES

)

all_results = []

for seed in SEEDS:

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

    seed_results = []

    for scale in DAMPING_SCALES:

        result = run_experiment(

            seed=seed,

            damping_scale=scale,

            initial_qpos=initial_qpos,

            initial_qvel=initial_qvel,

            actions=actions,

        )

        result["seed"] = seed

        result["damping_scale"] = scale

        seed_results.append(

            result

        )

        all_results.append(

            result

        )

        print()

        print(

            f"J4 damping x{scale:.1f}"

        )

        print(

            f"  A rollout MAE : "

            f"{result['a_rollout']:.8f}"

        )

        print(

            f"  B rollout MAE : "

            f"{result['b_rollout']:.8f}"

        )

        print(

            f"  A final MAE   : "

            f"{result['a_final']:.8f}"

        )

        print(

            f"  B final MAE   : "

            f"{result['b_final']:.8f}"

        )

        print(

            f"  A max error   : "

            f"{result['a_max']:.8f}"

        )

        print(

            f"  B max error   : "

            f"{result['b_max']:.8f}"

        )

        print(

            f"  B >=0.10      : "

            f"{result['first_01']}"

        )

        print(

            f"  B >=0.20      : "

            f"{result['first_02']}"

        )

        print(

            f"  B >=0.50      : "

            f"{result['first_05']}"

        )

print()

print("=" * 90)

print("SWEEP SUMMARY")

print("=" * 90)

print()

header = (

    f"{'Scale':>8} | "

    f"{'A Rollout':>12} | "

    f"{'B Rollout':>12} | "

    f"{'Increase':>12} | "

    f"{'A Final':>12} | "

    f"{'B Final':>12} | "

    f"{'B Max':>12}"

)

print(header)

print("-" * len(header))

aggregate = []

for scale in DAMPING_SCALES:

    rows = [

        r for r in all_results

        if r["damping_scale"] == scale

    ]

    a_rollout = np.mean(

        [r["a_rollout"] for r in rows]

    )

    b_rollout = np.mean(

        [r["b_rollout"] for r in rows]

    )

    a_final = np.mean(

        [r["a_final"] for r in rows]

    )

    b_final = np.mean(

        [r["b_final"] for r in rows]

    )

    b_max = np.mean(

        [r["b_max"] for r in rows]

    )

    increase = (

        b_rollout

        - a_rollout

    )

    aggregate.append(

        {

            "scale": scale,

            "a_rollout": a_rollout,

            "b_rollout": b_rollout,

            "increase": increase,

            "a_final": a_final,

            "b_final": b_final,

            "b_max": b_max,

        }

    )

    print(

        f"{scale:8.1f} | "

        f"{a_rollout:12.8f} | "

        f"{b_rollout:12.8f} | "

        f"{increase:+12.8f} | "

        f"{a_final:12.8f} | "

        f"{b_final:12.8f} | "

        f"{b_max:12.8f}"

    )

print()

print("=" * 90)

print("FAILURE THRESHOLD SUMMARY")

print("=" * 90)

for scale in DAMPING_SCALES:

    rows = [

        r for r in all_results

        if r["damping_scale"] == scale

    ]

    print()

    print(

        f"J4 damping x{scale:.1f}"

    )

    for threshold, key in [

        (0.10, "first_01"),

        (0.20, "first_02"),

        (0.50, "first_05"),

    ]:

        values = np.asarray(

            [r[key] for r in rows]

        )

        valid = values[

            values >= 0

        ]

        print(

            f"  >= {threshold:.2f}: "

            f"{len(valid)} / {len(values)} seeds"

        )

        if len(valid) > 0:

            print(

                f"      mean first step : "

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

print("PER-DIMENSION ERROR BY SCALE")

print("=" * 90)

for scale in DAMPING_SCALES:

    rows = [

        r for r in all_results

        if r["damping_scale"] == scale

    ]

    matrix = np.asarray(

        [

            r["b_dim_mae"]

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

        f"J4 damping x{scale:.1f}"

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

    "embodiment_shift_sweep_v1_results.npz",

    scales=np.asarray(

        DAMPING_SCALES,

        dtype=np.float32

    ),

    seeds=np.asarray(

        SEEDS,

        dtype=np.int32

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

    b_dim_errors=np.asarray(

        [

            r["b_dim_mae"]

            for r in all_results

        ],

        dtype=np.float32

    ),

)

print()

print("=" * 90)

print("SAVED")

print("=" * 90)

print(

    "embodiment_shift_sweep_v1_results.npz"

)

print()

print("=" * 90)

print("DONE")

print("=" * 90)
