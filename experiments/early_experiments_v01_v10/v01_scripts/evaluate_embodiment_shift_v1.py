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

J4_DAMPING_SCALE = 2.0

J4_POS = 3

J6_POS = 5

J7_POS = 6

J4_VEL = 10

J6_VEL = 12

J7_VEL = 13

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

a_rollout_maes = []

b_rollout_maes = []

a_final_maes = []

b_final_maes = []

a_max_errors = []

b_max_errors = []

b_first_01 = []

b_first_02 = []

b_first_05 = []

b_dim_errors = []

results = []

print("=" * 80)

print("EMBODIMENT SHIFT V1")

print("=" * 80)

print()

print(

    f"Model : {MODEL_FILE}"

)

print(

    f"Seeds : {SEEDS}"

)

print(

    f"Horizon : {ROLLOUT_STEPS}"

)

print()

print(

    "Robot A : normal dynamics"

)

print(

    "Robot B : J4 damping x "

    f"{J4_DAMPING_SCALE:.2f}"

)

for seed in SEEDS:

    env_a = make_env()

    env_b = make_env()

    np.random.seed(seed)

    torch.manual_seed(seed)

    obs_a = env_a.reset()

    np.random.seed(seed)

    torch.manual_seed(seed)

    obs_b = env_b.reset()

    env_b.sim.data.qpos[:] = (

        env_a.sim.data.qpos[:]

    )

    env_b.sim.data.qvel[:] = (

        env_a.sim.data.qvel[:]

    )

    j4_dof = int(

        env_b.robots[0].arm_joint_indexes[3]

    )

    original_damping = float(

        env_b.sim.model.dof_damping[j4_dof]

    )

    env_b.sim.model.dof_damping[j4_dof] = (

        original_damping

        * J4_DAMPING_SCALE

    )

    modified_damping = float(

        env_b.sim.model.dof_damping[j4_dof]

    )

    env_a.sim.forward()

    env_b.sim.forward()

    obs_a = env_a._get_observations(

        force_update=True

    )

    obs_b = env_b._get_observations(

        force_update=True

    )

    state_v7_a = extract_state(

        obs_a

    )

    state_v7_b = extract_state(

        obs_b

    )

    initial_difference = np.max(

        np.abs(

            state_v7_a

            - state_v7_b

        )

    )

    actual_a = []

    predicted_a = []

    actual_b = []

    predicted_b = []

    actions = []

    for step in range(

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

            action.copy()

        )

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

                state_v7_a,

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

                state_v7_b,

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

        state_v7_a = prediction_a.copy()

        state_v7_b = prediction_b.copy()

        obs_a = next_obs_a

        obs_b = next_obs_b

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

    a_rollout = error_a.mean()

    b_rollout = error_b.mean()

    a_final = error_a[-1].mean()

    b_final = error_b[-1].mean()

    a_max = error_a.max()

    b_max = error_b.max()

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

        return int(hits[0] + 1)

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

    b_dim_errors.append(

        b_dim_mae

    )

    a_rollout_maes.append(

        a_rollout

    )

    b_rollout_maes.append(

        b_rollout

    )

    a_final_maes.append(

        a_final

    )

    b_final_maes.append(

        b_final

    )

    a_max_errors.append(

        a_max

    )

    b_max_errors.append(

        b_max

    )

    b_first_01.append(

        first_01

    )

    b_first_02.append(

        first_02

    )

    b_first_05.append(

        first_05

    )

    results.append(

        {

            "seed": seed,

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

        }

    )

    print()

    print(

        f"Seed {seed}"

    )

    print(

        f"  J4 damping : "

        f"{original_damping:.8f}"

        f" -> "

        f"{modified_damping:.8f}"

    )

    print(

        f"  Initial A/B difference : "

        f"{initial_difference:.10f}"

    )

    print(

        f"  Robot A rollout MAE : "

        f"{a_rollout:.8f}"

    )

    print(

        f"  Robot B rollout MAE : "

        f"{b_rollout:.8f}"

    )

    print(

        f"  Robot A final MAE   : "

        f"{a_final:.8f}"

    )

    print(

        f"  Robot B final MAE   : "

        f"{b_final:.8f}"

    )

    print(

        f"  Robot A max error   : "

        f"{a_max:.8f}"

    )

    print(

        f"  Robot B max error   : "

        f"{b_max:.8f}"

    )

    print(

        f"  B first >=0.10      : "

        f"{first_01}"

    )

    print(

        f"  B first >=0.20      : "

        f"{first_02}"

    )

    print(

        f"  B first >=0.50      : "

        f"{first_05}"

    )

    env_a.close()

    env_b.close()

a_rollout_maes = np.asarray(

    a_rollout_maes

)

b_rollout_maes = np.asarray(

    b_rollout_maes

)

a_final_maes = np.asarray(

    a_final_maes

)

b_final_maes = np.asarray(

    b_final_maes

)

a_max_errors = np.asarray(

    a_max_errors

)

b_max_errors = np.asarray(

    b_max_errors

)

b_dim_errors = np.asarray(

    b_dim_errors

)

print()

print("=" * 80)

print("EMBODIMENT SHIFT V1 SUMMARY")

print("=" * 80)

print()

print("ROBOT A — TRAINING EMBODIMENT")

print(

    f"  Rollout MAE : "

    f"{a_rollout_maes.mean():.8f}"

)

print(

    f"  Final MAE   : "

    f"{a_final_maes.mean():.8f}"

)

print(

    f"  Max error   : "

    f"{a_max_errors.mean():.8f}"

)

print()

print("ROBOT B — SHIFTED EMBODIMENT")

print(

    f"  Rollout MAE : "

    f"{b_rollout_maes.mean():.8f}"

)

print(

    f"  Final MAE   : "

    f"{b_final_maes.mean():.8f}"

)

print(

    f"  Max error   : "

    f"{b_max_errors.mean():.8f}"

)

print()

print("SHIFT PENALTY")

print(

    f"  Rollout increase : "

    f"{(

        b_rollout_maes.mean()

        - a_rollout_maes.mean()

    ):+.8f}"

)

print(

    f"  Final increase   : "

    f"{(

        b_final_maes.mean()

        - a_final_maes.mean()

    ):+.8f}"

)

print(

    f"  Max increase     : "

    f"{(

        b_max_errors.mean()

        - a_max_errors.mean()

    ):+.8f}"

)

print()

print("=" * 80)

print("ROBOT B FAILURE TIMING")

print("=" * 80)

valid_01 = np.asarray(

    b_first_01

)

valid_02 = np.asarray(

    b_first_02

)

valid_05 = np.asarray(

    b_first_05

)

print()

print(

    "First B >= 0.10:"

)

print(

    f"  Seeds reached : "

    f"{(valid_01 >= 0).sum()} / {len(SEEDS)}"

)

if np.any(valid_01 >= 0):

    print(

        f"  Mean step     : "

        f"{valid_01[valid_01 >= 0].mean():.2f}"

    )

print()

print(

    "First B >= 0.20:"

)

print(

    f"  Seeds reached : "

    f"{(valid_02 >= 0).sum()} / {len(SEEDS)}"

)

if np.any(valid_02 >= 0):

    print(

        f"  Mean step     : "

        f"{valid_02[valid_02 >= 0].mean():.2f}"

    )

print()

print(

    "First B >= 0.50:"

)

print(

    f"  Seeds reached : "

    f"{(valid_05 >= 0).sum()} / {len(SEEDS)}"

)

if np.any(valid_05 >= 0):

    print(

        f"  Mean step     : "

        f"{valid_05[valid_05 >= 0].mean():.2f}"

    )

print()

print("=" * 80)

print("ROBOT B PER-DIMENSION ERROR")

print("=" * 80)

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

global_dim_mae = b_dim_errors.mean(

    axis=0

)

sorted_dims = np.argsort(

    global_dim_mae

)[::-1]

print()

for rank, dim in enumerate(

    sorted_dims,

    1

):

    print(

        f"{rank:2d}. "

        f"{dim_names[dim]:8s} "

        f"MAE = "

        f"{global_dim_mae[dim]:.8f}"

    )

np.savez(

    "embodiment_shift_v1_results.npz",

    seeds=np.asarray(

        SEEDS,

        dtype=np.int32

    ),

    a_rollout=a_rollout_maes,

    b_rollout=b_rollout_maes,

    a_final=a_final_maes,

    b_final=b_final_maes,

    a_max=a_max_errors,

    b_max=b_max_errors,

    b_first_01=valid_01,

    b_first_02=valid_02,

    b_first_05=valid_05,

    b_dim_errors=b_dim_errors,

)

print()

print("=" * 80)

print("SAVED")

print("=" * 80)

print(

    "embodiment_shift_v1_results.npz"

)

print()

print("=" * 80)

print("DONE")

print("=" * 80)
