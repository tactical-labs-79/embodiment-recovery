import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

MODEL_FILE = "dynamics_model_v4.pt"

TEST_SEEDS = [1001, 2002, 3003, 4004, 5005]

STEPS = 100

STATE_NAMES = [

    "Joint1_Pos",

    "Joint2_Pos",

    "Joint3_Pos",

    "Joint4_Pos",

    "Joint5_Pos",

    "Joint6_Pos",

    "Joint7_Pos",

    "Joint1_Vel",

    "Joint2_Vel",

    "Joint3_Vel",

    "Joint4_Vel",

    "Joint5_Vel",

    "Joint6_Vel",

    "Joint7_Vel",

    "EEF_X",

    "EEF_Y",

    "EEF_Z",

]

def get_state(obs):

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

    state = np.concatenate([

        joint_pos,

        joint_vel,

        eef_pos

    ]).astype(np.float32)

    if state.shape != (17,):

        raise ValueError(

            f"State dimension salah: {state.shape}"

        )

    return state

class DynamicsModelV4(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(24, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Linear(128, 17)

        )

    def forward(self, x):

        return self.network(x)

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu",

    weights_only=True

)

model = DynamicsModelV4()

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

all_errors = []

all_step_mae = []

worst_global = {

    "error": -1.0,

    "seed": None,

    "step": None,

    "dimension": None,

    "actual": None,

    "predicted": None,

}

print("=" * 70)

print("V4 ROLLOUT DIAGNOSTIC")

print("=" * 70)

for seed in TEST_SEEDS:

    env = suite.make(

        env_name="Lift",

        robots="Panda",

        has_renderer=False,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        control_freq=20,

    )

    np.random.seed(seed)

    obs = env.reset()

    real_state = get_state(obs)

    predicted_state = real_state.copy()

    seed_errors = []

    for step in range(STEPS):

        action = np.zeros(7, dtype=np.float32)

        action_index = np.random.randint(0, 7)

        action_sign = np.random.choice([-1.0, 1.0])

        action_magnitude = np.random.uniform(0.05, 0.20)

        action[action_index] = (

            action_sign * action_magnitude

        )

        next_obs, reward, done, info = env.step(action)

        real_next_state = get_state(next_obs)

        model_input = np.concatenate([

            predicted_state,

            action

        ]).astype(np.float32)

        with torch.no_grad():

            predicted_next_state = (

                model(

                    torch.from_numpy(

                        model_input

                    ).unsqueeze(0)

                )

                .squeeze(0)

                .numpy()

            )

        error = np.abs(

            predicted_next_state - real_next_state

        )

        seed_errors.append(error)

        local_idx = np.argmax(error)

        local_error = float(error[local_idx])

        if local_error > worst_global["error"]:

            worst_global = {

                "error": local_error,

                "seed": seed,

                "step": step + 1,

                "dimension": local_idx,

                "actual": float(real_next_state[local_idx]),

                "predicted": float(predicted_next_state[local_idx]),

            }

        real_state = real_next_state

        predicted_state = predicted_next_state

    env.close()

    seed_errors = np.asarray(seed_errors)

    all_errors.append(seed_errors)

    all_step_mae.append(seed_errors.mean(axis=1))

    print(

        f"Seed {seed} selesai | "

        f"MAE: {seed_errors.mean():.8f}"

    )

all_errors = np.asarray(all_errors)

dimension_mae = all_errors.mean(axis=(0, 1))

print()

print("=" * 70)

print("AVERAGE ERROR PER STATE DIMENSION")

print("=" * 70)

for i, name in enumerate(STATE_NAMES):

    print(

        f"{i:02d} | "

        f"{name:<12} | "

        f"MAE: {dimension_mae[i]:.8f}"

    )

ranking = np.argsort(

    dimension_mae

)[::-1]

print()

print("=" * 70)

print("WORST DIMENSIONS")

print("=" * 70)

for rank, idx in enumerate(ranking[:10], start=1):

    print(

        f"{rank:02d}. "

        f"{STATE_NAMES[idx]:<12} "

        f"MAE: {dimension_mae[idx]:.8f}"

    )

all_step_mae = np.asarray(all_step_mae)

time_mae = all_step_mae.mean(axis=0)

print()

print("=" * 70)

print("ERROR BY TIME")

print("=" * 70)

checkpoints = [

    1,

    5,

    10,

    20,

    30,

    40,

    50,

    60,

    70,

    80,

    90,

    100,

]

for step in checkpoints:

    print(

        f"Step {step:03d} | "

        f"MAE: {time_mae[step - 1]:.8f}"

    )

worst_steps = np.argsort(

    time_mae

)[::-1][:10]

print()

print("=" * 70)

print("WORST TIMESTEPS")

print("=" * 70)

for step_idx in worst_steps:

    print(

        f"Step {step_idx + 1:03d} | "

        f"MAE: {time_mae[step_idx]:.8f}"

    )

print()

print("=" * 70)

print("GLOBAL WORST ERROR")

print("=" * 70)

idx = worst_global["dimension"]

print(f"Seed      : {worst_global['seed']}")

print(f"Step      : {worst_global['step']}")

print(f"Dimension : {STATE_NAMES[idx]}")

print(f"Error     : {worst_global['error']:.8f}")

print(f"Actual    : {worst_global['actual']:.8f}")

print(f"Predicted : {worst_global['predicted']:.8f}")

print("=" * 70)
