import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

MODEL_FILE = "dynamics_model_v6.pt"

TEST_SEEDS = [1001, 2002, 3003, 4004, 5005]

STEPS = 400

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

            f"State dimension salah: {state.shape}, expected (17,)"

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

state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint; state_dict = {k.replace("net.", "network.", 1): v for k, v in state_dict.items()}; model.load_state_dict(state_dict)

model.eval()

assert checkpoint["input_dim"] == 24

assert checkpoint["output_dim"] == 17

print("=" * 60)

print("V4 MULTI-SEED ONE-STEP EVALUATION")

print("=" * 60)

print("Model : dynamics_model_v6.pt")

print("Input : 24")

print("Output: 17")

print()

all_mae = []

all_max = []

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

    state = get_state(obs)

    errors = []

    for step in range(STEPS):

        action = np.zeros(7, dtype=np.float32)

        action_index = np.random.randint(0, 7)

        action_sign = np.random.choice([-1.0, 1.0])

        action_magnitude = np.random.uniform(0.05, 0.20)

        action[action_index] = (

            action_sign * action_magnitude

        )

        next_obs, reward, done, info = env.step(action)

        actual_next_state = get_state(next_obs)

        model_input = np.concatenate(

            [state, action]

        ).astype(np.float32)

        if model_input.shape != (24,):

            raise ValueError(

                f"Input dimension salah: "

                f"{model_input.shape}, expected (24,)"

            )

        with torch.no_grad():

            x = torch.from_numpy(

                model_input

            ).unsqueeze(0)

            prediction = model(x).squeeze(0).numpy()

        if prediction.shape != (17,):

            raise ValueError(

                f"Output dimension salah: "

                f"{prediction.shape}, expected (17,)"

            )

        error = np.abs(

            prediction - actual_next_state

        )

        errors.append(error)

        state = actual_next_state

    env.close()

    errors = np.asarray(errors)

    mae = errors.mean()

    max_error = errors.max()

    all_mae.append(mae)

    all_max.append(max_error)

    print(

        f"Seed {seed} | "

        f"MAE: {mae:.8f} | "

        f"Max: {max_error:.8f}"

    )

all_mae = np.asarray(all_mae)

all_max = np.asarray(all_max)

print()

print("=" * 60)

print("V4 SUMMARY")

print("=" * 60)

print(f"Average MAE : {all_mae.mean():.8f}")

print(f"Std MAE     : {all_mae.std():.8f}")

print(f"Min MAE     : {all_mae.min():.8f}")

print(f"Max MAE     : {all_mae.max():.8f}")

print(f"Average Max : {all_max.mean():.8f}")

print("=" * 60)
