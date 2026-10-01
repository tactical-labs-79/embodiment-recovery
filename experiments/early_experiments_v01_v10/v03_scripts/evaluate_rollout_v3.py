import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

MODEL_FILE = "dynamics_model_v3.pt"

SEEDS = [

    1001,

    2002,

    3003,

    4004,

    5005

]

STEPS = 100

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(17, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Linear(128, 10)

        )

    def forward(self, x):

        return self.network(x)

def get_state(obs):

    joint_pos = np.asarray(

        obs["robot0_joint_pos"],

        dtype=np.float32

    )

    eef_pos = np.asarray(

        obs["robot0_eef_pos"],

        dtype=np.float32

    )

    return np.concatenate(

        [joint_pos, eef_pos]

    ).astype(np.float32)

model = DynamicsModel()

model.load_state_dict(

    torch.load(

        MODEL_FILE,

        map_location="cpu"

    )

)

model.eval()

def evaluate_seed(seed):

    np.random.seed(seed)

    env = suite.make(

        env_name="Lift",

        robots="Panda",

        has_renderer=False,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        control_freq=20,

    )

    obs = env.reset()

    rng = np.random.default_rng(seed)

    real_state = get_state(obs)

    predicted_state = real_state.copy()

    rollout_errors = []

    for step in range(STEPS):

        action = np.zeros(7, dtype=np.float32)

        action_idx = rng.integers(0, 7)

        direction = rng.choice([-1.0, 1.0])

        magnitude = rng.uniform(0.05, 0.20)

        action[action_idx] = direction * magnitude

        next_obs, reward, done, info = env.step(action)

        real_next_state = get_state(next_obs)

        model_input = np.concatenate(

            [predicted_state, action]

        ).astype(np.float32)

        model_input_tensor = torch.tensor(

            model_input,

            dtype=torch.float32

        ).unsqueeze(0)

        with torch.no_grad():

            predicted_next_state = (

                model(model_input_tensor)

                .squeeze(0)

                .numpy()

            )

        error = np.abs(

            predicted_next_state - real_next_state

        )

        rollout_errors.append(error)

        predicted_state = predicted_next_state

        real_state = real_next_state

        obs = next_obs

        if done:

            break

    env.close()

    rollout_errors = np.asarray(

        rollout_errors,

        dtype=np.float32

    )

    overall_mae = rollout_errors.mean()

    final_mae = rollout_errors[-1].mean()

    max_error = rollout_errors.max()

    return (

        overall_mae,

        final_mae,

        max_error,

        rollout_errors

    )

all_overall = []

all_final = []

all_max = []

print("=" * 60)

print("DYNAMICS V3 — MULTI-STEP ROLLOUT")

print("=" * 60)

print(f"Rollout steps : {STEPS}")

for seed in SEEDS:

    print(f"\nTesting seed {seed}...")

    (

        overall_mae,

        final_mae,

        max_error,

        errors

    ) = evaluate_seed(seed)

    all_overall.append(overall_mae)

    all_final.append(final_mae)

    all_max.append(max_error)

    print(

        f"Seed {seed}: "

        f"Rollout MAE = {overall_mae:.8f} | "

        f"Final MAE = {final_mae:.8f} | "

        f"Max Error = {max_error:.8f}"

    )

all_overall = np.asarray(all_overall)

all_final = np.asarray(all_final)

all_max = np.asarray(all_max)

print("\n" + "=" * 60)

print("ROLLOUT SUMMARY")

print("=" * 60)

for i, seed in enumerate(SEEDS):

    print(

        f"Seed {seed}: "

        f"MAE={all_overall[i]:.8f} | "

        f"Final={all_final[i]:.8f} | "

        f"Max={all_max[i]:.8f}"

    )

print("\n" + "-" * 60)

print(

    f"Average Rollout MAE : "

    f"{all_overall.mean():.8f}"

)

print(

    f"Average Final MAE   : "

    f"{all_final.mean():.8f}"

)

print(

    f"Average Max Error   : "

    f"{all_max.mean():.8f}"

)

print(

    f"Std Rollout MAE     : "

    f"{all_overall.std():.8f}"

)

print("=" * 60)

print("ROLLOUT TEST COMPLETE")

print("=" * 60)
