import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

STEPS = 400

SEED = 2002

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

model = DynamicsModel()

checkpoint = torch.load(

    "dynamics_model_v2.pt",

    map_location="cpu",

    weights_only=True

)

model.load_state_dict(checkpoint)

model.eval()

rng = np.random.default_rng(SEED)

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=True,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    control_freq=20,

)

obs = env.reset()

states = []

actions = []

for step in range(STEPS):

    action = np.zeros(7, dtype=np.float32)

    joint = rng.integers(0, 4)

    direction = rng.choice([-1.0, 1.0])

    magnitude = rng.uniform(0.05, 0.20)

    action[joint] = direction * magnitude

    state = np.concatenate([

        obs["robot0_joint_pos"],

        obs["robot0_eef_pos"]

    ]).astype(np.float32)

    states.append(state)

    actions.append(action.copy())

    obs, reward, done, info = env.step(action)

    env.render()

    if done:

        break

env.close()

states = np.array(states, dtype=np.float32)

actions = np.array(actions, dtype=np.float32)

X = np.concatenate(

    [states[:-1], actions[:-1]],

    axis=1

)

y = states[1:]

X_tensor = torch.tensor(

    X,

    dtype=torch.float32

)

with torch.no_grad():

    predictions = model(X_tensor).numpy()

errors = np.abs(predictions - y)

error_per_step = errors.mean(axis=1)

worst_indices = np.argsort(

    error_per_step

)[-10:][::-1]

print("\n========================================")

print("V2 FAILURE ANALYSIS")

print("========================================")

print(f"Seed  : {SEED}")

print(f"Steps : {len(states)}")

print("\n========================================")

print("TOP 10 WORST TIMESTEPS")

print("========================================")

for rank, idx in enumerate(worst_indices, start=1):

    print("\n----------------------------------------")

    print(f"#{rank} | Timestep: {idx}")

    print("----------------------------------------")

    print(

        f"Overall MAE: "

        f"{error_per_step[idx]:.8f}"

    )

    print("\nAction:")

    print(

        np.array2string(

            actions[idx],

            precision=6,

            suppress_small=True

        )

    )

    print("\nState error:")

    names = [

        "Joint 1",

        "Joint 2",

        "Joint 3",

        "Joint 4",

        "Joint 5",

        "Joint 6",

        "Joint 7",

        "EEF X",

        "EEF Y",

        "EEF Z",

    ]

    for i, name in enumerate(names):

        print(

            f"{name:8s}: "

            f"{errors[idx, i]:.8f}"

        )

mean_error_per_state = errors.mean(axis=0)

print("\n========================================")

print("AVERAGE ERROR PER STATE")

print("========================================")

for i, name in enumerate(names):

    print(

        f"{name:8s}: "

        f"{mean_error_per_state[i]:.8f}"

    )

max_index = np.unravel_index(

    np.argmax(errors),

    errors.shape

)

step_idx = max_index[0]

state_idx = max_index[1]

print("\n========================================")

print("ABSOLUTE WORST ERROR")

print("========================================")

print(f"Timestep : {step_idx}")

print(f"State    : {names[state_idx]}")

print(f"Error    : {errors[step_idx, state_idx]:.8f}")

print("\nActual:")

print(f"{y[step_idx, state_idx]:.8f}")

print("Predicted:")

print(f"{predictions[step_idx, state_idx]:.8f}")
