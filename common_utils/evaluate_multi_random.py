import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

STEPS = 400

SEEDS = [1001, 2002, 3003, 4004, 5005]

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

def load_model(path):

    model = DynamicsModel()

    checkpoint = torch.load(

        path,

        map_location="cpu",

        weights_only=True

    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model

v1 = load_model("dynamics_model.pt")

v2 = load_model("dynamics_model_v2.pt")

def collect_test(seed):

    rng = np.random.default_rng(seed)

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

    return (

        np.array(states, dtype=np.float32),

        np.array(actions, dtype=np.float32)

    )

def evaluate(model, states, actions):

    X = np.concatenate(

        [states[:-1], actions[:-1]],

        axis=1

    )

    y = states[1:]

    X = torch.tensor(

        X,

        dtype=torch.float32

    )

    with torch.no_grad():

        prediction = model(X).numpy()

    error = np.abs(prediction - y)

    mae = error.mean()

    max_error = error.max()

    return mae, max_error

v1_results = []

v2_results = []

print("\n========================================")

print("MULTI-SEED GENERALIZATION TEST")

print("========================================")

print(f"Seeds : {SEEDS}")

print(f"Steps : {STEPS}")

for seed in SEEDS:

    print("\n----------------------------------------")

    print(f"Testing seed: {seed}")

    print("----------------------------------------")

    states, actions = collect_test(seed)

    v1_mae, v1_max = evaluate(

        v1,

        states,

        actions

    )

    v2_mae, v2_max = evaluate(

        v2,

        states,

        actions

    )

    v1_results.append(v1_mae)

    v2_results.append(v2_mae)

    print(f"V1 | MAE: {v1_mae:.8f} | Max: {v1_max:.8f}")

    print(f"V2 | MAE: {v2_mae:.8f} | Max: {v2_max:.8f}")

print("\n========================================")

print("FINAL SUMMARY")

print("========================================")

print("\nV1 MAE per seed:")

for seed, value in zip(SEEDS, v1_results):

    print(f"Seed {seed}: {value:.8f}")

print("\nV2 MAE per seed:")

for seed, value in zip(SEEDS, v2_results):

    print(f"Seed {seed}: {value:.8f}")

print("\n========================================")

print("AVERAGE")

print("========================================")

print(f"V1 Average MAE: {np.mean(v1_results):.8f}")

print(f"V2 Average MAE: {np.mean(v2_results):.8f}")

print(f"V1 Std MAE    : {np.std(v1_results):.8f}")

print(f"V2 Std MAE    : {np.std(v2_results):.8f}")
