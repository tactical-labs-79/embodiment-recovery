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

STEPS = 400

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

    states = []

    actions = []

    next_states = []

    for step in range(STEPS):

        state = get_state(obs)

        action = np.zeros(7, dtype=np.float32)

        action_idx = rng.integers(0, 7)

        direction = rng.choice([-1.0, 1.0])

        magnitude = rng.uniform(0.05, 0.20)

        action[action_idx] = direction * magnitude

        next_obs, reward, done, info = env.step(action)

        next_state = get_state(next_obs)

        states.append(state)

        actions.append(action)

        next_states.append(next_state)

        obs = next_obs

        if done:

            break

    env.close()

    states = np.asarray(states, dtype=np.float32)

    actions = np.asarray(actions, dtype=np.float32)

    next_states = np.asarray(next_states, dtype=np.float32)

    assert states.ndim == 2

    assert actions.ndim == 2

    assert next_states.ndim == 2

    assert states.shape[1] == 10

    assert actions.shape[1] == 7

    assert next_states.shape[1] == 10

    assert states.shape[0] == actions.shape[0]

    assert states.shape[0] == next_states.shape[0]

    X = np.concatenate(

        [states, actions],

        axis=1

    )

    assert X.shape[1] == 17

    X_tensor = torch.tensor(

        X,

        dtype=torch.float32

    )

    with torch.no_grad():

        predictions = model(X_tensor).numpy()

    assert predictions.shape == next_states.shape

    assert predictions.shape[1] == 10

    errors = np.abs(

        predictions - next_states

    )

    mae = errors.mean()

    max_error = errors.max()

    return mae, max_error

results = []

print("=" * 60)

print("DYNAMICS V3 — MULTI SEED EVALUATION")

print("=" * 60)

for seed in SEEDS:

    print(f"\nTesting seed {seed}...")

    mae, max_error = evaluate_seed(seed)

    results.append(mae)

    print(

        f"Seed {seed}: "

        f"MAE = {mae:.8f} | "

        f"Max Error = {max_error:.8f}"

    )

results = np.asarray(results)

print("\n" + "=" * 60)

print("SUMMARY")

print("=" * 60)

for seed, mae in zip(SEEDS, results):

    print(

        f"Seed {seed}: "

        f"MAE = {mae:.8f}"

    )

print("\n" + "-" * 60)

print(

    f"Average MAE : {results.mean():.8f}"

)

print(

    f"Std MAE     : {results.std():.8f}"

)

print(

    f"Min MAE     : {results.min():.8f}"

)

print(

    f"Max MAE     : {results.max():.8f}"

)

print("=" * 60)

print("EVALUATION COMPLETE")

print("=" * 60)
