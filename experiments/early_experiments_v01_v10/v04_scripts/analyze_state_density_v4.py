import numpy as np

import torch

import robosuite as suite

DATA_FILE = "multi_trajectory_v4.npz"

MODEL_FILE = "dynamics_model_v4.pt"

SEEDS = [1001, 2002, 3003, 4004, 5005]

STEPS = 100

STATE_DIM = 17

ACTION_DIM = 7

INPUT_DIM = 24

OUTPUT_DIM = 17

FOCUS_DIMS = [3, 10, 5, 12]

FOCUS_NAMES = [

    "J4_Pos",

    "J4_Vel",

    "J6_Pos",

    "J6_Vel",

]

class DynamicsModel(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.network = torch.nn.Sequential(

            torch.nn.Linear(INPUT_DIM, 128),

            torch.nn.ReLU(),

            torch.nn.Linear(128, 128),

            torch.nn.ReLU(),

            torch.nn.Linear(128, OUTPUT_DIM),

        )

    def forward(self, x):

        return self.network(x)

data = np.load(DATA_FILE)

train_states = data["states"].astype(np.float32)

assert train_states.ndim == 2

assert train_states.shape[1] == STATE_DIM

print("=" * 75)

print("V4 STATE DENSITY ANALYSIS")

print("=" * 75)

print(f"Training states : {train_states.shape}")

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu"

)

model = DynamicsModel()

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

train_focus = train_states[:, FOCUS_DIMS]

focus_mean = train_focus.mean(axis=0)

focus_std = train_focus.std(axis=0)

focus_std = np.maximum(focus_std, 1e-6)

train_z = (

    train_focus - focus_mean

) / focus_std

all_predicted = []

all_actual = []

for seed in SEEDS:

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

    state = np.concatenate([

        obs["robot0_joint_pos"],

        obs["robot0_joint_vel"],

        obs["robot0_eef_pos"],

    ]).astype(np.float32)

    predicted_state = state.copy()

    seed_predicted = []

    seed_actual = []

    for step in range(STEPS):

        action = np.zeros(

            ACTION_DIM,

            dtype=np.float32

        )

        action_idx = np.random.randint(

            0,

            ACTION_DIM

        )

        magnitude = np.random.uniform(

            0.05,

            0.20

        )

        sign = np.random.choice(

            [-1.0, 1.0]

        )

        action[action_idx] = (

            magnitude * sign

        )

        obs_next, _, _, _ = env.step(action)

        actual_next = np.concatenate([

            obs_next["robot0_joint_pos"],

            obs_next["robot0_joint_vel"],

            obs_next["robot0_eef_pos"],

        ]).astype(np.float32)

        model_input = np.concatenate([

            predicted_state,

            action

        ])

        with torch.no_grad():

            prediction = model(

                torch.tensor(

                    model_input,

                    dtype=torch.float32

                ).unsqueeze(0)

            ).squeeze(0).numpy()

        seed_predicted.append(

            prediction

        )

        seed_actual.append(

            actual_next

        )

        predicted_state = prediction

    env.close()

    all_predicted.append(

        np.array(seed_predicted)

    )

    all_actual.append(

        np.array(seed_actual)

    )

predicted = np.array(all_predicted)

actual = np.array(all_actual)

print()

print(f"Predicted shape : {predicted.shape}")

print(f"Actual shape    : {actual.shape}")

def nearest_info(state):

    focus = state[FOCUS_DIMS]

    z = (

        focus - focus_mean

    ) / focus_std

    distances = np.sqrt(

        np.sum(

            (train_z - z) ** 2,

            axis=1

        )

    )

    indices = np.argsort(distances)[:10]

    return distances[indices], indices

pred_flat = predicted.reshape(

    -1,

    STATE_DIM

)

actual_flat = actual.reshape(

    -1,

    STATE_DIM

)

records = []

for flat_idx, state in enumerate(pred_flat):

    distances, indices = nearest_info(state)

    seed_idx = flat_idx // STEPS

    step_idx = flat_idx % STEPS

    records.append({

        "distance": distances[0],

        "flat_idx": flat_idx,

        "seed": SEEDS[seed_idx],

        "step": step_idx + 1,

        "state": state,

        "nearest_idx": indices[0],

    })

records.sort(

    key=lambda x: x["distance"],

    reverse=True

)

print()

print("=" * 75)

print("TOP 10 MOST DISTANT PREDICTED STATES")

print("=" * 75)

for rank, record in enumerate(

    records[:10],

    start=1

):

    state = record["state"]

    nearest_idx = record["nearest_idx"]

    nearest_state = train_states[nearest_idx]

    print()

    print(

        f"#{rank} "

        f"Seed {record['seed']} "

        f"Step {record['step']}"

    )

    print(

        f"NN distance : "

        f"{record['distance']:.6f}"

    )

    print()

    print(

        "Dimension          Predicted       "

        "Nearest Train      Difference"

    )

    for j, name in zip(

        FOCUS_DIMS,

        FOCUS_NAMES

    ):

        p = state[j]

        t = nearest_state[j]

        print(

            f"{name:16s} "

            f"{p:14.6f} "

            f"{t:16.6f} "

            f"{abs(p-t):13.6f}"

        )

print()

print("=" * 75)

print("ACTUAL VS PREDICTED DENSITY")

print("=" * 75)

for i, seed in enumerate(SEEDS):

    pred_distances = []

    actual_distances = []

    for step in range(STEPS):

        pd, _ = nearest_info(

            predicted[i, step]

        )

        ad, _ = nearest_info(

            actual[i, step]

        )

        pred_distances.append(

            pd[0]

        )

        actual_distances.append(

            ad[0]

        )

    print()

    print(f"Seed {seed}")

    print(

        f"  Actual mean NN : "

        f"{np.mean(actual_distances):.6f}"

    )

    print(

        f"  Pred mean NN   : "

        f"{np.mean(pred_distances):.6f}"

    )

    print(

        f"  Actual max NN  : "

        f"{np.max(actual_distances):.6f}"

    )

    print(

        f"  Pred max NN    : "

        f"{np.max(pred_distances):.6f}"

    )

print()

print("=" * 75)

print("DENSITY BY TIME")

print("=" * 75)

check_steps = [

    1,

    5,

    10,

    15,

    20,

    25,

    30,

    40,

    50,

    60,

    70,

    80,

    90,

    100,

]

print()

print(

    f"{'Step':>6} "

    f"{'Actual NN':>15} "

    f"{'Pred NN':>15}"

)

for step in check_steps:

    actual_distances = []

    pred_distances = []

    for i in range(len(SEEDS)):

        ad, _ = nearest_info(

            actual[i, step - 1]

        )

        pd, _ = nearest_info(

            predicted[i, step - 1]

        )

        actual_distances.append(

            ad[0]

        )

        pred_distances.append(

            pd[0]

        )

    print(

        f"{step:6d} "

        f"{np.mean(actual_distances):15.6f} "

        f"{np.mean(pred_distances):15.6f}"

    )

print()

print("=" * 75)

print("PREDICTED STATES BY DISTANCE THRESHOLD")

print("=" * 75)

all_distances = np.array([

    r["distance"]

    for r in records

])

for threshold in [

    0.25,

    0.50,

    1.00,

    1.50,

    2.00,

    3.00,

    4.00,

]:

    count = np.sum(

        all_distances >= threshold

    )

    percentage = (

        count /

        len(all_distances) *

        100

    )

    print(

        f"Distance >= {threshold:.2f} : "

        f"{count:4d} / {len(all_distances)} "

        f"({percentage:6.2f}%)"

    )

print()

print("=" * 75)

print("DONE")

print("=" * 75)
