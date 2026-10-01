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

print("=" * 70)

print("V4 ROLLOUT COVERAGE ANALYSIS")

print("=" * 70)

print(f"Training states : {train_states.shape}")

assert train_states.ndim == 2

assert train_states.shape[1] == STATE_DIM

train_min = train_states.min(axis=0)

train_max = train_states.max(axis=0)

train_mean = train_states.mean(axis=0)

train_std = train_states.std(axis=0)

state_names = [

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

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu"

)

model = DynamicsModel()

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

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

        action = np.zeros(ACTION_DIM, dtype=np.float32)

        action_idx = np.random.randint(0, ACTION_DIM)

        magnitude = np.random.uniform(0.05, 0.20)

        sign = np.random.choice([-1.0, 1.0])

        action[action_idx] = magnitude * sign

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

            pred = model(

                torch.tensor(

                    model_input,

                    dtype=torch.float32

                ).unsqueeze(0)

            ).squeeze(0).numpy()

        seed_predicted.append(pred)

        seed_actual.append(actual_next)

        predicted_state = pred

    env.close()

    all_predicted.append(np.array(seed_predicted))

    all_actual.append(np.array(seed_actual))

predicted = np.array(all_predicted)

actual = np.array(all_actual)

print()

print(f"Predicted shape : {predicted.shape}")

print(f"Actual shape    : {actual.shape}")

print()

print("=" * 70)

print("1. TRAINING RANGE")

print("=" * 70)

important_dims = [

    3,

    5,

    10,

    12,

]

for dim in important_dims:

    pred_values = predicted[:, :, dim].reshape(-1)

    outside = (

        (pred_values < train_min[dim]) |

        (pred_values > train_max[dim])

    )

    percentage = outside.mean() * 100

    print()

    print(state_names[dim])

    print(f"Train min       : {train_min[dim]: .6f}")

    print(f"Train max       : {train_max[dim]: .6f}")

    print(f"Pred min        : {pred_values.min(): .6f}")

    print(f"Pred max        : {pred_values.max(): .6f}")

    print(f"Outside range   : {outside.sum()} / {len(pred_values)}")

    print(f"Outside percent : {percentage:.2f}%")

print()

print("=" * 70)

print("2. FIRST OUT-OF-RANGE TIMESTEP")

print("=" * 70)

for dim in important_dims:

    print()

    print(state_names[dim])

    found = False

    for step in range(STEPS):

        values = predicted[:, step, dim]

        outside = (

            (values < train_min[dim]) |

            (values > train_max[dim])

        )

        if np.any(outside):

            seeds_out = [

                SEEDS[i]

                for i, flag in enumerate(outside)

                if flag

            ]

            print(f"First step : {step + 1}")

            print(f"Seeds      : {seeds_out}")

            for i, flag in enumerate(outside):

                if flag:

                    print(

                        f"  Seed {SEEDS[i]}: "

                        f"{values[i]:.6f}"

                    )

            found = True

            break

    if not found:

        print("Never leaves training min/max range.")

print()

print("=" * 70)

print("3. TRAINING-DISTRIBUTION DISTANCE")

print("=" * 70)

focus_dims = [

    3,

    10,

    5,

    12,

]

focus_names = [

    "J4_Pos",

    "J4_Vel",

    "J6_Pos",

    "J6_Vel",

]

focus_train = train_states[:, focus_dims]

focus_mean = focus_train.mean(axis=0)

focus_std = focus_train.std(axis=0)

focus_std = np.maximum(focus_std, 1e-6)

train_z = (

    focus_train - focus_mean

) / focus_std

def nearest_distance(states):

    states_z = (

        states[:, focus_dims] - focus_mean

    ) / focus_std

    distances = []

    chunk_size = 100

    for start in range(0, len(states_z), chunk_size):

        chunk = states_z[

            start:start + chunk_size

        ]

        diff = (

            chunk[:, None, :] -

            train_z[None, :, :]

        )

        dist = np.sqrt(

            np.sum(diff * diff, axis=2)

        )

        nearest = dist.min(axis=1)

        distances.extend(nearest)

    return np.array(distances)

pred_flat = predicted.reshape(-1, STATE_DIM)

actual_flat = actual.reshape(-1, STATE_DIM)

pred_dist = nearest_distance(pred_flat)

actual_dist = nearest_distance(actual_flat)

print()

print("Nearest-neighbor distance in:")

print("[J4_Pos, J4_Vel, J6_Pos, J6_Vel]")

print()

print(f"Actual mean distance    : {actual_dist.mean():.6f}")

print(f"Predicted mean distance : {pred_dist.mean():.6f}")

print()

print(f"Actual median distance  : {np.median(actual_dist):.6f}")

print(f"Pred median distance    : {np.median(pred_dist):.6f}")

print()

print(f"Actual max distance     : {actual_dist.max():.6f}")

print(f"Predicted max distance  : {pred_dist.max():.6f}")

print()

print("=" * 70)

print("4. DISTANCE OVER TIME")

print("=" * 70)

pred_dist_time = pred_dist.reshape(

    len(SEEDS),

    STEPS

)

actual_dist_time = actual_dist.reshape(

    len(SEEDS),

    STEPS

)

check_steps = [

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

print()

print(

    f"{'Step':>6} "

    f"{'Actual NN':>15} "

    f"{'Pred NN':>15} "

    f"{'Ratio':>12}"

)

for step in check_steps:

    a = actual_dist_time[:, step - 1].mean()

    p = pred_dist_time[:, step - 1].mean()

    ratio = p / max(a, 1e-8)

    print(

        f"{step:6d} "

        f"{a:15.6f} "

        f"{p:15.6f} "

        f"{ratio:12.2f}x"

    )

print()

print("=" * 70)

print("5. FINAL STATE COVERAGE")

print("=" * 70)

for i, seed in enumerate(SEEDS):

    final = predicted[i, -1]

    print()

    print(f"Seed {seed}")

    for dim in important_dims:

        value = final[dim]

        if value < train_min[dim]:

            status = "BELOW"

        elif value > train_max[dim]:

            status = "ABOVE"

        else:

            status = "INSIDE"

        print(

            f"  {state_names[dim]:12s}: "

            f"{value: .6f} "

            f"[{status}]"

        )

print()

print("=" * 70)

print("DONE")

print("=" * 70)
