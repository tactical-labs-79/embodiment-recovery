import numpy as np

import torch

import robosuite as suite

MODEL_FILE = "dynamics_model_v4.pt"

SEEDS = [1001, 2002, 3003, 4004, 5005]

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

class DynamicsModel(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.network = torch.nn.Sequential(

            torch.nn.Linear(24, 128),

            torch.nn.ReLU(),

            torch.nn.Linear(128, 128),

            torch.nn.ReLU(),

            torch.nn.Linear(128, 17)

        )

    def forward(self, x):

        return self.network(x)

print("=" * 70)

print("LOADING V4 MODEL")

print("=" * 70)

model = DynamicsModel()

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu"

)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    print("Checkpoint format : dictionary")

    print("State dim         :", checkpoint.get("state_dim", "?"))

    print("Action dim        :", checkpoint.get("action_dim", "?"))

    print("Input dim         :", checkpoint.get("input_dim", "?"))

    print("Output dim        :", checkpoint.get("output_dim", "?"))

else:

    model.load_state_dict(checkpoint)

    print("Checkpoint format : raw state_dict")

model.eval()

print("Model loaded successfully.")

print()

all_errors = []

all_actual = []

all_predicted = []

print("=" * 70)

print("RUNNING ROLLOUTS")

print("=" * 70)

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

    seed_errors = []

    seed_actual = []

    seed_predicted = []

    for step in range(STEPS):

        action = np.zeros(7, dtype=np.float32)

        idx = np.random.randint(0, 7)

        sign = np.random.choice(

            [-1.0, 1.0]

        )

        magnitude = np.random.uniform(

            0.05,

            0.20

        )

        action[idx] = sign * magnitude

        obs, reward, done, info = env.step(action)

        actual_next = np.concatenate([

            obs["robot0_joint_pos"],

            obs["robot0_joint_vel"],

            obs["robot0_eef_pos"],

        ]).astype(np.float32)

        model_input = np.concatenate([

            predicted_state,

            action

        ]).astype(np.float32)

        model_input_tensor = torch.tensor(

            model_input,

            dtype=torch.float32

        ).unsqueeze(0)

        with torch.no_grad():

            prediction = model(

                model_input_tensor

            ).numpy()[0]

        error = np.abs(

            prediction - actual_next

        )

        seed_errors.append(error)

        seed_actual.append(actual_next)

        seed_predicted.append(prediction)

        predicted_state = prediction

    env.close()

    seed_errors = np.array(seed_errors)

    seed_actual = np.array(seed_actual)

    seed_predicted = np.array(seed_predicted)

    all_errors.append(seed_errors)

    all_actual.append(seed_actual)

    all_predicted.append(seed_predicted)

    print(

        f"Seed {seed} selesai | "

        f"MAE: {seed_errors.mean():.8f}"

    )

all_errors = np.array(all_errors)

all_actual = np.array(all_actual)

all_predicted = np.array(all_predicted)

print()

print("Error array shape     :", all_errors.shape)

print("Actual array shape    :", all_actual.shape)

print("Predicted array shape :", all_predicted.shape)

print("\n" + "=" * 70)

print("JOINT4 & JOINT6 DRIFT")

print("=" * 70)

targets = {

    "Joint4_Pos": 3,

    "Joint6_Pos": 5,

    "Joint4_Vel": 10,

    "Joint6_Vel": 12,

}

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

    100

]

for name, idx in targets.items():

    print(f"\n{name}")

    for step in checkpoints:

        error = all_errors[

            :,

            step - 1,

            idx

        ].mean()

        print(

            f"Step {step:03d} | "

            f"MAE: {error:.8f}"

        )

print("\n" + "=" * 70)

print("FINAL STATE: ACTUAL vs PREDICTED")

print("=" * 70)

for idx in [3, 5, 10, 12]:

    actual = all_actual[

        :,

        -1,

        idx

    ]

    predicted = all_predicted[

        :,

        -1,

        idx

    ]

    error = np.abs(

        predicted - actual

    )

    print(f"\n{STATE_NAMES[idx]}")

    for i, seed in enumerate(SEEDS):

        print(

            f"Seed {seed} | "

            f"Actual: {actual[i]: .6f} | "

            f"Pred: {predicted[i]: .6f} | "

            f"Error: {error[i]: .6f}"

        )

print("\n" + "=" * 70)

print("FIRST SIGNIFICANT DRIFT")

print("=" * 70)

thresholds = [

    0.01,

    0.03,

    0.05,

    0.10,

    0.20

]

for name, idx in targets.items():

    print(f"\n{name}")

    mean_error = all_errors[

        :,

        :,

        idx

    ].mean(axis=0)

    for threshold in thresholds:

        found = np.where(

            mean_error >= threshold

        )[0]

        if len(found) > 0:

            step = found[0] + 1

            print(

                f"Error >= {threshold:.2f} "

                f"first at step {step}"

            )

        else:

            print(

                f"Error >= {threshold:.2f} "

                f"tidak tercapai"

            )

print("\n" + "=" * 70)

print("POSITION vs VELOCITY ERROR CORRELATION")

print("=" * 70)

pairs = [

    ("Joint4", 3, 10),

    ("Joint6", 5, 12),

]

for name, pos_idx, vel_idx in pairs:

    pos_error = all_errors[

        :,

        :,

        pos_idx

    ].flatten()

    vel_error = all_errors[

        :,

        :,

        vel_idx

    ].flatten()

    correlation = np.corrcoef(

        pos_error,

        vel_error

    )[0, 1]

    print(

        f"{name}: "

        f"corr(position_error, velocity_error) = "

        f"{correlation:.6f}"

    )

print("\n" + "=" * 70)

print("WORST JOINT4 / JOINT6 EVENTS")

print("=" * 70)

for name, idx in targets.items():

    flat_idx = np.argmax(

        all_errors[:, :, idx]

    )

    seed_idx, step_idx = np.unravel_index(

        flat_idx,

        all_errors[:, :, idx].shape

    )

    error_value = all_errors[

        seed_idx,

        step_idx,

        idx

    ]

    actual_value = all_actual[

        seed_idx,

        step_idx,

        idx

    ]

    predicted_value = all_predicted[

        seed_idx,

        step_idx,

        idx

    ]

    print(f"\n{name}")

    print(

        f"Seed      : "

        f"{SEEDS[seed_idx]}"

    )

    print(

        f"Step      : "

        f"{step_idx + 1}"

    )

    print(

        f"Error     : "

        f"{error_value:.8f}"

    )

    print(

        f"Actual    : "

        f"{actual_value:.8f}"

    )

    print(

        f"Predicted : "

        f"{predicted_value:.8f}"

    )

print("\n" + "=" * 70)

print("ERROR GROUP SUMMARY")

print("=" * 70)

position_mae = all_errors[:, :, 0:7].mean()

velocity_mae = all_errors[:, :, 7:14].mean()

eef_mae = all_errors[:, :, 14:17].mean()

print(

    f"Joint Position MAE : "

    f"{position_mae:.8f}"

)

print(

    f"Joint Velocity MAE : "

    f"{velocity_mae:.8f}"

)

print(

    f"EEF Position MAE   : "

    f"{eef_mae:.8f}"

)

print("\n" + "=" * 70)

print("ALL DIMENSIONS SORTED BY ROLLOUT ERROR")

print("=" * 70)

dimension_mae = all_errors.mean(

    axis=(0, 1)

)

ranking = np.argsort(

    dimension_mae

)[::-1]

for rank, idx in enumerate(ranking, start=1):

    print(

        f"{rank:02d}. "

        f"{STATE_NAMES[idx]:15s} | "

        f"MAE: {dimension_mae[idx]:.8f}"

    )

print("\n" + "=" * 70)

print("GLOBAL ERROR GROWTH")

print("=" * 70)

global_time_error = all_errors.mean(

    axis=(0, 2)

)

for step in checkpoints:

    print(

        f"Step {step:03d} | "

        f"MAE: "

        f"{global_time_error[step - 1]:.8f}"

    )

print("\n" + "=" * 70)

print("GLOBAL WORST ERROR")

print("=" * 70)

flat_idx = np.argmax(all_errors)

seed_idx, step_idx, dim_idx = np.unravel_index(

    flat_idx,

    all_errors.shape

)

print(

    f"Seed      : "

    f"{SEEDS[seed_idx]}"

)

print(

    f"Step      : "

    f"{step_idx + 1}"

)

print(

    f"Dimension : "

    f"{STATE_NAMES[dim_idx]}"

)

print(

    f"Error     : "

    f"{all_errors[seed_idx, step_idx, dim_idx]:.8f}"

)

print(

    f"Actual    : "

    f"{all_actual[seed_idx, step_idx, dim_idx]:.8f}"

)

print(

    f"Predicted : "

    f"{all_predicted[seed_idx, step_idx, dim_idx]:.8f}"

)

print("\n" + "=" * 70)

print("DIAGNOSTIC SELESAI")

print("=" * 70)
