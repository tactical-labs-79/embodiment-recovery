import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

MODEL_FILE = "dynamics_model_v6.pt"

DATA_FILE = "combined_dataset_v6.npz"

SEEDS = [

    1001,

    2002,

    3003,

    4004,

    5005,

]

ROLLOUT_STEPS = 100

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

FOCUS_DIMS = [3, 10, 5, 12]

FOCUS_NAMES = [

    "J4_Pos",

    "J4_Vel",

    "J6_Pos",

    "J6_Vel",

]

STEP_CHECKS = [

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

device = torch.device("cpu")

model = DynamicsModel().to(device)

checkpoint = torch.load(

    MODEL_FILE,

    map_location=device

)

if "model_state_dict" in checkpoint:

    state_dict = checkpoint[

        "model_state_dict"

    ]

else:

    state_dict = checkpoint

state_dict = {

    k.replace("net.", "network.", 1): v

    for k, v in state_dict.items()

}

model.load_state_dict(

    state_dict

)

model.eval()

data = np.load(DATA_FILE)

train_states = data[

    "states"

].astype(np.float32)

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

    return np.concatenate(

        [

            joint_pos,

            joint_vel,

            eef_pos,

        ]

    ).astype(np.float32)

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=False,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    use_object_obs=False,

    control_freq=20,

)

all_actual = []

all_predicted = []

all_errors = []

seed_results = []

print("=" * 80)

print("V6 ROLLOUT DIAGNOSTIC")

print("=" * 80)

print(

    f"Model         : {MODEL_FILE}"

)

print(

    f"Training data : {DATA_FILE}"

)

print(

    f"Seeds         : {SEEDS}"

)

print(

    f"Training states: {train_states.shape}"

)

for seed in SEEDS:

    np.random.seed(seed)

    torch.manual_seed(seed)

    obs = env.reset()

    state = extract_state(obs)

    actual_states = []

    predicted_states = []

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

        next_obs, reward, done, info = env.step(

            action

        )

        actual_next_state = extract_state(

            next_obs

        )

        model_input = np.concatenate(

            [

                state,

                action

            ]

        ).astype(

            np.float32

        )

        with torch.no_grad():

            x = torch.from_numpy(

                model_input

            ).unsqueeze(0)

            predicted_next_state = (

                model(x)

                .cpu()

                .numpy()[0]

            )

        actual_states.append(

            actual_next_state.copy()

        )

        predicted_states.append(

            predicted_next_state.copy()

        )

        state = predicted_next_state.copy()

        obs = next_obs

    actual_states = np.asarray(

        actual_states,

        dtype=np.float32

    )

    predicted_states = np.asarray(

        predicted_states,

        dtype=np.float32

    )

    errors = np.abs(

        predicted_states

        - actual_states

    )

    all_actual.append(

        actual_states

    )

    all_predicted.append(

        predicted_states

    )

    all_errors.append(

        errors

    )

    seed_results.append(

        {

            "seed": seed,

            "rollout_mae": errors.mean(),

            "final_mae": errors[-1].mean(),

            "max_error": errors.max(),

        }

    )

all_actual = np.asarray(

    all_actual

)

all_predicted = np.asarray(

    all_predicted

)

all_errors = np.asarray(

    all_errors

)

print()

print("=" * 80)

print("ROLLOUT SHAPES")

print("=" * 80)

print(

    f"Actual    : {all_actual.shape}"

)

print(

    f"Predicted : {all_predicted.shape}"

)

print(

    f"Errors    : {all_errors.shape}"

)

print()

print("=" * 80)

print("1. PER-SEED SUMMARY")

print("=" * 80)

for result in seed_results:

    print()

    print(

        f"Seed {result['seed']}"

    )

    print(

        f"  Rollout MAE : "

        f"{result['rollout_mae']:.8f}"

    )

    print(

        f"  Final MAE   : "

        f"{result['final_mae']:.8f}"

    )

    print(

        f"  Max error   : "

        f"{result['max_error']:.8f}"

    )

print()

print("=" * 80)

print("2. GLOBAL PER-DIMENSION ERROR")

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

        "EEF_Z",

    ]

)

dim_mae = all_errors.mean(

    axis=(0, 1)

)

sorted_dims = np.argsort(

    dim_mae

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

        f"{dim_mae[dim]:.8f}"

    )

print()

print("=" * 80)

print("3. ERROR BY TIME")

print("=" * 80)

step_mae = all_errors.mean(

    axis=(0, 2)

)

print()

print(

    f"{'Step':>6} "

    f"{'MAE':>14}"

)

for step in STEP_CHECKS:

    value = step_mae[

        step - 1

    ]

    print(

        f"{step:6d} "

        f"{value:14.8f}"

    )

print()

print("=" * 80)

print("4. J4/J6 ERROR BY TIME")

print("=" * 80)

for dim, name in zip(

    FOCUS_DIMS,

    FOCUS_NAMES

):

    print()

    print(name)

    dim_error = all_errors[

        :, :, dim

    ].mean(axis=0)

    print(

        f"{'Step':>6} "

        f"{'MAE':>14}"

    )

    for step in STEP_CHECKS:

        print(

            f"{step:6d} "

            f"{dim_error[step - 1]:14.8f}"

        )

print()

print("=" * 80)

print("5. FIRST SIGNIFICANT DRIFT")

print("=" * 80)

for dim, name in zip(

    FOCUS_DIMS,

    FOCUS_NAMES

):

    dim_error = all_errors[

        :, :, dim

    ].mean(axis=0)

    print()

    print(name)

    for threshold in [

        0.01,

        0.03,

        0.05,

        0.10,

        0.20,

    ]:

        hits = np.where(

            dim_error >= threshold

        )[0]

        if len(hits) == 0:

            print(

                f"  >= {threshold:.2f} : "

                f"never"

            )

        else:

            print(

                f"  >= {threshold:.2f} : "

                f"step {hits[0] + 1}"

            )

print()

print("=" * 80)

print("6. WORST GLOBAL EVENT")

print("=" * 80)

flat_index = np.argmax(

    all_errors

)

seed_i, step_i, dim_i = np.unravel_index(

    flat_index,

    all_errors.shape

)

worst_error = all_errors[

    seed_i,

    step_i,

    dim_i

]

print()

print(

    f"Seed       : "

    f"{SEEDS[seed_i]}"

)

print(

    f"Step       : "

    f"{step_i + 1}"

)

print(

    f"Dimension  : "

    f"{dim_names[dim_i]}"

)

print(

    f"Error      : "

    f"{worst_error:.8f}"

)

print(

    f"Actual     : "

    f"{all_actual[seed_i, step_i, dim_i]:.8f}"

)

print(

    f"Predicted  : "

    f"{all_predicted[seed_i, step_i, dim_i]:.8f}"

)

print()

print("=" * 80)

print("7. TRAINING MANIFOLD DISTANCE")

print("=" * 80)

train_focus = train_states[

    :,

    FOCUS_DIMS

]

actual_focus = all_actual[

    :,

    :,

    FOCUS_DIMS

].reshape(

    -1,

    len(FOCUS_DIMS)

)

pred_focus = all_predicted[

    :,

    :,

    FOCUS_DIMS

].reshape(

    -1,

    len(FOCUS_DIMS)

)

train_mean = train_focus.mean(

    axis=0

)

train_std = train_focus.std(

    axis=0

)

train_std[

    train_std < 1e-8

] = 1.0

train_z = (

    train_focus - train_mean

) / train_std

actual_z = (

    actual_focus - train_mean

) / train_std

pred_z = (

    pred_focus - train_mean

) / train_std

def nearest_distance(

    queries,

    reference,

    batch_size=100

):

    output = np.empty(

        len(queries),

        dtype=np.float32

    )

    for start in range(

        0,

        len(queries),

        batch_size

    ):

        end = min(

            start + batch_size,

            len(queries)

        )

        q = queries[

            start:end

        ]

        diff = (

            q[:, None, :]

            - reference[None, :, :]

        )

        dist = np.sqrt(

            np.sum(

                diff * diff,

                axis=2

            )

        )

        output[

            start:end

        ] = dist.min(axis=1)

    return output

actual_nn = nearest_distance(

    actual_z,

    train_z

)

pred_nn = nearest_distance(

    pred_z,

    train_z

)

print()

print("Actual states:")

print(

    f"  Mean NN : "

    f"{actual_nn.mean():.8f}"

)

print(

    f"  Median  : "

    f"{np.median(actual_nn):.8f}"

)

print(

    f"  Max NN  : "

    f"{actual_nn.max():.8f}"

)

print()

print("Predicted states:")

print(

    f"  Mean NN : "

    f"{pred_nn.mean():.8f}"

)

print(

    f"  Median  : "

    f"{np.median(pred_nn):.8f}"

)

print(

    f"  Max NN  : "

    f"{pred_nn.max():.8f}"

)

print()

print("=" * 80)

print("8. MANIFOLD DISTANCE BY TIME")

print("=" * 80)

actual_z_time = actual_z.reshape(

    len(SEEDS),

    ROLLOUT_STEPS,

    len(FOCUS_DIMS)

)

pred_z_time = pred_z.reshape(

    len(SEEDS),

    ROLLOUT_STEPS,

    len(FOCUS_DIMS)

)

actual_nn_time = []

pred_nn_time = []

for step in range(

    ROLLOUT_STEPS

):

    a = nearest_distance(

        actual_z_time[:, step, :],

        train_z

    )

    p = nearest_distance(

        pred_z_time[:, step, :],

        train_z

    )

    actual_nn_time.append(

        a.mean()

    )

    pred_nn_time.append(

        p.mean()

    )

actual_nn_time = np.asarray(

    actual_nn_time

)

pred_nn_time = np.asarray(

    pred_nn_time

)

print()

print(

    f"{'Step':>6} "

    f"{'Actual NN':>15} "

    f"{'Pred NN':>15}"

)

for step in STEP_CHECKS:

    print(

        f"{step:6d} "

        f"{actual_nn_time[step - 1]:15.8f} "

        f"{pred_nn_time[step - 1]:15.8f}"

    )

print()

print("=" * 80)

print("9. PREDICTED MANIFOLD SHIFT")

print("=" * 80)

for threshold in [

    0.25,

    0.50,

    1.00,

    1.50,

    2.00,

    3.00,

    4.00,

]:

    count = int(

        (pred_nn >= threshold).sum()

    )

    print(

        f"NN >= {threshold:4.2f} : "

        f"{count} / {len(pred_nn)} "

        f"({count / len(pred_nn) * 100:.2f}%)"

    )

print()

print("=" * 80)

print("10. TOP 10 MOST DISTANT V6 PREDICTED STATES")

print("=" * 80)

top = np.argsort(

    pred_nn

)[-10:][::-1]

print()

print(

    f"{'Rank':>5} "

    f"{'Seed':>7} "

    f"{'Step':>6} "

    f"{'NN':>12} "

    f"{'J4':>12} "

    f"{'J4Vel':>12} "

    f"{'J6':>12} "

    f"{'J6Vel':>12}"

)

for rank, flat in enumerate(

    top,

    1

):

    seed_i = flat // ROLLOUT_STEPS

    step_i = flat % ROLLOUT_STEPS

    p = all_predicted[

        seed_i,

        step_i

    ]

    print(

        f"{rank:5d} "

        f"{SEEDS[seed_i]:7d} "

        f"{step_i + 1:6d} "

        f"{pred_nn[flat]:12.6f} "

        f"{p[J4_POS if False else 3]:12.6f} "

        f"{p[10]:12.6f} "

        f"{p[5]:12.6f} "

        f"{p[12]:12.6f}"

    )

np.savez(

    "rollout_v6_diagnostic.npz",

    actual=all_actual,

    predicted=all_predicted,

    errors=all_errors,

    actual_nn=actual_nn,

    predicted_nn=pred_nn,

    actual_nn_time=actual_nn_time,

    predicted_nn_time=pred_nn_time,

)

print()

print("=" * 80)

print("SAVED")

print("=" * 80)

print(

    "rollout_v6_diagnostic.npz"

)

print()

print("=" * 80)

print("DONE")

print("=" * 80)
