import logging

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

MODEL_FILE = "dynamics_model_v6.pt"

DATA_FILE = "combined_dataset_v6.npz"

DIAGNOSTIC_FILE = "rollout_v6_diagnostic.npz"

SEED = 5005

FAIL_STEP = 76

WINDOW_START = 65

WINDOW_END = 85

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

K_NEAREST = 20

J4_POS = 3

J6_POS = 5

J7_POS = 6

J4_VEL = 10

J6_VEL = 12

J7_VEL = 13

FOCUS_DIMS = [

    J4_POS,

    J4_VEL,

    J6_POS,

    J6_VEL,

    J7_POS,

    J7_VEL,

]

FOCUS_NAMES = [

    "J4_Pos",

    "J4_Vel",

    "J6_Pos",

    "J6_Vel",

    "J7_Pos",

    "J7_Vel",

]

logging.getLogger("robosuite").setLevel(

    logging.WARNING

)

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

data = np.load(

    DATA_FILE

)

train_states = data[

    "states"

].astype(np.float32)

train_actions = data[

    "actions"

].astype(np.float32)

train_next_states = data[

    "next_states"

].astype(np.float32)

diagnostic = np.load(

    DIAGNOSTIC_FILE

)

saved_actual = diagnostic[

    "actual"

]

saved_predicted = diagnostic[

    "predicted"

]

saved_errors = diagnostic[

    "errors"

]

seed_list = [

    1001,

    2002,

    3003,

    4004,

    5005,

]

seed_index = seed_list.index(

    SEED

)

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

np.random.seed(SEED)

torch.manual_seed(SEED)

obs = env.reset()

state = extract_state(obs)

actual_states = []

predicted_states = []

actions_history = []

for step in range(100):

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

    ).astype(np.float32)

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

    actions_history.append(

        action.copy()

    )

    state = predicted_next_state.copy()

    obs = next_obs

env.close()

actual_states = np.asarray(

    actual_states,

    dtype=np.float32

)

predicted_states = np.asarray(

    predicted_states,

    dtype=np.float32

)

actions_history = np.asarray(

    actions_history,

    dtype=np.float32

)

print("=" * 80)

print("V6 J7 FAILURE ANALYSIS")

print("=" * 80)

print()

print(

    f"Seed       : {SEED}"

)

print(

    f"Failure step: {FAIL_STEP}"

)

repro_actual_error = np.max(

    np.abs(

        actual_states

        - saved_actual[seed_index]

    )

)

repro_pred_error = np.max(

    np.abs(

        predicted_states

        - saved_predicted[seed_index]

    )

)

print()

print("Reproduction check:")

print(

    f"  Max actual difference    : "

    f"{repro_actual_error:.10f}"

)

print(

    f"  Max predicted difference : "

    f"{repro_pred_error:.10f}"

)

print()

print("=" * 80)

print("1. FAILURE CONTEXT")

print("=" * 80)

print()

print(

    f"{'Step':>5} "

    f"{'J7Pos A':>11} "

    f"{'J7Pos P':>11} "

    f"{'J7Vel A':>11} "

    f"{'J7Vel P':>11} "

    f"{'Err':>11} "

    f"{'Action':>10}"

)

for step in range(

    WINDOW_START,

    WINDOW_END + 1

):

    i = step - 1

    actual = actual_states[i]

    predicted = predicted_states[i]

    action = actions_history[i]

    action_idx = np.argmax(

        np.abs(action)

    )

    action_value = action[

        action_idx

    ]

    error = abs(

        predicted[J7_VEL]

        - actual[J7_VEL]

    )

    print(

        f"{step:5d} "

        f"{actual[J7_POS]:11.6f} "

        f"{predicted[J7_POS]:11.6f} "

        f"{actual[J7_VEL]:11.6f} "

        f"{predicted[J7_VEL]:11.6f} "

        f"{error:11.6f} "

        f"A{action_idx + 1}:{action_value:+.3f}"

    )

print()

print("=" * 80)

print("2. STATE AT FAILURE")

print("=" * 80)

failure_i = FAIL_STEP - 1

failure_actual = actual_states[

    failure_i

]

failure_predicted = predicted_states[

    failure_i

]

failure_action = actions_history[

    failure_i

]

action_idx = np.argmax(

    np.abs(failure_action)

)

print()

print("J7 failure:")

print(

    f"  Actual J7_Pos : "

    f"{failure_actual[J7_POS]: .8f}"

)

print(

    f"  Pred J7_Pos   : "

    f"{failure_predicted[J7_POS]: .8f}"

)

print(

    f"  Actual J7_Vel : "

    f"{failure_actual[J7_VEL]: .8f}"

)

print(

    f"  Pred J7_Vel   : "

    f"{failure_predicted[J7_VEL]: .8f}"

)

print(

    f"  Error         : "

    f"{abs(failure_predicted[J7_VEL] - failure_actual[J7_VEL]): .8f}"

)

print()

print(

    f"Action A{action_idx + 1} : "

    f"{failure_action[action_idx]:+.8f}"

)

print()

print("Other focus dimensions:")

for dim, name in zip(

    FOCUS_DIMS,

    FOCUS_NAMES

):

    print(

        f"  {name:8s} "

        f"Actual={failure_actual[dim]: .8f} "

        f"Pred={failure_predicted[dim]: .8f} "

        f"Err={abs(failure_predicted[dim] - failure_actual[dim]): .8f}"

    )

pre_failure_state = predicted_states[

    FAIL_STEP - 2

]

failure_action = actions_history[

    FAIL_STEP - 1

]

model_input = np.concatenate(

    [

        pre_failure_state,

        failure_action

    ]

).astype(np.float32)

print()

print("=" * 80)

print("3. NEAREST TRAINING STATES BEFORE FAILURE")

print("=" * 80)

train_mean = train_states.mean(

    axis=0

)

train_std = train_states.std(

    axis=0

)

train_std[

    train_std < 1e-8

] = 1.0

train_z = (

    train_states

    - train_mean

) / train_std

query_z = (

    pre_failure_state

    - train_mean

) / train_std

distances = np.sqrt(

    np.sum(

        (

            train_z

            - query_z

        ) ** 2,

        axis=1

    )

)

nearest = np.argsort(

    distances

)[:K_NEAREST]

print()

print(

    f"Target predicted state = "

    f"step {FAIL_STEP - 1}"

)

print()

print(

    f"{'Rank':>5} "

    f"{'Index':>8} "

    f"{'Dist':>10} "

    f"{'J7Pos':>10} "

    f"{'J7Vel':>10} "

    f"{'Action':>10} "

    f"{'NextJ7V':>10}"

)

for rank, idx in enumerate(

    nearest,

    1

):

    action = train_actions[

        idx

    ]

    action_idx = np.argmax(

        np.abs(action)

    )

    action_value = action[

        action_idx

    ]

    next_j7_vel = train_next_states[

        idx,

        J7_VEL

    ]

    print(

        f"{rank:5d} "

        f"{idx:8d} "

        f"{distances[idx]:10.6f} "

        f"{train_states[idx, J7_POS]:10.6f} "

        f"{train_states[idx, J7_VEL]:10.6f} "

        f"A{action_idx + 1}:{action_value:+.3f} "

        f"{next_j7_vel:10.6f}"

    )

print()

print("=" * 80)

print("4. MODEL ERROR ON NEAREST TRAINING TRANSITIONS")

print("=" * 80)

nearest_states = train_states[

    nearest

]

nearest_actions = train_actions[

    nearest

]

nearest_next = train_next_states[

    nearest

]

nearest_inputs = np.concatenate(

    [

        nearest_states,

        nearest_actions

    ],

    axis=1

)

with torch.no_grad():

    prediction = model(

        torch.from_numpy(

            nearest_inputs

        )

    ).cpu().numpy()

nearest_j7_error = np.abs(

    prediction[:, J7_VEL]

    - nearest_next[:, J7_VEL]

)

print()

print(

    f"{'Rank':>5} "

    f"{'Index':>8} "

    f"{'StateDist':>10} "

    f"{'J7VelActual':>13} "

    f"{'J7VelPred':>12} "

    f"{'J7VelErr':>11}"

)

for rank, (idx, error) in enumerate(

    zip(nearest, nearest_j7_error),

    1

):

    print(

        f"{rank:5d} "

        f"{idx:8d} "

        f"{distances[rank - 1]:10.6f} "

        f"{nearest_next[rank - 1, J7_VEL]:13.6f} "

        f"{prediction[rank - 1, J7_VEL]:12.6f} "

        f"{error:11.6f}"

    )

print()

print("=" * 80)

print("5. LOCAL ACTION COVERAGE OF NEAREST 100 STATES")

print("=" * 80)

local_k = min(

    100,

    len(train_states)

)

local_indices = np.argsort(

    distances

)[:local_k]

local_actions = train_actions[

    local_indices

]

for i in range(7):

    values = local_actions[

        np.abs(

            local_actions[:, i]

        ) > 1e-8,

        i

    ]

    if len(values) == 0:

        print(

            f"Action {i + 1}: 0"

        )

        continue

    print(

        f"Action {i + 1}: "

        f"count={len(values):3d} "

        f"pos={(values > 0).sum():3d} "

        f"neg={(values < 0).sum():3d} "

        f"min={values.min():+.4f} "

        f"max={values.max():+.4f}"

    )

print()

print("=" * 80)

print("6. J7 VELOCITY COVERAGE")

print("=" * 80)

print()

print("Global training J7 velocity:")

print(

    f"  Min : "

    f"{train_states[:, J7_VEL].min(): .8f}"

)

print(

    f"  Max : "

    f"{train_states[:, J7_VEL].max(): .8f}"

)

print(

    f"  Mean: "

    f"{train_states[:, J7_VEL].mean(): .8f}"

)

print()

print("Failure predicted J7 velocity:")

print(

    f"  Predicted : "

    f"{failure_predicted[J7_VEL]: .8f}"

)

print(

    f"  Actual    : "

    f"{failure_actual[J7_VEL]: .8f}"

)

pred_j7_vel = predicted_states[

    :,

    J7_VEL

].reshape(-1)

actual_j7_vel = actual_states[

    :,

    J7_VEL

].reshape(-1)

train_j7_vel = train_states[

    :,

    J7_VEL

]

print()

print(

    "Predicted J7 velocity outside global "

    "training range:"

)

outside = (

    (pred_j7_vel < train_j7_vel.min())

    |

    (pred_j7_vel > train_j7_vel.max())

)

print(

    f"  {outside.sum()} / "

    f"{len(pred_j7_vel)} "

    f"({outside.mean() * 100:.2f}%)"

)

print()

print("=" * 80)

print("7. INTERPRETATION DATA")

print("=" * 80)

print()

print(

    "Failure event:"

)

print(

    f"  Seed = {SEED}"

)

print(

    f"  Step = {FAIL_STEP}"

)

print(

    f"  J7 velocity error = "

    f"{abs(failure_predicted[J7_VEL] - failure_actual[J7_VEL]):.8f}"

)

print()

print(

    "Nearest training-state distance:"

)

print(

    f"  Best = {distances[nearest[0]]:.8f}"

)

print()

print(

    "Nearest-transition J7 velocity error:"

)

print(

    f"  Mean = {nearest_j7_error.mean():.8f}"

)

print(

    f"  Max  = {nearest_j7_error.max():.8f}"

)

print()

print("=" * 80)

print("DONE")

print("=" * 80)
