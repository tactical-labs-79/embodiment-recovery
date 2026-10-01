import numpy as np

import torch

import torch.nn as nn

MODEL_FILE = "dynamics_model_v6.pt"

DIAGNOSTIC_FILE = "rollout_v6_diagnostic.npz"

SEED = 5005

FAIL_STEP = 76

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

J4_POS = 3

J5_POS = 4

J6_POS = 5

J7_POS = 6

J4_VEL = 10

J5_VEL = 11

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

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint

state_dict = {

    k.replace("net.", "network.", 1): v

    for k, v in state_dict.items()

}

model.load_state_dict(state_dict)

model.eval()

data = np.load(DIAGNOSTIC_FILE)

actual = data["actual"]

predicted = data["predicted"]

errors = data["errors"]

seed_list = [

    1001,

    2002,

    3003,

    4004,

    5005,

]

seed_index = seed_list.index(SEED)

np.random.seed(SEED)

actions = []

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

    actions.append(

        action

    )

actions = np.asarray(

    actions,

    dtype=np.float32

)

fail_i = FAIL_STEP - 1

pre_i = FAIL_STEP - 2

actual_pre = actual[

    seed_index,

    pre_i

]

predicted_pre = predicted[

    seed_index,

    pre_i

]

actual_next = actual[

    seed_index,

    fail_i

]

predicted_next_free = predicted[

    seed_index,

    fail_i

]

action = actions[

    fail_i

]

input_from_actual = np.concatenate(

    [

        actual_pre,

        action

    ]

).astype(np.float32)

input_from_predicted = np.concatenate(

    [

        predicted_pre,

        action

    ]

).astype(np.float32)

with torch.no_grad():

    pred_from_actual = model(

        torch.from_numpy(

            input_from_actual

        ).unsqueeze(0)

    ).cpu().numpy()[0]

    pred_from_predicted = model(

        torch.from_numpy(

            input_from_predicted

        ).unsqueeze(0)

    ).cpu().numpy()[0]

error_from_actual = np.abs(

    pred_from_actual

    - actual_next

)

error_from_predicted = np.abs(

    pred_from_predicted

    - actual_next

)

print("=" * 80)

print("V6 FAILURE ONE-STEP ANALYSIS")

print("=" * 80)

print()

print(

    f"Seed       : {SEED}"

)

print(

    f"Failure step: {FAIL_STEP}"

)

print(

    f"Using stored diagnostic data: {DIAGNOSTIC_FILE}"

)

action_idx = np.argmax(

    np.abs(action)

)

print()

print("=" * 80)

print("1. FAILURE TRANSITION")

print("=" * 80)

print()

print(

    f"Action : A{action_idx + 1}"

)

print(

    f"Value  : {action[action_idx]:+.8f}"

)

print()

print(

    f"Actual state at step {FAIL_STEP - 1}:"

)

print(

    f"  J4_Pos = {actual_pre[J4_POS]: .8f}"

)

print(

    f"  J6_Pos = {actual_pre[J6_POS]: .8f}"

)

print(

    f"  J7_Pos = {actual_pre[J7_POS]: .8f}"

)

print(

    f"  J4_Vel = {actual_pre[J4_VEL]: .8f}"

)

print(

    f"  J6_Vel = {actual_pre[J6_VEL]: .8f}"

)

print(

    f"  J7_Vel = {actual_pre[J7_VEL]: .8f}"

)

print()

print(

    f"Predicted free-running state at "

    f"step {FAIL_STEP - 1}:"

)

print(

    f"  J4_Pos = {predicted_pre[J4_POS]: .8f}"

)

print(

    f"  J6_Pos = {predicted_pre[J6_POS]: .8f}"

)

print(

    f"  J7_Pos = {predicted_pre[J7_POS]: .8f}"

)

print(

    f"  J4_Vel = {predicted_pre[J4_VEL]: .8f}"

)

print(

    f"  J6_Vel = {predicted_pre[J6_VEL]: .8f}"

)

print(

    f"  J7_Vel = {predicted_pre[J7_VEL]: .8f}"

)

print()

print(

    "Actual state vs predicted free-running state "

    "BEFORE failure:"

)

print(

    f"  State MAE = "

    f"{np.mean(np.abs(actual_pre - predicted_pre)):.8f}"

)

print()

print("=" * 80)

print("2. KEY J7 ONE-STEP TEST")

print("=" * 80)

print()

print("Actual next J7 velocity:")

print(

    f"  {actual_next[J7_VEL]: .8f}"

)

print()

print("Model prediction from REAL state:")

print(

    f"  {pred_from_actual[J7_VEL]: .8f}"

)

print(

    f"  Error = "

    f"{error_from_actual[J7_VEL]: .8f}"

)

print()

print("Model prediction from FREE-RUNNING state:")

print(

    f"  {pred_from_predicted[J7_VEL]: .8f}"

)

print(

    f"  Error = "

    f"{error_from_predicted[J7_VEL]: .8f}"

)

print()

print("=" * 80)

print("3. GLOBAL ONE-STEP COMPARISON")

print("=" * 80)

print()

print(

    f"{'Dimension':>12} "

    f"{'ActualState':>15} "

    f"{'PredFromActual':>16} "

    f"{'ErrActual':>13} "

    f"{'PredFromFree':>16} "

    f"{'ErrFree':>13}"

)

for dim in range(17):

    print(

        f"{dim:12d} "

        f"{actual_next[dim]:15.8f} "

        f"{pred_from_actual[dim]:16.8f} "

        f"{error_from_actual[dim]:13.8f} "

        f"{pred_from_predicted[dim]:16.8f} "

        f"{error_from_predicted[dim]:13.8f}"

    )

print()

print("=" * 80)

print("4. FOCUS DIMENSIONS")

print("=" * 80)

print()

for dim, name in zip(

    FOCUS_DIMS,

    FOCUS_NAMES

):

    print(name)

    print(

        f"  Actual next      : "

        f"{actual_next[dim]: .8f}"

    )

    print(

        f"  From actual state: "

        f"{pred_from_actual[dim]: .8f}"

    )

    print(

        f"  Error            : "

        f"{error_from_actual[dim]: .8f}"

    )

    print(

        f"  From free state  : "

        f"{pred_from_predicted[dim]: .8f}"

    )

    print(

        f"  Error            : "

        f"{error_from_predicted[dim]: .8f}"

    )

    print()

print("=" * 80)

print("5. MAE COMPARISON")

print("=" * 80)

mae_from_actual = error_from_actual.mean()

mae_from_predicted = error_from_predicted.mean()

print()

print(

    f"One-step MAE from REAL state       : "

    f"{mae_from_actual:.8f}"

)

print(

    f"One-step MAE from FREE-RUNNING state: "

    f"{mae_from_predicted:.8f}"

)

print()

if mae_from_actual > 0:

    ratio = (

        mae_from_predicted

        / mae_from_actual

    )

    print(

        f"Free/Real error ratio : "

        f"{ratio:.4f}x"

    )

j7_real_error = error_from_actual[J7_VEL]

j7_free_error = error_from_predicted[J7_VEL]

print()

print("=" * 80)

print("6. J7 ERROR GAP")

print("=" * 80)

print()

print(

    f"J7 error from REAL state       : "

    f"{j7_real_error:.8f}"

)

print(

    f"J7 error from FREE-RUN state   : "

    f"{j7_free_error:.8f}"

)

print(

    f"Additional error caused by "

    f"state drift : "

    f"{j7_free_error - j7_real_error:+.8f}"

)

print()

print("=" * 80)

print("7. FAILURE PROPAGATION AROUND STEP 76")

print("=" * 80)

print()

print(

    f"{'Step':>6} "

    f"{'J7 Actual':>13} "

    f"{'J7 Pred':>13} "

    f"{'J7 Error':>13}"

)

for step in range(

    max(1, FAIL_STEP - 10),

    min(100, FAIL_STEP + 10) + 1

):

    i = step - 1

    actual_j7 = actual[

        seed_index,

        i,

        J7_VEL

    ]

    predicted_j7 = predicted[

        seed_index,

        i,

        J7_VEL

    ]

    error_j7 = abs(

        predicted_j7

        - actual_j7

    )

    print(

        f"{step:6d} "

        f"{actual_j7:13.8f} "

        f"{predicted_j7:13.8f} "

        f"{error_j7:13.8f}"

    )

print()

print("=" * 80)

print("DONE")

print("=" * 80)
