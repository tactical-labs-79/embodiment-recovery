import numpy as np

import torch

import torch.nn as nn

MODEL_FILE = "dynamics_model_v6.pt"

DIAGNOSTIC_FILE = "rollout_v6_diagnostic.npz"

SEEDS = [

    1001,

    2002,

    3003,

    4004,

    5005,

]

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

J4_POS = 3

J6_POS = 5

J7_POS = 6

J4_VEL = 10

J6_VEL = 12

J7_VEL = 13

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

    DIAGNOSTIC_FILE

)

actual = data["actual"]

predicted = data["predicted"]

all_actions = []

for seed in SEEDS:

    np.random.seed(seed)

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

    all_actions.append(

        np.asarray(actions)

    )

all_actions = np.asarray(

    all_actions,

    dtype=np.float32

)

teacher_errors = []

free_errors = []

teacher_j7 = []

free_j7 = []

teacher_global = []

free_global = []

for seed_i in range(

    len(SEEDS)

):

    seed_teacher = []

    seed_free = []

    seed_teacher_j7 = []

    seed_free_j7 = []

    for step_i in range(

        1,

        100

    ):

        actual_pre = actual[

            seed_i,

            step_i - 1

        ]

        predicted_pre = predicted[

            seed_i,

            step_i - 1

        ]

        action = all_actions[

            seed_i,

            step_i

        ]

        actual_next = actual[

            seed_i,

            step_i

        ]

        input_actual = np.concatenate(

            [

                actual_pre,

                action

            ]

        ).astype(np.float32)

        input_predicted = np.concatenate(

            [

                predicted_pre,

                action

            ]

        ).astype(np.float32)

        with torch.no_grad():

            pred_from_actual = model(

                torch.from_numpy(

                    input_actual

                ).unsqueeze(0)

            ).cpu().numpy()[0]

            pred_from_predicted = model(

                torch.from_numpy(

                    input_predicted

                ).unsqueeze(0)

            ).cpu().numpy()[0]

        err_teacher = np.abs(

            pred_from_actual

            - actual_next

        )

        err_free = np.abs(

            pred_from_predicted

            - actual_next

        )

        seed_teacher.append(

            err_teacher.mean()

        )

        seed_free.append(

            err_free.mean()

        )

        seed_teacher_j7.append(

            err_teacher[J7_VEL]

        )

        seed_free_j7.append(

            err_free[J7_VEL]

        )

    teacher_errors.append(

        seed_teacher

    )

    free_errors.append(

        seed_free

    )

    teacher_j7.append(

        seed_teacher_j7

    )

    free_j7.append(

        seed_free_j7

    )

teacher_errors = np.asarray(

    teacher_errors

)

free_errors = np.asarray(

    free_errors

)

teacher_j7 = np.asarray(

    teacher_j7

)

free_j7 = np.asarray(

    free_j7

)

print("=" * 80)

print("V6 FIRST DRIFT ANALYSIS")

print("=" * 80)

for seed_i, seed in enumerate(

    SEEDS

):

    print()

    print(

        "=" * 80

    )

    print(

        f"SEED {seed}"

    )

    print(

        "=" * 80

    )

    worst_i = np.argmax(

        free_j7[seed_i]

    )

    worst_step = worst_i + 2

    print()

    print(

        f"Worst free J7 transition:"

    )

    print(

        f"  Step       : {worst_step}"

    )

    print(

        f"  Teacher J7 : "

        f"{teacher_j7[seed_i, worst_i]:.8f}"

    )

    print(

        f"  Free J7    : "

        f"{free_j7[seed_i, worst_i]:.8f}"

    )

    for threshold in [

        0.03,

        0.05,

        0.10,

        0.20,

        0.50,

    ]:

        hits = np.where(

            free_j7[seed_i]

            >= threshold

        )[0]

        if len(hits) == 0:

            print(

                f"  Free J7 >= "

                f"{threshold:.2f} : never"

            )

        else:

            step = hits[0] + 2

            print(

                f"  Free J7 >= "

                f"{threshold:.2f} : "

                f"step {step}"

            )

seed_i = SEEDS.index(5005)

print()

print("=" * 80)

print("SEED 5005 — FIRST DRIFT WINDOW")

print("=" * 80)

print()

print(

    f"{'Step':>6} "

    f"{'TeacherMAE':>13} "

    f"{'FreeMAE':>13} "

    f"{'TeacherJ7':>13} "

    f"{'FreeJ7':>13}"

)

for step in range(

    2,

    81

):

    i = step - 2

    if (

        step <= 20

        or step >= 55

    ):

        print(

            f"{step:6d} "

            f"{teacher_errors[seed_i, i]:13.8f} "

            f"{free_errors[seed_i, i]:13.8f} "

            f"{teacher_j7[seed_i, i]:13.8f} "

            f"{free_j7[seed_i, i]:13.8f}"

        )

print()

print("=" * 80)

print("FIRST DIVERGENCE POINT")

print("=" * 80)

seed_teacher = teacher_errors[seed_i]

seed_free = free_errors[seed_i]

difference = (

    seed_free

    - seed_teacher

)

for ratio_threshold in [

    2.0,

    3.0,

    5.0,

    10.0,

]:

    valid = (

        seed_teacher > 1e-8

    )

    ratios = np.zeros_like(

        seed_free

    )

    ratios[valid] = (

        seed_free[valid]

        / seed_teacher[valid]

    )

    hits = np.where(

        ratios >= ratio_threshold

    )[0]

    if len(hits) == 0:

        print(

            f"Free/Teacher >= "

            f"{ratio_threshold:.0f}x : "

            f"never"

        )

    else:

        i = hits[0]

        step = i + 2

        print(

            f"Free/Teacher >= "

            f"{ratio_threshold:.0f}x : "

            f"step {step} "

            f"(ratio={ratios[i]:.2f}x)"

        )

print()

print("=" * 80)

print("TOP 15 FREE-vs-TEACHER DIVERGENCE EVENTS")

print("=" * 80)

ratios = (

    free_errors[seed_i]

    /

    np.maximum(

        teacher_errors[seed_i],

        1e-8

    )

)

top = np.argsort(

    ratios

)[-15:][::-1]

print()

print(

    f"{'Rank':>5} "

    f"{'Step':>6} "

    f"{'Teacher':>12} "

    f"{'Free':>12} "

    f"{'Ratio':>12} "

    f"{'J7Teacher':>12} "

    f"{'J7Free':>12}"

)

for rank, i in enumerate(

    top,

    1

):

    print(

        f"{rank:5d} "

        f"{i + 2:6d} "

        f"{teacher_errors[seed_i, i]:12.8f} "

        f"{free_errors[seed_i, i]:12.8f} "

        f"{ratios[i]:12.4f} "

        f"{teacher_j7[seed_i, i]:12.8f} "

        f"{free_j7[seed_i, i]:12.8f}"

    )

print()

print("=" * 80)

print("5. ERROR BEFORE FIRST MAJOR J7 SPIKE")

print("=" * 80)

print()

print(

    f"{'Step':>6} "

    f"{'TeacherMAE':>13} "

    f"{'FreeMAE':>13} "

    f"{'J7Teacher':>13} "

    f"{'J7Free':>13}"

)

for step in range(

    50,

    68

):

    i = step - 2

    print(

        f"{step:6d} "

        f"{teacher_errors[seed_i, i]:13.8f} "

        f"{free_errors[seed_i, i]:13.8f} "

        f"{teacher_j7[seed_i, i]:13.8f} "

        f"{free_j7[seed_i, i]:13.8f}"

    )

print()

print("=" * 80)

print("6. SUMMARY")

print("=" * 80)

print()

print(

    "Interpretation rule:"

)

print(

    "If teacher-forced error stays small while "

    "free-running error grows,"

)

print(

    "the main issue is accumulated state drift "

    "rather than local one-step dynamics."

)

print()

print(

    f"Seed 5005 teacher mean "

    f"MAE (steps 2-100): "

    f"{teacher_errors[seed_i].mean():.8f}"

)

print(

    f"Seed 5005 free mean "

    f"MAE (steps 2-100): "

    f"{free_errors[seed_i].mean():.8f}"

)

print()

print("=" * 80)

print("DONE")

print("=" * 80)
