import numpy as np

import torch

import torch.nn as nn

MODEL_FILE = "dynamics_model_v7.pt"

DIAGNOSTIC_FILE = "rollout_v7_diagnostic.npz"

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

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(INPUT_DIM, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, OUTPUT_DIM),

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

data = np.load(

    DIAGNOSTIC_FILE

)

actual = data["actual"]

stored_predicted = data["predicted"]

all_actions = []

for seed in SEEDS:

    np.random.seed(seed)

    actions = []

    for _ in range(100):

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

        actions.append(action)

    all_actions.append(

        np.asarray(

            actions,

            dtype=np.float32

        )

    )

all_actions = np.asarray(

    all_actions,

    dtype=np.float32

)

teacher_errors = []

free_errors = []

teacher_j7 = []

free_j7 = []

for seed_i in range(len(SEEDS)):

    seed_teacher = []

    seed_free = []

    seed_teacher_j7 = []

    seed_free_j7 = []

    for step_i in range(1, 100):

        actual_current = actual[

            seed_i,

            step_i - 1

        ]

        free_current = stored_predicted[

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

        teacher_input = np.concatenate(

            [

                actual_current,

                action

            ]

        ).astype(np.float32)

        free_input = np.concatenate(

            [

                free_current,

                action

            ]

        ).astype(np.float32)

        with torch.no_grad():

            teacher_prediction = model(

                torch.from_numpy(

                    teacher_input

                ).unsqueeze(0)

            ).cpu().numpy()[0]

            free_prediction = model(

                torch.from_numpy(

                    free_input

                ).unsqueeze(0)

            ).cpu().numpy()[0]

        teacher_error = np.abs(

            teacher_prediction

            - actual_next

        )

        free_error = np.abs(

            free_prediction

            - actual_next

        )

        seed_teacher.append(

            teacher_error.mean()

        )

        seed_free.append(

            free_error.mean()

        )

        seed_teacher_j7.append(

            teacher_error[13]

        )

        seed_free_j7.append(

            free_error[13]

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

print("V7 TEACHER-FORCED VS FREE-RUNNING")

print("=" * 80)

print()

for i, seed in enumerate(SEEDS):

    teacher_mean = teacher_errors[

        i

    ].mean()

    free_mean = free_errors[

        i

    ].mean()

    teacher_j7_mean = teacher_j7[

        i

    ].mean()

    free_j7_mean = free_j7[

        i

    ].mean()

    ratio = (

        free_mean

        / max(teacher_mean, 1e-12)

    )

    print(

        f"Seed {seed}"

    )

    print(

        f"  Teacher MAE       : "

        f"{teacher_mean:.8f}"

    )

    print(

        f"  Free MAE          : "

        f"{free_mean:.8f}"

    )

    print(

        f"  Free/Teacher      : "

        f"{ratio:.4f}x"

    )

    print(

        f"  Teacher J7 MAE    : "

        f"{teacher_j7_mean:.8f}"

    )

    print(

        f"  Free J7 MAE       : "

        f"{free_j7_mean:.8f}"

    )

    print()

teacher_global = teacher_errors.mean()

free_global = free_errors.mean()

teacher_j7_global = teacher_j7.mean()

free_j7_global = free_j7.mean()

print("=" * 80)

print("GLOBAL SUMMARY")

print("=" * 80)

print()

print(

    f"Teacher-forced MAE : "

    f"{teacher_global:.8f}"

)

print(

    f"Free-running MAE   : "

    f"{free_global:.8f}"

)

print(

    f"Free/Teacher ratio : "

    f"{free_global / max(teacher_global, 1e-12):.4f}x"

)

print()

print(

    f"Teacher J7 MAE     : "

    f"{teacher_j7_global:.8f}"

)

print(

    f"Free J7 MAE        : "

    f"{free_j7_global:.8f}"

)

print(

    f"J7 Free/Teacher    : "

    f"{free_j7_global / max(teacher_j7_global, 1e-12):.4f}x"

)

teacher_time = teacher_errors.mean(

    axis=0

)

free_time = free_errors.mean(

    axis=0

)

teacher_j7_time = teacher_j7.mean(

    axis=0

)

free_j7_time = free_j7.mean(

    axis=0

)

print()

print("=" * 80)

print("ERROR BY TIME")

print("=" * 80)

print()

print(

    f"{'Step':>6} "

    f"{'Teacher':>14} "

    f"{'Free':>14} "

    f"{'Ratio':>12}"

)

for step in [

    2,

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

]:

    i = step - 2

    ratio = (

        free_time[i]

        / max(teacher_time[i], 1e-12)

    )

    print(

        f"{step:6d} "

        f"{teacher_time[i]:14.8f} "

        f"{free_time[i]:14.8f} "

        f"{ratio:12.4f}"

    )

print()

print("=" * 80)

print("FIRST FREE/TEACHER DIVERGENCE")

print("=" * 80)

for threshold in [

    2,

    3,

    5,

    10,

]:

    ratios = (

        free_time

        / np.maximum(

            teacher_time,

            1e-12

        )

    )

    hits = np.where(

        ratios >= threshold

    )[0]

    if len(hits) == 0:

        print(

            f">= {threshold}x : never"

        )

    else:

        i = hits[0]

        print(

            f">= {threshold}x : "

            f"step {i + 2} "

            f"(ratio={ratios[i]:.4f}x)"

        )

print()

print("=" * 80)

print("MAXIMUM FREE-RUNNING ERROR")

print("=" * 80)

worst = np.unravel_index(

    np.argmax(free_errors),

    free_errors.shape

)

seed_i = worst[0]

step_i = worst[1]

print()

print(

    f"Seed       : "

    f"{SEEDS[seed_i]}"

)

print(

    f"Step       : "

    f"{step_i + 2}"

)

print(

    f"Teacher MAE: "

    f"{teacher_errors[seed_i, step_i]:.8f}"

)

print(

    f"Free MAE   : "

    f"{free_errors[seed_i, step_i]:.8f}"

)

print(

    f"Free/Teacher: "

    f"{free_errors[seed_i, step_i] / max(teacher_errors[seed_i, step_i], 1e-12):.4f}x"

)

np.savez(

    "teacher_free_v7.npz",

    teacher_errors=teacher_errors,

    free_errors=free_errors,

    teacher_j7=teacher_j7,

    free_j7=free_j7,

    teacher_time=teacher_time,

    free_time=free_time,

)

print()

print("=" * 80)

print("SAVED")

print("=" * 80)

print(

    "teacher_free_v7.npz"

)

print("=" * 80)

print("DONE")

print("=" * 80)
