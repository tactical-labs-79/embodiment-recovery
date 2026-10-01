import logging

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

V6_MODEL = "dynamics_model_v6.pt"

V7_MODEL = "dynamics_model_v7.pt"

SEEDS = list(range(6001, 6021))

ROLLOUT_STEPS = 100

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

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

def load_model(filename):

    model = DynamicsModel().cpu()

    checkpoint = torch.load(

        filename,

        map_location="cpu"

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

    return model

model_v6 = load_model(

    V6_MODEL

)

model_v7 = load_model(

    V7_MODEL

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

    state = np.concatenate(

        [

            joint_pos,

            joint_vel,

            eef_pos

        ]

    ).astype(np.float32)

    assert state.shape == (17,)

    return state

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=False,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    use_object_obs=False,

    control_freq=20,

)

v6_rollout_maes = []

v6_final_maes = []

v6_max_errors = []

v7_rollout_maes = []

v7_final_maes = []

v7_max_errors = []

v6_one_step_maes = []

v7_one_step_maes = []

print("=" * 80)

print("UNSEEN SEED V6 vs V7 EVALUATION")

print("=" * 80)

print()

print(

    f"Seeds       : {SEEDS}"

)

print(

    f"Seed count  : {len(SEEDS)}"

)

print(

    f"Horizon     : {ROLLOUT_STEPS}"

)

print()

print(

    "Important:"

)

print(

    "V6 and V7 use the SAME environment trajectory"

)

print(

    "and the SAME action sequence for each seed."

)

for seed in SEEDS:

    np.random.seed(seed)

    torch.manual_seed(seed)

    obs = env.reset()

    state_v6 = extract_state(obs).copy()

    state_v7 = state_v6.copy()

    actual_states = []

    predicted_v6 = []

    predicted_v7 = []

    actions = []

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

        actions.append(

            action.copy()

        )

        next_obs, reward, done, info = env.step(

            action

        )

        actual_next = extract_state(

            next_obs

        )

        actual_states.append(

            actual_next.copy()

        )

        input_v6 = np.concatenate(

            [

                state_v6,

                action

            ]

        ).astype(np.float32)

        with torch.no_grad():

            pred_v6 = model_v6(

                torch.from_numpy(

                    input_v6

                ).unsqueeze(0)

            ).numpy()[0]

        predicted_v6.append(

            pred_v6.copy()

        )

        input_v7 = np.concatenate(

            [

                state_v7,

                action

            ]

        ).astype(np.float32)

        with torch.no_grad():

            pred_v7 = model_v7(

                torch.from_numpy(

                    input_v7

                ).unsqueeze(0)

            ).numpy()[0]

        predicted_v7.append(

            pred_v7.copy()

        )

        state_v6 = pred_v6.copy()

        state_v7 = pred_v7.copy()

        obs = next_obs

    actual_states = np.asarray(

        actual_states,

        dtype=np.float32

    )

    predicted_v6 = np.asarray(

        predicted_v6,

        dtype=np.float32

    )

    predicted_v7 = np.asarray(

        predicted_v7,

        dtype=np.float32

    )

    error_v6 = np.abs(

        predicted_v6

        - actual_states

    )

    error_v7 = np.abs(

        predicted_v7

        - actual_states

    )

    v6_one = error_v6[0].mean()

    v7_one = error_v7[0].mean()

    v6_rollout = error_v6.mean()

    v7_rollout = error_v7.mean()

    v6_final = error_v6[-1].mean()

    v7_final = error_v7[-1].mean()

    v6_max = error_v6.max()

    v7_max = error_v7.max()

    v6_one_step_maes.append(

        v6_one

    )

    v7_one_step_maes.append(

        v7_one

    )

    v6_rollout_maes.append(

        v6_rollout

    )

    v7_rollout_maes.append(

        v7_rollout

    )

    v6_final_maes.append(

        v6_final

    )

    v7_final_maes.append(

        v7_final

    )

    v6_max_errors.append(

        v6_max

    )

    v7_max_errors.append(

        v7_max

    )

    print()

    print(

        f"Seed {seed}"

    )

    print(

        f"  V6 rollout : "

        f"{v6_rollout:.8f}   "

        f"V7 rollout : "

        f"{v7_rollout:.8f}"

    )

    print(

        f"  V6 final   : "

        f"{v6_final:.8f}   "

        f"V7 final   : "

        f"{v7_final:.8f}"

    )

    print(

        f"  V6 max     : "

        f"{v6_max:.8f}   "

        f"V7 max     : "

        f"{v7_max:.8f}"

    )

v6_one_step_maes = np.asarray(

    v6_one_step_maes

)

v7_one_step_maes = np.asarray(

    v7_one_step_maes

)

v6_rollout_maes = np.asarray(

    v6_rollout_maes

)

v7_rollout_maes = np.asarray(

    v7_rollout_maes

)

v6_final_maes = np.asarray(

    v6_final_maes

)

v7_final_maes = np.asarray(

    v7_final_maes

)

v6_max_errors = np.asarray(

    v6_max_errors

)

v7_max_errors = np.asarray(

    v7_max_errors

)

print()

print("=" * 80)

print("UNSEEN SEED SUMMARY")

print("=" * 80)

print()

print("ONE-STEP MAE")

print(

    f"  V6 mean : "

    f"{v6_one_step_maes.mean():.8f}"

)

print(

    f"  V7 mean : "

    f"{v7_one_step_maes.mean():.8f}"

)

print()

print("100-STEP ROLLOUT MAE")

print(

    f"  V6 mean : "

    f"{v6_rollout_maes.mean():.8f}"

)

print(

    f"  V7 mean : "

    f"{v7_rollout_maes.mean():.8f}"

)

print()

print("FINAL-STEP MAE")

print(

    f"  V6 mean : "

    f"{v6_final_maes.mean():.8f}"

)

print(

    f"  V7 mean : "

    f"{v7_final_maes.mean():.8f}"

)

print()

print("MAX ERROR")

print(

    f"  V6 mean : "

    f"{v6_max_errors.mean():.8f}"

)

print(

    f"  V7 mean : "

    f"{v7_max_errors.mean():.8f}"

)

def improvement(

    baseline,

    new

):

    return (

        (

            baseline.mean()

            - new.mean()

        )

        /

        baseline.mean()

        * 100.0

    )

print()

print("=" * 80)

print("V7 IMPROVEMENT OVER V6")

print("=" * 80)

print()

print(

    f"One-step MAE improvement : "

    f"{improvement(v6_one_step_maes, v7_one_step_maes):+.2f}%"

)

print(

    f"Rollout MAE improvement  : "

    f"{improvement(v6_rollout_maes, v7_rollout_maes):+.2f}%"

)

print(

    f"Final MAE improvement    : "

    f"{improvement(v6_final_maes, v7_final_maes):+.2f}%"

)

print(

    f"Max error improvement    : "

    f"{improvement(v6_max_errors, v7_max_errors):+.2f}%"

)

print()

print("=" * 80)

print("SEED-BY-SEED WINS")

print("=" * 80)

rollout_wins = int(

    (

        v7_rollout_maes

        < v6_rollout_maes

    ).sum()

)

final_wins = int(

    (

        v7_final_maes

        < v6_final_maes

    ).sum()

)

max_wins = int(

    (

        v7_max_errors

        < v6_max_errors

    ).sum()

)

print()

print(

    f"V7 rollout wins : "

    f"{rollout_wins} / {len(SEEDS)}"

)

print(

    f"V7 final wins   : "

    f"{final_wins} / {len(SEEDS)}"

)

print(

    f"V7 max wins     : "

    f"{max_wins} / {len(SEEDS)}"

)

np.savez(

    "unseen_v6_v7_results.npz",

    seeds=np.asarray(

        SEEDS,

        dtype=np.int32

    ),

    v6_one_step=v6_one_step_maes,

    v7_one_step=v7_one_step_maes,

    v6_rollout=v6_rollout_maes,

    v7_rollout=v7_rollout_maes,

    v6_final=v6_final_maes,

    v7_final=v7_final_maes,

    v6_max=v6_max_errors,

    v7_max=v7_max_errors,

)

env.close()

print()

print("=" * 80)

print("SAVED")

print("=" * 80)

print(

    "unseen_v6_v7_results.npz"

)

print()

print("=" * 80)

print("DONE")

print("=" * 80)
