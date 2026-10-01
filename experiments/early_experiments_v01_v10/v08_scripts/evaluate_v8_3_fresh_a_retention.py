import copy

import logging

import os

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

MODEL_FILE = "dynamics_model_v7.pt"

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

ROLLOUT_STEPS = 120

POS_AMPLITUDE = 0.18

ROT_AMPLITUDE = 0.12

DEVICE = torch.device("cpu")

A_SEEDS = [

    9401,

    9402,

    9403,

    9404,

    9405,

]

MODELS = {

    "V7_baseline":

        MODEL_FILE,

    "5traj_subset1":

        "v8_2_5traj_subset1.pt",

    "5traj_subset2":

        "v8_2_5traj_subset2.pt",

    "5traj_subset3":

        "v8_2_5traj_subset3.pt",

    "5traj_subset4":

        "v8_2_5traj_subset4.pt",

    "5traj_subset5":

        "v8_2_5traj_subset5.pt",

    "10traj_subset1":

        "v8_2_10traj_subset1.pt",

    "10traj_subset2":

        "v8_2_10traj_subset2.pt",

    "10traj_subset3":

        "v8_2_10traj_subset3.pt",

    "10traj_subset4":

        "v8_2_10traj_subset4.pt",

    "10traj_subset5":

        "v8_2_10traj_subset5.pt",

    "20traj":

        "v8_2_20traj_subset0.pt",

}

logging.getLogger(

    "robosuite"

).setLevel(

    logging.WARNING

)

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(

                INPUT_DIM,

                HIDDEN

            ),

            nn.ReLU(),

            nn.Linear(

                HIDDEN,

                HIDDEN

            ),

            nn.ReLU(),

            nn.Linear(

                HIDDEN,

                OUTPUT_DIM

            )

        )

    def forward(self, x):

        return self.network(x)

def load_model(path):

    model = DynamicsModel().to(

        DEVICE

    )

    checkpoint = torch.load(

        path,

        map_location=DEVICE

    )

    if isinstance(

        checkpoint,

        dict

    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[

            "model_state_dict"

        ]

    else:

        state_dict = checkpoint

    state_dict = {

        k.replace(

            "net.",

            "network.",

            1

        ): v

        for k, v in state_dict.items()

    }

    model.load_state_dict(

        state_dict

    )

    model.eval()

    return model

def make_env(seed):

    return suite.make(

        env_name="Lift",

        robots="Panda",

        has_renderer=False,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        use_object_obs=False,

        control_freq=20,

        seed=seed,

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

    ).astype(

        np.float32

    )

    assert state.shape == (17,)

    return state

def generate_structured_actions(seed):

    rng = np.random.default_rng(

        seed

    )

    actions = []

    amplitudes = np.array(

        [

            POS_AMPLITUDE,

            POS_AMPLITUDE,

            POS_AMPLITUDE,

            ROT_AMPLITUDE,

            ROT_AMPLITUDE,

            ROT_AMPLITUDE,

        ],

        dtype=np.float32

    )

    dimensions = np.arange(6)

    rng.shuffle(

        dimensions

    )

    while len(actions) < ROLLOUT_STEPS:

        for dim in dimensions:

            if len(actions) >= ROLLOUT_STEPS:

                break

            amplitude = amplitudes[dim]

            for _ in range(3):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = amplitude

                actions.append(

                    action

                )

            for _ in range(6):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = -amplitude

                actions.append(

                    action

                )

            for _ in range(3):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = amplitude

                actions.append(

                    action

                )

            if len(actions) < ROLLOUT_STEPS:

                actions.append(

                    np.zeros(

                        7,

                        dtype=np.float32

                    )

                )

    return np.asarray(

        actions[:ROLLOUT_STEPS],

        dtype=np.float32

    )

def evaluate_a_rollout(

    model,

    seed

):

    env = make_env(

        seed

    )

    try:

        obs = env.reset()

        env.sim.forward()

        obs = env._get_observations(

            force_update=True

        )

        state = extract_state(

            obs

        )

        actions = generate_structured_actions(

            seed

        )

        errors = []

        free_state = state.copy()

        for step in range(

            ROLLOUT_STEPS

        ):

            action = actions[step]

            model_input = np.concatenate(

                [

                    free_state,

                    action

                ]

            ).astype(

                np.float32

            )

            with torch.no_grad():

                prediction = (

                    model(

                        torch.from_numpy(

                            model_input

                        ).unsqueeze(0)

                    )

                    .cpu()

                    .numpy()[0]

                )

            next_obs, _, _, _ = (

                env.step(action)

            )

            next_state = extract_state(

                next_obs

            )

            errors.append(

                np.abs(

                    prediction

                    - next_state

                )

            )

            free_state = (

                prediction.copy()

            )

        errors = np.asarray(

            errors,

            dtype=np.float32

        )

        return {

            "rollout_mae":

                float(

                    errors.mean()

                ),

            "final":

                float(

                    errors[-1].mean()

                ),

            "max":

                float(

                    errors.max()

                ),

        }

    finally:

        env.close()

def evaluate_model(

    model_name,

    model_path

):

    model = load_model(

        model_path

    )

    results = []

    for seed in A_SEEDS:

        result = evaluate_a_rollout(

            model,

            seed

        )

        results.append(

            result

        )

    rollout = np.asarray(

        [

            r["rollout_mae"]

            for r in results

        ],

        dtype=np.float64

    )

    final = np.asarray(

        [

            r["final"]

            for r in results

        ],

        dtype=np.float64

    )

    max_error = np.asarray(

        [

            r["max"]

            for r in results

        ],

        dtype=np.float64

    )

    return {

        "name":

            model_name,

        "rollout_mean":

            float(

                rollout.mean()

            ),

        "rollout_std":

            float(

                rollout.std(ddof=1)

            ),

        "final_mean":

            float(

                final.mean()

            ),

        "max_mean":

            float(

                max_error.mean()

            ),

        "per_seed":

            rollout,

    }

print("=" * 90)

print(

    "V8.3 — FRESH A RETENTION AUDIT"

)

print("=" * 90)

print()

print(

    "A seeds:",

    A_SEEDS

)

print()

missing = []

for name, path in MODELS.items():

    if not os.path.exists(path):

        missing.append(

            path

        )

if missing:

    print(

        "WARNING — missing model files:"

    )

    for path in missing:

        print(

            " ",

            path

        )

    raise FileNotFoundError(

        "One or more required checkpoint files are missing."

    )

results = []

for name, path in MODELS.items():

    print()

    print(

        "-" * 90

    )

    print(

        "Evaluating:",

        name

    )

    print(

        "Checkpoint:",

        path

    )

    result = evaluate_model(

        name,

        path

    )

    results.append(

        result

    )

    print(

        "Rollout:",

        f"{result['rollout_mean']:.8f}"

    )

    print(

        "Std:",

        f"{result['rollout_std']:.8f}"

    )

    print(

        "Final:",

        f"{result['final_mean']:.8f}"

    )

    print(

        "Max:",

        f"{result['max_mean']:.8f}"

    )

baseline = results[0]

baseline_rollout = (

    baseline["rollout_mean"]

)

print()

print("=" * 90)

print(

    "V8.3 FRESH A RETENTION SUMMARY"

)

print("=" * 90)

print()

header = (

    f"{'Model':>20} | "

    f"{'A Rollout':>12} | "

    f"{'Delta':>12} | "

    f"{'Change %':>10}"

)

print(header)

print(

    "-" * len(header)

)

for result in results:

    delta = (

        result["rollout_mean"]

        - baseline_rollout

    )

    change_pct = (

        delta

        / max(

            baseline_rollout,

            1e-12

        )

        * 100.0

    )

    print(

        f"{result['name']:>20} | "

        f"{result['rollout_mean']:12.8f} | "

        f"{delta:+12.8f} | "

        f"{change_pct:+9.3f}%"

    )

print()

print("=" * 90)

print(

    "GROUP RETENTION"

)

print("=" * 90)

groups = {

    "5traj": [

        r

        for r in results

        if r["name"].startswith(

            "5traj"

        )

    ],

    "10traj": [

        r

        for r in results

        if r["name"].startswith(

            "10traj"

        )

    ],

    "20traj": [

        r

        for r in results

        if r["name"] == "20traj"

    ],

}

for group_name, group_results in groups.items():

    values = np.asarray(

        [

            r["rollout_mean"]

            for r in group_results

        ],

        dtype=np.float64

    )

    changes = (

        (

            values

            - baseline_rollout

        )

        / max(

            baseline_rollout,

            1e-12

        )

        * 100.0

    )

    print()

    print(

        group_name

    )

    print(

        "  Mean A rollout:",

        f"{values.mean():.8f}"

    )

    print(

        "  Std:",

        f"{values.std(ddof=1) if len(values) > 1 else 0.0:.8f}"

    )

    print(

        "  Mean change:",

        f"{changes.mean():+.3f}%"

    )

    print(

        "  Min change:",

        f"{changes.min():+.3f}%"

    )

    print(

        "  Max change:",

        f"{changes.max():+.3f}%"

    )

np.savez(

    "embodiment_recovery_v8_3_fresh_a_retention.npz",

    names=np.asarray(

        [

            r["name"]

            for r in results

        ]

    ),

    rollout_mean=np.asarray(

        [

            r["rollout_mean"]

            for r in results

        ],

        dtype=np.float32

    ),

    rollout_std=np.asarray(

        [

            r["rollout_std"]

            for r in results

        ],

        dtype=np.float32

    ),

    final_mean=np.asarray(

        [

            r["final_mean"]

            for r in results

        ],

        dtype=np.float32

    ),

    max_mean=np.asarray(

        [

            r["max_mean"]

            for r in results

        ],

        dtype=np.float32

    ),

    baseline_rollout=np.float32(

        baseline_rollout

    ),

)

print()

print("=" * 90)

print(

    "SAVED"

)

print("=" * 90)

print(

    "embodiment_recovery_v8_3_fresh_a_retention.npz"

)

print()

print("DONE")
