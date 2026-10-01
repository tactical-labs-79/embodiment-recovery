import logging

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

import mujoco

from robosuite.utils.mjmod import DynamicsModder

MODEL_FILE = "dynamics_model_v7.pt"

SEEDS = [

    8001,

    8002,

    8003,

    8004,

    8005,

]

ROLLOUT_STEPS = 100

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

MASS_MULTIPLIERS = [

    1.0,

    1.5,

    2.0,

    3.0,

    5.0,

]

SANITY_TOL = 1e-6

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

model = DynamicsModel().cpu()

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu"

)

if "model_state_dict" in checkpoint:

    state_dict = (

        checkpoint["model_state_dict"]

    )

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

def generate_actions(seed):

    rng = np.random.default_rng(

        seed

    )

    actions = []

    for _ in range(

        ROLLOUT_STEPS

    ):

        action = np.zeros(

            7,

            dtype=np.float32

        )

        action_idx = rng.integers(

            0,

            7

        )

        magnitude = rng.uniform(

            0.05,

            0.20

        )

        sign = rng.choice(

            [-1.0, 1.0]

        )

        action[action_idx] = (

            sign * magnitude

        )

        actions.append(

            action

        )

    return np.asarray(

        actions,

        dtype=np.float32

    )

def find_link5_body(env):

    model = env.sim.model._model

    matches = []

    for body_id in range(

        model.nbody

    ):

        body_name = mujoco.mj_id2name(

            model,

            mujoco.mjtObj.mjOBJ_BODY,

            body_id

        )

        if body_name is None:

            continue

        if body_name.endswith(

            "link5"

        ):

            matches.append(

                (

                    body_id,

                    body_name

                )

            )

    if len(matches) == 0:

        raise RuntimeError(

            "Could not find Panda link5 body."

        )

    if len(matches) > 1:

        raise RuntimeError(

            f"Multiple link5 bodies found: "

            f"{matches}"

        )

    return matches[0]

def run_sanity_test(

    seed,

    actions

):

    env_a = make_env(

        seed

    )

    env_b = make_env(

        seed

    )

    try:

        obs_a = env_a.reset()

        obs_b = env_b.reset()

        state_a = extract_state(

            obs_a

        )

        state_b = extract_state(

            obs_b

        )

        initial_diff = float(

            np.max(

                np.abs(

                    state_a

                    - state_b

                )

            )

        )

        max_step_diff = 0.0

        for action in actions:

            next_obs_a, _, _, _ = (

                env_a.step(action)

            )

            next_obs_b, _, _, _ = (

                env_b.step(action)

            )

            next_state_a = extract_state(

                next_obs_a

            )

            next_state_b = extract_state(

                next_obs_b

            )

            step_diff = float(

                np.max(

                    np.abs(

                        next_state_a

                        - next_state_b

                    )

                )

            )

            max_step_diff = max(

                max_step_diff,

                step_diff

            )

        passed = (

            max(

                initial_diff,

                max_step_diff

            )

            <= SANITY_TOL

        )

        return {

            "passed": passed,

            "initial_diff":

                initial_diff,

            "max_step_diff":

                max_step_diff,

        }

    finally:

        env_a.close()

        env_b.close()

def apply_mass_shift(

    env,

    body_name,

    target_mass

):

    model = env.sim.model._model

    body_id = mujoco.mj_name2id(

        model,

        mujoco.mjtObj.mjOBJ_BODY,

        body_name

    )

    if body_id < 0:

        raise RuntimeError(

            f"Body not found: {body_name}"

        )

    before_mass = float(

        env.sim.model.body_mass[

            body_id

        ]

    )

    modder = DynamicsModder(

        env.sim,

        randomize_mass=False

    )

    modder.mod_mass(

        body_name,

        float(target_mass)

    )

    modder.update()

    after_mass = float(

        env.sim.model.body_mass[

            body_id

        ]

    )

    env.sim.forward()

    return (

        before_mass,

        after_mass

    )

def run_shift_experiment(

    seed,

    multiplier,

    baseline_mass,

    body_name,

    actions

):

    env_a = make_env(

        seed

    )

    env_b = make_env(

        seed

    )

    try:

        obs_a = env_a.reset()

        obs_b = env_b.reset()

        state_a = extract_state(

            obs_a

        )

        state_b = extract_state(

            obs_b

        )

        initial_state_diff = float(

            np.max(

                np.abs(

                    state_a

                    - state_b

                )

            )

        )

        target_mass = (

            baseline_mass

            * multiplier

        )

        (

            before_mass,

            after_mass

        ) = apply_mass_shift(

            env_b,

            body_name,

            target_mass

        )

        env_a.sim.forward()

        env_b.sim.forward()

        postmod_obs_a = (

            env_a._get_observations(

                force_update=True

            )

        )

        postmod_obs_b = (

            env_b._get_observations(

                force_update=True

            )

        )

        postmod_state_a = (

            extract_state(

                postmod_obs_a

            )

        )

        postmod_state_b = (

            extract_state(

                postmod_obs_b

            )

        )

        postmod_state_diff = float(

            np.max(

                np.abs(

                    postmod_state_a

                    - postmod_state_b

                )

            )

        )

        free_state_a = (

            postmod_state_a.copy()

        )

        free_state_b = (

            postmod_state_b.copy()

        )

        actual_a = []

        actual_b = []

        predicted_a = []

        predicted_b = []

        one_step_errors_a = []

        one_step_errors_b = []

        rollout_errors_a = []

        rollout_errors_b = []

        actual_ab_diffs = []

        actions_used = []

        for step in range(

            ROLLOUT_STEPS

        ):

            action = actions[

                step

            ]

            actions_used.append(

                action.copy()

            )

            current_obs_a = (

                env_a._get_observations(

                    force_update=True

                )

            )

            current_obs_b = (

                env_b._get_observations(

                    force_update=True

                )

            )

            current_actual_a = (

                extract_state(

                    current_obs_a

                )

            )

            current_actual_b = (

                extract_state(

                    current_obs_b

                )

            )

            tf_input_a = np.concatenate(

                [

                    current_actual_a,

                    action

                ]

            ).astype(

                np.float32

            )

            with torch.no_grad():

                tf_prediction_a = model(

                    torch.from_numpy(

                        tf_input_a

                    ).unsqueeze(0)

                ).numpy()[0]

            tf_input_b = np.concatenate(

                [

                    current_actual_b,

                    action

                ]

            ).astype(

                np.float32

            )

            with torch.no_grad():

                tf_prediction_b = model(

                    torch.from_numpy(

                        tf_input_b

                    ).unsqueeze(0)

                ).numpy()[0]

            next_obs_a, _, _, _ = (

                env_a.step(action)

            )

            actual_next_a = extract_state(

                next_obs_a

            )

            next_obs_b, _, _, _ = (

                env_b.step(action)

            )

            actual_next_b = extract_state(

                next_obs_b

            )

            actual_ab_diffs.append(

                np.abs(

                    actual_next_a

                    - actual_next_b

                )

            )

            one_step_errors_a.append(

                np.abs(

                    tf_prediction_a

                    - actual_next_a

                )

            )

            one_step_errors_b.append(

                np.abs(

                    tf_prediction_b

                    - actual_next_b

                )

            )

            free_input_a = np.concatenate(

                [

                    free_state_a,

                    action

                ]

            ).astype(

                np.float32

            )

            with torch.no_grad():

                free_prediction_a = model(

                    torch.from_numpy(

                        free_input_a

                    ).unsqueeze(0)

                ).numpy()[0]

            free_input_b = np.concatenate(

                [

                    free_state_b,

                    action

                ]

            ).astype(

                np.float32

            )

            with torch.no_grad():

                free_prediction_b = model(

                    torch.from_numpy(

                        free_input_b

                    ).unsqueeze(0)

                ).numpy()[0]

            rollout_errors_a.append(

                np.abs(

                    free_prediction_a

                    - actual_next_a

                )

            )

            rollout_errors_b.append(

                np.abs(

                    free_prediction_b

                    - actual_next_b

                )

            )

            actual_a.append(

                actual_next_a.copy()

            )

            actual_b.append(

                actual_next_b.copy()

            )

            predicted_a.append(

                free_prediction_a.copy()

            )

            predicted_b.append(

                free_prediction_b.copy()

            )

            free_state_a = (

                free_prediction_a.copy()

            )

            free_state_b = (

                free_prediction_b.copy()

            )

        actual_a = np.asarray(

            actual_a,

            dtype=np.float32

        )

        actual_b = np.asarray(

            actual_b,

            dtype=np.float32

        )

        predicted_a = np.asarray(

            predicted_a,

            dtype=np.float32

        )

        predicted_b = np.asarray(

            predicted_b,

            dtype=np.float32

        )

        one_step_errors_a = np.asarray(

            one_step_errors_a,

            dtype=np.float32

        )

        one_step_errors_b = np.asarray(

            one_step_errors_b,

            dtype=np.float32

        )

        rollout_errors_a = np.asarray(

            rollout_errors_a,

            dtype=np.float32

        )

        rollout_errors_b = np.asarray(

            rollout_errors_b,

            dtype=np.float32

        )

        actual_ab_diffs = np.asarray(

            actual_ab_diffs,

            dtype=np.float32

        )

        actions_used = np.asarray(

            actions_used,

            dtype=np.float32

        )

        a_one_step = float(

            one_step_errors_a.mean()

        )

        b_one_step = float(

            one_step_errors_b.mean()

        )

        a_rollout = float(

            rollout_errors_a.mean()

        )

        b_rollout = float(

            rollout_errors_b.mean()

        )

        a_final = float(

            rollout_errors_a[-1].mean()

        )

        b_final = float(

            rollout_errors_b[-1].mean()

        )

        a_max = float(

            rollout_errors_a.max()

        )

        b_max = float(

            rollout_errors_b.max()

        )

        physical_gap_mean = float(

            actual_ab_diffs.mean()

        )

        physical_gap_max = float(

            actual_ab_diffs.max()

        )

        b_time_error = (

            rollout_errors_b.mean(

                axis=1

            )

        )

        def first_threshold(

            threshold

        ):

            hits = np.where(

                b_time_error >= threshold

            )[0]

            if len(hits) == 0:

                return -1

            return int(

                hits[0] + 1

            )

        first_01 = first_threshold(

            0.10

        )

        first_02 = first_threshold(

            0.20

        )

        first_05 = first_threshold(

            0.50

        )

        b_one_step_dim = (

            one_step_errors_b.mean(

                axis=0

            )

        )

        b_rollout_dim = (

            rollout_errors_b.mean(

                axis=0

            )

        )

        return {

            "seed":

                seed,

            "multiplier":

                multiplier,

            "baseline_mass":

                baseline_mass,

            "before_mass":

                before_mass,

            "target_mass":

                target_mass,

            "after_mass":

                after_mass,

            "initial_state_diff":

                initial_state_diff,

            "postmod_state_diff":

                postmod_state_diff,

            "physical_gap_mean":

                physical_gap_mean,

            "physical_gap_max":

                physical_gap_max,

            "a_one_step":

                a_one_step,

            "b_one_step":

                b_one_step,

            "a_rollout":

                a_rollout,

            "b_rollout":

                b_rollout,

            "a_final":

                a_final,

            "b_final":

                b_final,

            "a_max":

                a_max,

            "b_max":

                b_max,

            "first_01":

                first_01,

            "first_02":

                first_02,

            "first_05":

                first_05,

            "b_one_step_dim":

                b_one_step_dim,

            "b_rollout_dim":

                b_rollout_dim,

            "actual_a":

                actual_a,

            "actual_b":

                actual_b,

            "predicted_a":

                predicted_a,

            "predicted_b":

                predicted_b,

            "actions":

                actions_used,

            "actual_ab_diffs":

                actual_ab_diffs,

        }

    finally:

        env_a.close()

        env_b.close()

DIM_NAMES = []

for i in range(7):

    DIM_NAMES.append(

        f"J{i + 1}_Pos"

    )

for i in range(7):

    DIM_NAMES.append(

        f"J{i + 1}_Vel"

    )

DIM_NAMES.extend(

    [

        "EEF_X",

        "EEF_Y",

        "EEF_Z",

    ]

)

print("=" * 90)

print(

    "EMBODIMENT SHIFT V4 — DISTAL MASS"

)

print("=" * 90)

print()

print(

    f"Model   : {MODEL_FILE}"

)

print(

    f"Seeds   : {SEEDS}"

)

print(

    f"Horizon : {ROLLOUT_STEPS}"

)

print(

    "Mass multipliers:",

    MASS_MULTIPLIERS

)

print(

    f"Sanity tolerance: "

    f"{SANITY_TOL:.1e}"

)

print()

print("=" * 90)

print(

    "PHASE 1 — A/B SANITY TEST"

)

print("=" * 90)

print()

sanity_results = []

for seed in SEEDS:

    actions = generate_actions(

        seed

    )

    result = run_sanity_test(

        seed,

        actions

    )

    sanity_results.append(

        result

    )

    print(

        f"Seed {seed}"

    )

    print(

        f"  Initial diff : "

        f"{result['initial_diff']:.12e}"

    )

    print(

        f"  Max step diff: "

        f"{result['max_step_diff']:.12e}"

    )

    print(

        f"  PASS         : "

        f"{result['passed']}"

    )

sanity_passed = all(

    r["passed"]

    for r in sanity_results

)

if not sanity_passed:

    print()

    print(

        "=" * 90

    )

    print(

        "SANITY FAILED"

    )

    print(

        "Do not continue."

    )

    print(

        "=" * 90

    )

    raise SystemExit(

        1

    )

print()

print(

    "SANITY PASSED"

)

print(

    "Robot A/B are identical before "

    "the mass shift."

)

reference_env = make_env(

    SEEDS[0]

)

reference_env.reset()

(

    body_id,

    body_name

) = find_link5_body(

    reference_env

)

baseline_mass = float(

    reference_env

    .sim

    .model

    .body_mass[

        body_id

    ]

)

reference_env.close()

print()

print("=" * 90)

print(

    "PHASE 2 — MASS SETUP"

)

print("=" * 90)

print()

print(

    f"Body ID       : {body_id}"

)

print(

    f"Body name     : {body_name}"

)

print(

    f"Baseline mass : "

    f"{baseline_mass:.8f} kg"

)

print()

print(

    "Targets:"

)

for multiplier in MASS_MULTIPLIERS:

    target = (

        baseline_mass

        * multiplier

    )

    print(

        f"  x{multiplier:.1f}"

        f" -> "

        f"{target:.8f} kg"

    )

print()

print("=" * 90)

print(

    "PHASE 3 — EMBODIMENT SHIFT"

)

print("=" * 90)

all_results = []

for seed in SEEDS:

    print()

    print(

        "#" * 90

    )

    print(

        f"SEED {seed}"

    )

    print(

        "#" * 90

    )

    actions = generate_actions(

        seed

    )

    for multiplier in MASS_MULTIPLIERS:

        result = run_shift_experiment(

            seed=seed,

            multiplier=multiplier,

            baseline_mass=baseline_mass,

            body_name=body_name,

            actions=actions,

        )

        all_results.append(

            result

        )

        print()

        print(

            f"{body_name} mass "

            f"x{multiplier:.1f}"

        )

        print(

            f"  before       : "

            f"{result['before_mass']:.8f}"

        )

        print(

            f"  target       : "

            f"{result['target_mass']:.8f}"

        )

        print(

            f"  after        : "

            f"{result['after_mass']:.8f}"

        )

        print(

            f"  initial diff : "

            f"{result['initial_state_diff']:.10e}"

        )

        print(

            f"  postmod diff : "

            f"{result['postmod_state_diff']:.10e}"

        )

        print(

            f"  physical gap : "

            f"{result['physical_gap_mean']:.8f}"

        )

        print(

            f"  A 1-step     : "

            f"{result['a_one_step']:.8f}"

        )

        print(

            f"  B 1-step     : "

            f"{result['b_one_step']:.8f}"

        )

        print(

            f"  A rollout    : "

            f"{result['a_rollout']:.8f}"

        )

        print(

            f"  B rollout    : "

            f"{result['b_rollout']:.8f}"

        )

        print(

            f"  A final      : "

            f"{result['a_final']:.8f}"

        )

        print(

            f"  B final      : "

            f"{result['b_final']:.8f}"

        )

        print(

            f"  B max        : "

            f"{result['b_max']:.8f}"

        )

        print(

            f"  B >=0.10     : "

            f"{result['first_01']}"

        )

        print(

            f"  B >=0.20     : "

            f"{result['first_02']}"

        )

        print(

            f"  B >=0.50     : "

            f"{result['first_05']}"

        )

print()

print("=" * 90)

print(

    "V4 MASS SHIFT SUMMARY"

)

print("=" * 90)

print()

header = (

    f"{'Mult':>7} | "

    f"{'Target':>10} | "

    f"{'PhysGap':>10} | "

    f"{'A 1-step':>11} | "

    f"{'B 1-step':>11} | "

    f"{'Δ1-step':>11} | "

    f"{'A Roll':>11} | "

    f"{'B Roll':>11} | "

    f"{'ΔRoll':>11}"

)

print(

    header

)

print(

    "-" * len(header)

)

for multiplier in MASS_MULTIPLIERS:

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    target = np.mean(

        [

            r["target_mass"]

            for r in rows

        ]

    )

    physical_gap = np.mean(

        [

            r["physical_gap_mean"]

            for r in rows

        ]

    )

    a_one = np.mean(

        [

            r["a_one_step"]

            for r in rows

        ]

    )

    b_one = np.mean(

        [

            r["b_one_step"]

            for r in rows

        ]

    )

    a_roll = np.mean(

        [

            r["a_rollout"]

            for r in rows

        ]

    )

    b_roll = np.mean(

        [

            r["b_rollout"]

            for r in rows

        ]

    )

    print(

        f"{multiplier:7.1f} | "

        f"{target:10.4f} | "

        f"{physical_gap:10.6f} | "

        f"{a_one:11.8f} | "

        f"{b_one:11.8f} | "

        f"{b_one-a_one:+11.8f} | "

        f"{a_roll:11.8f} | "

        f"{b_roll:11.8f} | "

        f"{b_roll-a_roll:+11.8f}"

    )

print()

print("=" * 90)

print(

    "FINAL / MAX ERROR"

)

print("=" * 90)

for multiplier in MASS_MULTIPLIERS:

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    b_final = np.mean(

        [

            r["b_final"]

            for r in rows

        ]

    )

    b_max = np.mean(

        [

            r["b_max"]

            for r in rows

        ]

    )

    print()

    print(

        f"Mass x{multiplier:.1f}"

    )

    print(

        f"  B final : "

        f"{b_final:.8f}"

    )

    print(

        f"  B max   : "

        f"{b_max:.8f}"

    )

print()

print("=" * 90)

print(

    "FAILURE THRESHOLD SUMMARY"

)

print("=" * 90)

for multiplier in MASS_MULTIPLIERS:

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    print()

    print(

        f"Mass x{multiplier:.1f}"

    )

    for threshold, key in [

        (

            0.10,

            "first_01"

        ),

        (

            0.20,

            "first_02"

        ),

        (

            0.50,

            "first_05"

        ),

    ]:

        values = np.asarray(

            [

                r[key]

                for r in rows

            ],

            dtype=np.int32

        )

        valid = values[

            values >= 0

        ]

        print(

            f"  >= {threshold:.2f}: "

            f"{len(valid)} / "

            f"{len(values)} seeds"

        )

        if len(valid) > 0:

            print(

                f"      mean step = "

                f"{valid.mean():.2f}"

            )

print()

print("=" * 90)

print(

    "TOP ROBOT B ERROR DIMENSIONS"

)

print("=" * 90)

for multiplier in MASS_MULTIPLIERS:

    rows = [

        r

        for r in all_results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    matrix = np.asarray(

        [

            r["b_rollout_dim"]

            for r in rows

        ]

    )

    mean_dim = matrix.mean(

        axis=0

    )

    sorted_dims = np.argsort(

        mean_dim

    )[::-1]

    print()

    print(

        f"Mass x{multiplier:.1f}"

    )

    for rank, dim in enumerate(

        sorted_dims[:5],

        1

    ):

        print(

            f"  {rank}. "

            f"{DIM_NAMES[dim]:8s} "

            f"{mean_dim[dim]:.8f}"

        )

np.savez(

    "embodiment_shift_v4_mass_results.npz",

    seeds=np.asarray(

        SEEDS,

        dtype=np.int32

    ),

    multipliers=np.asarray(

        MASS_MULTIPLIERS,

        dtype=np.float32

    ),

    baseline_mass=np.float32(

        baseline_mass

    ),

    target_masses=np.asarray(

        [

            baseline_mass * multiplier

            for multiplier

            in MASS_MULTIPLIERS

        ],

        dtype=np.float32

    ),

    a_one_step=np.asarray(

        [

            r["a_one_step"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_one_step=np.asarray(

        [

            r["b_one_step"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    a_rollout=np.asarray(

        [

            r["a_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_rollout=np.asarray(

        [

            r["b_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    a_final=np.asarray(

        [

            r["a_final"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_final=np.asarray(

        [

            r["b_final"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    a_max=np.asarray(

        [

            r["a_max"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_max=np.asarray(

        [

            r["b_max"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    physical_gap_mean=np.asarray(

        [

            r["physical_gap_mean"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    physical_gap_max=np.asarray(

        [

            r["physical_gap_max"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    first_01=np.asarray(

        [

            r["first_01"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    first_02=np.asarray(

        [

            r["first_02"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    first_05=np.asarray(

        [

            r["first_05"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    b_one_step_dim=np.asarray(

        [

            r["b_one_step_dim"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_rollout_dim=np.asarray(

        [

            r["b_rollout_dim"]

            for r in all_results

        ],

        dtype=np.float32

    ),

)

for result in all_results:

    seed = result[

        "seed"

    ]

    multiplier = result[

        "multiplier"

    ]

    tag = (

        f"seed{seed}_"

        f"mult{multiplier:.1f}"

        .replace(

            ".",

            "_"

        )

    )

    np.savez(

        f"embodiment_v4_{tag}.npz",

        actual_a=result[

            "actual_a"

        ],

        actual_b=result[

            "actual_b"

        ],

        predicted_a=result[

            "predicted_a"

        ],

        predicted_b=result[

            "predicted_b"

        ],

        actions=result[

            "actions"

        ],

        actual_ab_diffs=result[

            "actual_ab_diffs"

        ],

    )

print()

print("=" * 90)

print(

    "SAVED"

)

print("=" * 90)

print(

    "embodiment_shift_v4_mass_results.npz"

)

print(

    "Full trajectories:"

)

print(

    "embodiment_v4_seed*_mult*.npz"

)

print()

print("=" * 90)

print(

    "DONE"

)

print("=" * 90)
