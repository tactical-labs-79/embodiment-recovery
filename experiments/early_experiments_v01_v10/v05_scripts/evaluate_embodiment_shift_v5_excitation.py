import logging

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

import mujoco

from robosuite.utils.mjmod import DynamicsModder

MODEL_FILE = "dynamics_model_v7.pt"

SEEDS = [

    9001,

    9002,

    9003,

    9004,

    9005,

]

ROLLOUT_STEPS = 120

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

MASS_MULTIPLIERS = [

    1.0,

    2.0,

    3.0,

    5.0,

]

POS_AMPLITUDE = 0.18

ROT_AMPLITUDE = 0.12

SANITY_TOL = 1e-6

logging.getLogger("robosuite").setLevel(logging.WARNING)

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

model = DynamicsModel().cpu()

checkpoint = torch.load(

    MODEL_FILE,

    map_location="cpu"

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

def make_env(seed):

    env = suite.make(

        env_name="Lift",

        robots="Panda",

        has_renderer=False,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        use_object_obs=False,

        control_freq=20,

        seed=seed,

    )

    return env

def generate_structured_actions(seed):

    rng = np.random.default_rng(seed)

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

    rng.shuffle(dimensions)

    while len(actions) < ROLLOUT_STEPS:

        for dim in dimensions:

            if len(actions) >= ROLLOUT_STEPS:

                break

            amp = amplitudes[dim]

            for _ in range(3):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = amp

                actions.append(action)

            for _ in range(6):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = -amp

                actions.append(action)

            for _ in range(3):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = amp

                actions.append(action)

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

def find_link5_body(env):

    model = env.sim.model._model

    matches = []

    for body_id in range(model.nbody):

        name = mujoco.mj_id2name(

            model,

            mujoco.mjtObj.mjOBJ_BODY,

            body_id

        )

        if name is not None and name.endswith("link5"):

            matches.append(

                (

                    body_id,

                    name

                )

            )

    if len(matches) != 1:

        raise RuntimeError(

            f"Expected exactly one link5 body, got: {matches}"

        )

    return matches[0]

def run_sanity_test(seed, actions):

    env_a = make_env(seed)

    env_b = make_env(seed)

    try:

        obs_a = env_a.reset()

        obs_b = env_b.reset()

        initial_a = extract_state(obs_a)

        initial_b = extract_state(obs_b)

        initial_diff = float(

            np.max(

                np.abs(initial_a - initial_b)

            )

        )

        max_step_diff = 0.0

        for action in actions:

            obs_a, _, _, _ = env_a.step(action)

            obs_b, _, _, _ = env_b.step(action)

            state_a = extract_state(obs_a)

            state_b = extract_state(obs_b)

            diff = float(

                np.max(

                    np.abs(state_a - state_b)

                )

            )

            max_step_diff = max(

                max_step_diff,

                diff

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

            "initial_diff": initial_diff,

            "max_step_diff": max_step_diff,

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

    before = float(

        env.sim.model.body_mass[body_id]

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

    after = float(

        env.sim.model.body_mass[body_id]

    )

    env.sim.forward()

    return before, after

def run_experiment(

    seed,

    multiplier,

    baseline_mass,

    body_name,

    actions

):

    env_a = make_env(seed)

    env_b = make_env(seed)

    try:

        obs_a = env_a.reset()

        obs_b = env_b.reset()

        initial_diff = float(

            np.max(

                np.abs(

                    extract_state(obs_a)

                    - extract_state(obs_b)

                )

            )

        )

        target_mass = (

            baseline_mass * multiplier

        )

        before_mass, after_mass = (

            apply_mass_shift(

                env_b,

                body_name,

                target_mass

            )

        )

        env_a.sim.forward()

        env_b.sim.forward()

        obs_a = env_a._get_observations(

            force_update=True

        )

        obs_b = env_b._get_observations(

            force_update=True

        )

        state_a = extract_state(obs_a)

        state_b = extract_state(obs_b)

        postmod_diff = float(

            np.max(

                np.abs(state_a - state_b)

            )

        )

        free_a = state_a.copy()

        free_b = state_b.copy()

        actual_ab = []

        one_a = []

        one_b = []

        rollout_a = []

        rollout_b = []

        for step in range(ROLLOUT_STEPS):

            action = actions[step]

            tf_a = np.concatenate(

                [

                    state_a,

                    action

                ]

            ).astype(np.float32)

            tf_b = np.concatenate(

                [

                    state_b,

                    action

                ]

            ).astype(np.float32)

            with torch.no_grad():

                pred_a = model(

                    torch.from_numpy(

                        tf_a

                    ).unsqueeze(0)

                ).numpy()[0]

                pred_b = model(

                    torch.from_numpy(

                        tf_b

                    ).unsqueeze(0)

                ).numpy()[0]

            next_obs_a, _, _, _ = (

                env_a.step(action)

            )

            next_obs_b, _, _, _ = (

                env_b.step(action)

            )

            next_a = extract_state(

                next_obs_a

            )

            next_b = extract_state(

                next_obs_b

            )

            one_a.append(

                np.abs(pred_a - next_a)

            )

            one_b.append(

                np.abs(pred_b - next_b)

            )

            actual_ab.append(

                np.abs(next_a - next_b)

            )

            free_input_a = np.concatenate(

                [

                    free_a,

                    action

                ]

            ).astype(np.float32)

            with torch.no_grad():

                free_pred_a = model(

                    torch.from_numpy(

                        free_input_a

                    ).unsqueeze(0)

                ).numpy()[0]

            free_input_b = np.concatenate(

                [

                    free_b,

                    action

                ]

            ).astype(np.float32)

            with torch.no_grad():

                free_pred_b = model(

                    torch.from_numpy(

                        free_input_b

                    ).unsqueeze(0)

                ).numpy()[0]

            rollout_a.append(

                np.abs(

                    free_pred_a - next_a

                )

            )

            rollout_b.append(

                np.abs(

                    free_pred_b - next_b

                )

            )

            free_a = free_pred_a.copy()

            free_b = free_pred_b.copy()

            state_a = next_a

            state_b = next_b

        one_a = np.asarray(one_a)

        one_b = np.asarray(one_b)

        rollout_a = np.asarray(rollout_a)

        rollout_b = np.asarray(rollout_b)

        actual_ab = np.asarray(actual_ab)

        a_one = float(one_a.mean())

        b_one = float(one_b.mean())

        a_roll = float(rollout_a.mean())

        b_roll = float(rollout_b.mean())

        physical_gap = float(

            actual_ab.mean()

        )

        relative_one = (

            (b_one - a_one)

            / max(a_one, 1e-12)

        )

        relative_roll = (

            (b_roll - a_roll)

            / max(a_roll, 1e-12)

        )

        time_error = rollout_b.mean(axis=1)

        def first_threshold(threshold):

            hits = np.where(

                time_error >= threshold

            )[0]

            if len(hits) == 0:

                return -1

            return int(hits[0] + 1)

        return {

            "seed": seed,

            "multiplier": multiplier,

            "target_mass": target_mass,

            "before_mass": before_mass,

            "after_mass": after_mass,

            "initial_diff": initial_diff,

            "postmod_diff": postmod_diff,

            "physical_gap": physical_gap,

            "a_one": a_one,

            "b_one": b_one,

            "a_roll": a_roll,

            "b_roll": b_roll,

            "delta_one":

                b_one - a_one,

            "delta_roll":

                b_roll - a_roll,

            "relative_one":

                relative_one,

            "relative_roll":

                relative_roll,

            "b_final":

                float(rollout_b[-1].mean()),

            "b_max":

                float(rollout_b.max()),

            "first_01":

                first_threshold(0.10),

            "first_02":

                first_threshold(0.20),

            "first_05":

                first_threshold(0.50),

            "b_one_dim":

                one_b.mean(axis=0),

            "b_roll_dim":

                rollout_b.mean(axis=0),

        }

    finally:

        env_a.close()

        env_b.close()

DIM_NAMES = []

for i in range(7):

    DIM_NAMES.append(f"J{i+1}_Pos")

for i in range(7):

    DIM_NAMES.append(f"J{i+1}_Vel")

DIM_NAMES.extend(

    [

        "EEF_X",

        "EEF_Y",

        "EEF_Z",

    ]

)

print("=" * 90)

print("V5 — STRUCTURED EXCITATION EMBODIMENT SHIFT")

print("=" * 90)

print()

print("Model :", MODEL_FILE)

print("Seeds :", SEEDS)

print("Horizon :", ROLLOUT_STEPS)

print(

    "Mass multipliers:",

    MASS_MULTIPLIERS

)

print(

    "Position amplitude:",

    POS_AMPLITUDE

)

print(

    "Rotation amplitude:",

    ROT_AMPLITUDE

)

test_env = make_env(SEEDS[0])

try:

    test_env.reset()

    print()

    print("=" * 90)

    print("ACTION SPACE AUDIT")

    print("=" * 90)

    print(

        "env.action_dim =",

        test_env.action_dim

    )

    low, high = test_env.action_spec

    print(

        "action low  =",

        low

    )

    print(

        "action high =",

        high

    )

finally:

    test_env.close()

print()

print("=" * 90)

print("PHASE 1 — SANITY")

print("=" * 90)

all_sanity_pass = True

for seed in SEEDS:

    actions = generate_structured_actions(seed)

    result = run_sanity_test(

        seed,

        actions

    )

    print()

    print("Seed", seed)

    print(

        "  initial:",

        f"{result['initial_diff']:.12e}"

    )

    print(

        "  max step:",

        f"{result['max_step_diff']:.12e}"

    )

    print(

        "  PASS:",

        result["passed"]

    )

    all_sanity_pass &= result["passed"]

if not all_sanity_pass:

    raise RuntimeError(

        "SANITY FAILED — STOP."

    )

print()

print("SANITY PASSED")

env = make_env(SEEDS[0])

try:

    env.reset()

    body_id, body_name = (

        find_link5_body(env)

    )

    baseline_mass = float(

        env.sim.model.body_mass[body_id]

    )

finally:

    env.close()

print()

print("=" * 90)

print("PHASE 2 — MASS")

print("=" * 90)

print(

    "Body:",

    body_name

)

print(

    "Baseline:",

    f"{baseline_mass:.6f} kg"

)

for m in MASS_MULTIPLIERS:

    print(

        f"x{m:.1f} -> "

        f"{baseline_mass*m:.6f} kg"

    )

print()

print("=" * 90)

print("PHASE 3 — STRUCTURED EXCITATION")

print("=" * 90)

results = []

for seed in SEEDS:

    actions = generate_structured_actions(seed)

    for multiplier in MASS_MULTIPLIERS:

        result = run_experiment(

            seed=seed,

            multiplier=multiplier,

            baseline_mass=baseline_mass,

            body_name=body_name,

            actions=actions,

        )

        results.append(result)

        print()

        print(

            f"Seed {seed} | "

            f"Mass x{multiplier:.1f}"

        )

        print(

            "  physical gap :",

            f"{result['physical_gap']:.8f}"

        )

        print(

            "  A 1-step     :",

            f"{result['a_one']:.8f}"

        )

        print(

            "  B 1-step     :",

            f"{result['b_one']:.8f}"

        )

        print(

            "  Δ 1-step     :",

            f"{result['delta_one']:+.8f}"

        )

        print(

            "  A rollout    :",

            f"{result['a_roll']:.8f}"

        )

        print(

            "  B rollout    :",

            f"{result['b_roll']:.8f}"

        )

        print(

            "  Δ rollout    :",

            f"{result['delta_roll']:+.8f}"

        )

        print(

            "  relative roll:",

            f"{result['relative_roll']*100:+.3f}%"

        )

        print(

            "  B final      :",

            f"{result['b_final']:.8f}"

        )

        print(

            "  B max        :",

            f"{result['b_max']:.8f}"

        )

        print(

            "  >=0.10       :",

            result["first_01"]

        )

        print(

            "  >=0.20       :",

            result["first_02"]

        )

        print(

            "  >=0.50       :",

            result["first_05"]

        )

print()

print("=" * 90)

print("V5 SUMMARY")

print("=" * 90)

header = (

    f"{'Mass':>6} | "

    f"{'PhysGap':>10} | "

    f"{'A Roll':>11} | "

    f"{'B Roll':>11} | "

    f"{'ΔRoll':>11} | "

    f"{'Rel%':>9}"

)

print(header)

print("-" * len(header))

for multiplier in MASS_MULTIPLIERS:

    rows = [

        r

        for r in results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    phys = np.mean([

        r["physical_gap"]

        for r in rows

    ])

    a_roll = np.mean([

        r["a_roll"]

        for r in rows

    ])

    b_roll = np.mean([

        r["b_roll"]

        for r in rows

    ])

    delta = b_roll - a_roll

    relative = (

        delta

        / max(a_roll, 1e-12)

        * 100.0

    )

    print(

        f"{multiplier:6.1f} | "

        f"{phys:10.6f} | "

        f"{a_roll:11.8f} | "

        f"{b_roll:11.8f} | "

        f"{delta:+11.8f} | "

        f"{relative:+8.3f}%"

    )

print()

print("=" * 90)

print("FAILURE THRESHOLDS")

print("=" * 90)

for multiplier in MASS_MULTIPLIERS:

    rows = [

        r

        for r in results

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

        (0.10, "first_01"),

        (0.20, "first_02"),

        (0.50, "first_05"),

    ]:

        values = np.asarray([

            r[key]

            for r in rows

        ])

        valid = values[

            values >= 0

        ]

        print(

            f"  >= {threshold:.2f}: "

            f"{len(valid)}/{len(values)} seeds"

        )

        if len(valid):

            print(

                f"      mean step = "

                f"{valid.mean():.2f}"

            )

print()

print("=" * 90)

print("TOP ROBOT-B ERROR DIMENSIONS")

print("=" * 90)

for multiplier in MASS_MULTIPLIERS:

    rows = [

        r

        for r in results

        if np.isclose(

            r["multiplier"],

            multiplier

        )

    ]

    matrix = np.asarray([

        r["b_roll_dim"]

        for r in rows

    ])

    mean_dim = matrix.mean(axis=0)

    order = np.argsort(

        mean_dim

    )[::-1]

    print()

    print(

        f"Mass x{multiplier:.1f}"

    )

    for rank, idx in enumerate(

        order[:5],

        1

    ):

        print(

            f"  {rank}. "

            f"{DIM_NAMES[idx]:8s} "

            f"{mean_dim[idx]:.8f}"

        )

print()

print("=" * 90)

print("ACTION DISTRIBUTION AUDIT")

print("=" * 90)

for seed in SEEDS:

    actions = generate_structured_actions(seed)

    max_abs = float(

        np.max(

            np.abs(actions[:, :6])

        )

    )

    rms = float(

        np.sqrt(

            np.mean(

                actions[:, :6] ** 2

            )

        )

    )

    print()

    print(

        f"Seed {seed}"

    )

    print(

        "  max abs pose action:",

        f"{max_abs:.6f}"

    )

    print(

        "  pose RMS:",

        f"{rms:.6f}"

    )

    print(

        "  gripper unique:",

        np.unique(actions[:, 6])

    )

np.savez(

    "embodiment_shift_v5_excitation_results.npz",

    multipliers=np.asarray(

        MASS_MULTIPLIERS,

        dtype=np.float32

    ),

    baseline_mass=np.float32(

        baseline_mass

    ),

    physical_gap=np.asarray(

        [

            r["physical_gap"]

            for r in results

        ],

        dtype=np.float32

    ),

    a_one=np.asarray(

        [

            r["a_one"]

            for r in results

        ],

        dtype=np.float32

    ),

    b_one=np.asarray(

        [

            r["b_one"]

            for r in results

        ],

        dtype=np.float32

    ),

    a_roll=np.asarray(

        [

            r["a_roll"]

            for r in results

        ],

        dtype=np.float32

    ),

    b_roll=np.asarray(

        [

            r["b_roll"]

            for r in results

        ],

        dtype=np.float32

    ),

    delta_one=np.asarray(

        [

            r["delta_one"]

            for r in results

        ],

        dtype=np.float32

    ),

    delta_roll=np.asarray(

        [

            r["delta_roll"]

            for r in results

        ],

        dtype=np.float32

    ),

    relative_roll=np.asarray(

        [

            r["relative_roll"]

            for r in results

        ],

        dtype=np.float32

    ),

)

np.save(

    "v5_structured_actions.npy",

    generate_structured_actions(SEEDS[0])

)

print()

print("=" * 90)

print("SAVED")

print("=" * 90)

print(

    "embodiment_shift_v5_excitation_results.npz"

)

print(

    "v5_structured_actions.npy"

)

print()

print("=" * 90)

print("DONE")

print("=" * 90)
