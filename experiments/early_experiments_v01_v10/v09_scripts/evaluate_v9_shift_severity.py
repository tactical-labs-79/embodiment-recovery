import copy

import logging

import os

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

import mujoco

from robosuite.utils.mjmod import DynamicsModder

MODEL_FILE = "dynamics_model_v7.pt"

SHIFT_MULTIPLIERS = [

    2.0,

    3.0,

    5.0,

]

ADAPT_SEEDS = list(range(9101, 9121))

VALIDATION_SEEDS = [

    9301,

    9302,

    9303,

    9304,

    9305,

]

CONFIRMATION_SEEDS = [

    9501,

    9502,

    9503,

    9504,

    9505,

]

BUDGETS = [

    5,

    10,

]

SUBSET_REPEATS = 5

ROLLOUT_STEPS = 120

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

POS_AMPLITUDE = 0.18

ROT_AMPLITUDE = 0.12

LEARNING_RATE = 1e-4

MAX_EPOCHS = 100

PATIENCE = 10

BATCH_SIZE = 256

WEIGHT_DECAY = 1e-6

MIN_DELTA = 1e-7

DEVICE = torch.device("cpu")

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

def load_v7_model():

    model = DynamicsModel().to(

        DEVICE

    )

    checkpoint = torch.load(

        MODEL_FILE,

        map_location=DEVICE

    )

    if (

        isinstance(checkpoint, dict)

        and "model_state_dict" in checkpoint

    ):

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

def freeze_backbone_train_head(model):

    for param in model.parameters():

        param.requires_grad = False

    for param in model.network[-1].parameters():

        param.requires_grad = True

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

def find_link5_body(env):

    model = env.sim.model._model

    matches = []

    for body_id in range(

        model.nbody

    ):

        name = mujoco.mj_id2name(

            model,

            mujoco.mjtObj.mjOBJ_BODY,

            body_id

        )

        if (

            name is not None

            and name.endswith("link5")

        ):

            matches.append(

                (

                    body_id,

                    name

                )

            )

    if len(matches) != 1:

        raise RuntimeError(

            f"Expected one link5 body, got {matches}"

        )

    return matches[0]

def get_mass_config():

    env = make_env(

        ADAPT_SEEDS[0]

    )

    try:

        env.reset()

        body_id, body_name = (

            find_link5_body(env)

        )

        baseline_mass = float(

            env.sim.model.body_mass[

                body_id

            ]

        )

        return (

            body_name,

            baseline_mass

        )

    finally:

        env.close()

def apply_mass_shift(

    env,

    body_name,

    baseline_mass,

    multiplier

):

    target_mass = (

        baseline_mass

        * multiplier

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

    env.sim.forward()

    body_id = mujoco.mj_name2id(

        env.sim.model._model,

        mujoco.mjtObj.mjOBJ_BODY,

        body_name

    )

    actual_mass = float(

        env.sim.model.body_mass[

            body_id

        ]

    )

    if not np.isclose(

        actual_mass,

        target_mass,

        atol=1e-7

    ):

        raise RuntimeError(

            f"Mass mismatch: "

            f"{actual_mass} vs {target_mass}"

        )

    return actual_mass

def collect_b_trajectory(

    seed,

    body_name,

    baseline_mass,

    multiplier

):

    env = make_env(

        seed

    )

    try:

        env.reset()

        apply_mass_shift(

            env,

            body_name,

            baseline_mass,

            multiplier

        )

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

        inputs = []

        targets = []

        for step in range(

            ROLLOUT_STEPS

        ):

            action = actions[step]

            model_input = np.concatenate(

                [

                    state,

                    action

                ]

            ).astype(

                np.float32

            )

            next_obs, _, _, _ = (

                env.step(action)

            )

            next_state = extract_state(

                next_obs

            )

            inputs.append(

                model_input

            )

            targets.append(

                next_state

            )

            state = next_state

        return {

            "seed":

                seed,

            "inputs":

                np.asarray(

                    inputs,

                    dtype=np.float32

                ),

            "targets":

                np.asarray(

                    targets,

                    dtype=np.float32

                ),

        }

    finally:

        env.close()

def build_b_pool(

    seeds,

    body_name,

    baseline_mass,

    multiplier

):

    pool = []

    for seed in seeds:

        print(

            f"Collecting B "

            f"x{multiplier:.1f} "

            f"seed={seed}"

        )

        pool.append(

            collect_b_trajectory(

                seed,

                body_name,

                baseline_mass,

                multiplier

            )

        )

    return pool

def evaluate_model_on_b(

    model,

    seed,

    body_name,

    baseline_mass,

    multiplier

):

    env = make_env(

        seed

    )

    try:

        env.reset()

        apply_mass_shift(

            env,

            body_name,

            baseline_mass,

            multiplier

        )

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

        teacher_errors = []

        rollout_errors = []

        free_state = state.copy()

        for step in range(

            ROLLOUT_STEPS

        ):

            action = actions[step]

            teacher_input = np.concatenate(

                [

                    state,

                    action

                ]

            ).astype(

                np.float32

            )

            with torch.no_grad():

                teacher_prediction = (

                    model(

                        torch.from_numpy(

                            teacher_input

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

            teacher_errors.append(

                np.abs(

                    teacher_prediction

                    - next_state

                )

            )

            free_input = np.concatenate(

                [

                    free_state,

                    action

                ]

            ).astype(

                np.float32

            )

            with torch.no_grad():

                free_prediction = (

                    model(

                        torch.from_numpy(

                            free_input

                        ).unsqueeze(0)

                    )

                    .cpu()

                    .numpy()[0]

                )

            rollout_errors.append(

                np.abs(

                    free_prediction

                    - next_state

                )

            )

            free_state = (

                free_prediction.copy()

            )

            state = next_state

        teacher_errors = np.asarray(

            teacher_errors,

            dtype=np.float32

        )

        rollout_errors = np.asarray(

            rollout_errors,

            dtype=np.float32

        )

        return {

            "teacher":

                float(

                    teacher_errors.mean()

                ),

            "rollout":

                float(

                    rollout_errors.mean()

                ),

            "final":

                float(

                    rollout_errors[-1].mean()

                ),

            "max":

                float(

                    rollout_errors.max()

                ),

        }

    finally:

        env.close()

def train_head_only(

    base_state_dict,

    train_trajectories,

    validation_fn,

    run_seed

):

    model = DynamicsModel().to(

        DEVICE

    )

    model.load_state_dict(

        copy.deepcopy(

            base_state_dict

        )

    )

    freeze_backbone_train_head(

        model

    )

    inputs = np.concatenate(

        [

            t["inputs"]

            for t in train_trajectories

        ],

        axis=0

    )

    targets = np.concatenate(

        [

            t["targets"]

            for t in train_trajectories

        ],

        axis=0

    )

    x = torch.from_numpy(

        inputs

    )

    y = torch.from_numpy(

        targets

    )

    dataset = torch.utils.data.TensorDataset(

        x,

        y

    )

    generator = torch.Generator()

    generator.manual_seed(

        run_seed

    )

    loader = torch.utils.data.DataLoader(

        dataset,

        batch_size=min(

            BATCH_SIZE,

            len(dataset)

        ),

        shuffle=True,

        generator=generator

    )

    optimizer = torch.optim.Adam(

        [

            p

            for p in model.parameters()

            if p.requires_grad

        ],

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    criterion = nn.MSELoss()

    best_state = copy.deepcopy(

        model.state_dict()

    )

    best_val = float("inf")

    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(

        1,

        MAX_EPOCHS + 1

    ):

        model.train()

        running_loss = 0.0

        count = 0

        for batch_x, batch_y in loader:

            optimizer.zero_grad(

                set_to_none=True

            )

            prediction = model(

                batch_x

            )

            loss = criterion(

                prediction,

                batch_y

            )

            loss.backward()

            optimizer.step()

            n = batch_x.shape[0]

            running_loss += (

                loss.item() * n

            )

            count += n

        train_loss = (

            running_loss

            / max(count, 1)

        )

        val_rollout = validation_fn(

            model

        )

        history.append(

            (

                epoch,

                train_loss,

                val_rollout

            )

        )

        improved = (

            val_rollout

            < best_val - MIN_DELTA

        )

        if improved:

            best_val = val_rollout

            best_epoch = epoch

            best_state = copy.deepcopy(

                model.state_dict()

            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (

            epoch == 1

            or epoch % 10 == 0

            or improved

        ):

            print(

                f"Epoch {epoch:03d} | "

                f"train={train_loss:.10f} | "

                f"val={val_rollout:.8f} | "

                f"best={best_val:.8f}"

            )

        if (

            patience_counter

            >= PATIENCE

        ):

            print(

                f"Early stopping at epoch "

                f"{epoch}"

            )

            break

    model.load_state_dict(

        best_state

    )

    model.eval()

    return (

        model,

        history,

        best_epoch,

        best_val

    )

def build_subset_specs():

    specs = []

    for budget in BUDGETS:

        for repeat in range(

            1,

            SUBSET_REPEATS + 1

        ):

            rng = np.random.default_rng(

                82000

                + budget * 100

                + repeat

            )

            selected = rng.choice(

                ADAPT_SEEDS,

                size=budget,

                replace=False

            )

            selected = sorted(

                selected.tolist()

            )

            specs.append(

                {

                    "budget":

                        budget,

                    "repeat":

                        repeat,

                    "seeds":

                        selected,

                }

            )

    return specs

print("=" * 100)

print(

    "V9 — SHIFT SEVERITY GENERALIZATION"

)

print("=" * 100)

print()

print(

    "Shift multipliers:",

    SHIFT_MULTIPLIERS

)

print(

    "Budgets:",

    BUDGETS

)

print(

    "Subset repeats:",

    SUBSET_REPEATS

)

print(

    "Validation:",

    VALIDATION_SEEDS

)

print(

    "Fresh confirmation:",

    CONFIRMATION_SEEDS

)

(

    body_name,

    baseline_mass

) = get_mass_config()

print()

print("=" * 100)

print(

    "MASS CONFIG"

)

print("=" * 100)

print(

    "Body:",

    body_name

)

print(

    "Baseline mass:",

    f"{baseline_mass:.6f} kg"

)

for multiplier in SHIFT_MULTIPLIERS:

    print(

        f"x{multiplier:.1f}: "

        f"{baseline_mass * multiplier:.6f} kg"

    )

base_model = load_v7_model()

base_state_dict = copy.deepcopy(

    base_model.state_dict()

)

subset_specs = build_subset_specs()

print()

print("=" * 100)

print(

    "FIXED SUBSETS"

)

print("=" * 100)

for spec in subset_specs:

    print(

        f"Budget {spec['budget']:2d} | "

        f"Subset {spec['repeat']} | "

        f"{spec['seeds']}"

    )

all_results = []

for multiplier in SHIFT_MULTIPLIERS:

    print()

    print("#" * 100)

    print(

        f"SHIFT ×{multiplier:.1f}"

    )

    print("#" * 100)

    adaptation_pool = build_b_pool(

        ADAPT_SEEDS,

        body_name,

        baseline_mass,

        multiplier

    )

    seed_to_trajectory = {

        t["seed"]: t

        for t in adaptation_pool

    }

    print()

    print("=" * 100)

    print(

        f"ZERO-SHOT CONFIRMATION ×{multiplier:.1f}"

    )

    print("=" * 100)

    zero_results = []

    for seed in CONFIRMATION_SEEDS:

        result = evaluate_model_on_b(

            base_model,

            seed,

            body_name,

            baseline_mass,

            multiplier

        )

        zero_results.append(

            result

        )

        print(

            f"Seed {seed}: "

            f"rollout={result['rollout']:.8f}"

        )

    zero_rollout = float(

        np.mean(

            [

                r["rollout"]

                for r in zero_results

            ]

        )

    )

    print()

    print(

        "Zero-shot confirmation:",

        f"{zero_rollout:.8f}"

    )

    for spec in subset_specs:

        budget = spec["budget"]

        repeat = spec["repeat"]

        selected_seeds = spec["seeds"]

        print()

        print("=" * 100)

        print(

            f"x{multiplier:.1f} | "

            f"Budget {budget} | "

            f"Subset {repeat}"

        )

        print("=" * 100)

        print(

            "Adaptation seeds:",

            selected_seeds

        )

        train_trajectories = [

            seed_to_trajectory[seed]

            for seed in selected_seeds

        ]

        def validation_fn(candidate):

            candidate.eval()

            values = []

            for seed in VALIDATION_SEEDS:

                result = evaluate_model_on_b(

                    candidate,

                    seed,

                    body_name,

                    baseline_mass,

                    multiplier

                )

                values.append(

                    result["rollout"]

                )

            return float(

                np.mean(values)

            )

        run_seed = (

            900000

            + int(multiplier * 10000)

            + budget * 100

            + repeat

        )

        (

            model,

            history,

            best_epoch,

            best_val

        ) = train_head_only(

            base_state_dict,

            train_trajectories,

            validation_fn,

            run_seed

        )

        val_results = []

        for seed in VALIDATION_SEEDS:

            val_results.append(

                evaluate_model_on_b(

                    model,

                    seed,

                    body_name,

                    baseline_mass,

                    multiplier

                )

            )

        val_rollout = float(

            np.mean(

                [

                    r["rollout"]

                    for r in val_results

                ]

            )

        )

        confirmation_results = []

        for seed in CONFIRMATION_SEEDS:

            confirmation_results.append(

                evaluate_model_on_b(

                    model,

                    seed,

                    body_name,

                    baseline_mass,

                    multiplier

                )

            )

        confirmation_rollout = float(

            np.mean(

                [

                    r["rollout"]

                    for r in confirmation_results

                ]

            )

        )

        reduction = (

            zero_rollout

            - confirmation_rollout

        )

        recovery_pct = (

            reduction

            / max(

                zero_rollout,

                1e-12

            )

            * 100.0

        )

        confirmation_per_seed = np.asarray(

            [

                r["rollout"]

                for r in confirmation_results

            ],

            dtype=np.float32

        )

        result = {

            "multiplier":

                multiplier,

            "budget":

                budget,

            "repeat":

                repeat,

            "best_epoch":

                best_epoch,

            "best_val":

                best_val,

            "val_rollout":

                val_rollout,

            "zero_rollout":

                zero_rollout,

            "confirmation_rollout":

                confirmation_rollout,

            "confirmation_std":

                float(

                    confirmation_per_seed.std(

                        ddof=1

                    )

                ),

            "reduction":

                reduction,

            "recovery_pct":

                recovery_pct,

        }

        all_results.append(

            result

        )

        checkpoint_name = (

            f"v9_x{multiplier:.0f}_"

            f"{budget}traj_"

            f"subset{repeat}.pt"

        )

        torch.save(

            model.state_dict(),

            checkpoint_name

        )

        print()

        print(

            "RESULT"

        )

        print(

            "  Best epoch:",

            best_epoch

        )

        print(

            "  Validation:",

            f"{val_rollout:.8f}"

        )

        print(

            "  Zero-shot:",

            f"{zero_rollout:.8f}"

        )

        print(

            "  Confirmation:",

            f"{confirmation_rollout:.8f}"

        )

        print(

            "  Recovery:",

            f"{recovery_pct:+.3f}%"

        )

        print(

            "  Confirmation std:",

            f"{result['confirmation_std']:.8f}"

        )

        print(

            "  Saved:",

            checkpoint_name

        )

print()

print("=" * 100)

print(

    "V9 SHIFT SEVERITY SUMMARY"

)

print("=" * 100)

for multiplier in SHIFT_MULTIPLIERS:

    print()

    print(

        f"SHIFT ×{multiplier:.1f}"

    )

    for budget in BUDGETS:

        subset_results = [

            r

            for r in all_results

            if (

                r["multiplier"]

                == multiplier

                and

                r["budget"]

                == budget

            )

        ]

        recoveries = np.asarray(

            [

                r["recovery_pct"]

                for r in subset_results

            ],

            dtype=np.float64

        )

        confirmations = np.asarray(

            [

                r["confirmation_rollout"]

                for r in subset_results

            ],

            dtype=np.float64

        )

        print(

            f"  {budget:2d} traj | "

            f"mean={recoveries.mean():+.3f}% | "

            f"std={recoveries.std(ddof=1):.3f}% | "

            f"min={recoveries.min():+.3f}% | "

            f"max={recoveries.max():+.3f}% | "

            f"positive="

            f"{np.sum(recoveries > 0)}/"

            f"{len(recoveries)}"

        )

        print(

            f"           "

            f"confirm="

            f"{confirmations.mean():.8f}"

        )

print()

print("=" * 100)

print(

    "CROSS-SHIFT RECOVERY TABLE"

)

print("=" * 100)

header = (

    f"{'Shift':>8} | "

    f"{'Budget':>8} | "

    f"{'MeanRec':>10} | "

    f"{'Std':>8} | "

    f"{'Min':>10} | "

    f"{'Max':>10} | "

    f"{'Positive':>9}"

)

print(header)

print(

    "-" * len(header)

)

for multiplier in SHIFT_MULTIPLIERS:

    for budget in BUDGETS:

        subset_results = [

            r

            for r in all_results

            if (

                r["multiplier"]

                == multiplier

                and

                r["budget"]

                == budget

            )

        ]

        recoveries = np.asarray(

            [

                r["recovery_pct"]

                for r in subset_results

            ],

            dtype=np.float64

        )

        print(

            f"{multiplier:8.1f} | "

            f"{budget:8d} | "

            f"{recoveries.mean():+9.3f}% | "

            f"{recoveries.std(ddof=1):8.3f} | "

            f"{recoveries.min():+9.3f}% | "

            f"{recoveries.max():+9.3f}% | "

            f"{np.sum(recoveries > 0):4d}/"

            f"{len(recoveries):<4d}"

        )

np.savez(

    "embodiment_recovery_v9_shift_severity_results.npz",

    multiplier=np.asarray(

        [

            r["multiplier"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    budget=np.asarray(

        [

            r["budget"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    repeat=np.asarray(

        [

            r["repeat"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    best_epoch=np.asarray(

        [

            r["best_epoch"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    best_val=np.asarray(

        [

            r["best_val"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    val_rollout=np.asarray(

        [

            r["val_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    zero_rollout=np.asarray(

        [

            r["zero_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    confirmation_rollout=np.asarray(

        [

            r["confirmation_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    confirmation_std=np.asarray(

        [

            r["confirmation_std"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    reduction=np.asarray(

        [

            r["reduction"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    recovery_pct=np.asarray(

        [

            r["recovery_pct"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    baseline_mass=np.float32(

        baseline_mass

    ),

)

print()

print("=" * 100)

print(

    "SAVED"

)

print("=" * 100)

print(

    "embodiment_recovery_v9_shift_severity_results.npz"

)

print()

print(

    "DONE"

)
