import copy

import logging

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

import mujoco

from robosuite.utils.mjmod import DynamicsModder

MODEL_FILE = "dynamics_model_v7.pt"

MASS_MULTIPLIER = 5.0

ROLLOUT_STEPS = 120

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

ADAPT_SEEDS = list(range(9101, 9121))

VALIDATION_SEEDS = [

    9301, 9302, 9303, 9304, 9305

]

CONFIRMATION_SEEDS = [

    9401, 9402, 9403, 9404, 9405

]

BUDGETS = [

    1,

    5,

    10,

    20,

]

SUBSET_REPEATS = 5

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

    if "model_state_dict" in checkpoint:

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

    return model

def freeze_backbone_train_head(model):

    for param in model.parameters():

        param.requires_grad = False

    for param in model.network[-1].parameters():

        param.requires_grad = True

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

                actions.append(action)

            for _ in range(6):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = -amplitude

                actions.append(action)

            for _ in range(3):

                if len(actions) >= ROLLOUT_STEPS:

                    break

                action = np.zeros(

                    7,

                    dtype=np.float32

                )

                action[dim] = amplitude

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

def apply_mass_shift(

    env,

    body_name,

    baseline_mass

):

    target_mass = (

        baseline_mass

        * MASS_MULTIPLIER

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

    return float(

        env.sim.model.body_mass[

            body_id

        ]

    )

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

        target_mass = (

            baseline_mass

            * MASS_MULTIPLIER

        )

        return (

            body_name,

            baseline_mass,

            target_mass

        )

    finally:

        env.close()

def collect_b_trajectory(

    seed,

    body_name,

    baseline_mass

):

    env = make_env(

        seed

    )

    try:

        obs = env.reset()

        actual_mass = apply_mass_shift(

            env,

            body_name,

            baseline_mass

        )

        expected_mass = (

            baseline_mass

            * MASS_MULTIPLIER

        )

        if not np.isclose(

            actual_mass,

            expected_mass,

            atol=1e-7

        ):

            raise RuntimeError(

                f"Mass mismatch: "

                f"{actual_mass} vs {expected_mass}"

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

            next_obs, _, _, _ = env.step(

                action

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

            "seed": seed,

            "inputs": np.asarray(

                inputs,

                dtype=np.float32

            ),

            "targets": np.asarray(

                targets,

                dtype=np.float32

            ),

        }

    finally:

        env.close()

def build_b_pool(

    seeds,

    body_name,

    baseline_mass

):

    pool = []

    for seed in seeds:

        print(

            f"Collecting B trajectory seed={seed}"

        )

        pool.append(

            collect_b_trajectory(

                seed,

                body_name,

                baseline_mass

            )

        )

    return pool

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

        if patience_counter >= PATIENCE:

            print(

                f"Early stopping at epoch {epoch}"

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

def evaluate_model_on_b(

    model,

    seed,

    body_name,

    baseline_mass

):

    env = make_env(

        seed

    )

    try:

        obs = env.reset()

        actual_mass = apply_mass_shift(

            env,

            body_name,

            baseline_mass

        )

        expected_mass = (

            baseline_mass

            * MASS_MULTIPLIER

        )

        if not np.isclose(

            actual_mass,

            expected_mass,

            atol=1e-7

        ):

            raise RuntimeError(

                "Mass was not applied correctly."

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

            next_obs, _, _, _ = env.step(

                action

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

            free_state = free_prediction.copy()

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

            "teacher_mae":

                float(

                    teacher_errors.mean()

                ),

            "rollout_mae":

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

def evaluate_model_on_a(

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

            next_obs, _, _, _ = env.step(

                action

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

            state = next_state

        errors = np.asarray(

            errors,

            dtype=np.float32

        )

        return float(

            errors.mean()

        )

    finally:

        env.close()

def mean_result(results):

    return float(

        np.mean(

            [

                x["rollout_mae"]

                for x in results

            ]

        )

    )

print("=" * 90)

print(

    "V8.2 — ADAPTATION SUBSET STABILITY"

)

print("=" * 90)

print()

print(

    "Adaptation pool:",

    ADAPT_SEEDS

)

print(

    "Validation:",

    VALIDATION_SEEDS

)

print(

    "Fresh confirmation:",

    CONFIRMATION_SEEDS

)

print(

    "Budgets:",

    BUDGETS

)

print(

    "Subset repeats:",

    SUBSET_REPEATS

)

(

    body_name,

    baseline_mass,

    target_mass

) = get_mass_config()

print()

print("=" * 90)

print("MASS CONFIG")

print("=" * 90)

print(

    "Body:",

    body_name

)

print(

    "Baseline:",

    f"{baseline_mass:.6f} kg"

)

print(

    "Robot B:",

    f"{target_mass:.6f} kg"

)

base_model = load_v7_model()

base_state_dict = copy.deepcopy(

    base_model.state_dict()

)

print()

print("=" * 90)

print("BUILDING ADAPTATION POOL")

print("=" * 90)

adaptation_pool = build_b_pool(

    ADAPT_SEEDS,

    body_name,

    baseline_mass

)

print()

print("=" * 90)

print("ZERO-SHOT CONFIRMATION")

print("=" * 90)

zero_confirmation = []

for seed in CONFIRMATION_SEEDS:

    result = evaluate_model_on_b(

        base_model,

        seed,

        body_name,

        baseline_mass

    )

    zero_confirmation.append(

        result

    )

zero_confirmation_rollout = mean_result(

    zero_confirmation

)

print(

    "Confirmation zero-shot:",

    f"{zero_confirmation_rollout:.8f}"

)

subset_specs = []

for budget in BUDGETS:

    if budget == len(ADAPT_SEEDS):

        subset_specs.append(

            (

                budget,

                0,

                ADAPT_SEEDS.copy()

            )

        )

        continue

    for repeat in range(

        SUBSET_REPEATS

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

        subset_specs.append(

            (

                budget,

                repeat + 1,

                selected

            )

        )

results = []

for budget, repeat, selected_seeds in subset_specs:

    print()

    print("=" * 90)

    if repeat == 0:

        print(

            f"BUDGET {budget} — FULL POOL"

        )

    else:

        print(

            f"BUDGET {budget} — "

            f"SUBSET {repeat}/{SUBSET_REPEATS}"

        )

    print("=" * 90)

    print(

        "Adaptation seeds:",

        selected_seeds

    )

    train_trajectories = []

    seed_to_trajectory = {

        t["seed"]: t

        for t in adaptation_pool

    }

    for seed in selected_seeds:

        train_trajectories.append(

            seed_to_trajectory[seed]

        )

    def validation_fn(candidate):

        candidate.eval()

        values = []

        for seed in VALIDATION_SEEDS:

            result = evaluate_model_on_b(

                candidate,

                seed,

                body_name,

                baseline_mass

            )

            values.append(

                result["rollout_mae"]

            )

        return float(

            np.mean(values)

        )

    training_seed = (

        600000

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

        training_seed

    )

    val_results = []

    for seed in VALIDATION_SEEDS:

        val_results.append(

            evaluate_model_on_b(

                model,

                seed,

                body_name,

                baseline_mass

            )

        )

    val_rollout = mean_result(

        val_results

    )

    confirmation_results = []

    for seed in CONFIRMATION_SEEDS:

        confirmation_results.append(

            evaluate_model_on_b(

                model,

                seed,

                body_name,

                baseline_mass

            )

        )

    confirmation_rollout = mean_result(

        confirmation_results

    )

    a_values = []

    for seed in CONFIRMATION_SEEDS:

        a_values.append(

            evaluate_model_on_a(

                model,

                seed

            )

        )

    a_rollout = float(

        np.mean(a_values)

    )

    reduction = (

        zero_confirmation_rollout

        - confirmation_rollout

    )

    recovery_pct = (

        reduction

        / max(

            zero_confirmation_rollout,

            1e-12

        )

        * 100.0

    )

    result = {

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

        "confirmation_rollout":

            confirmation_rollout,

        "recovery_pct":

            recovery_pct,

        "a_rollout":

            a_rollout,

    }

    results.append(

        result

    )

    checkpoint_name = (

        f"v8_2_"

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

        "  Confirmation:",

        f"{confirmation_rollout:.8f}"

    )

    print(

        "  Recovery:",

        f"{recovery_pct:+.3f}%"

    )

    print(

        "  A rollout:",

        f"{a_rollout:.8f}"

    )

    print(

        "  Saved:",

        checkpoint_name

    )

print()

print("=" * 90)

print("V8.2 SUBSET STABILITY SUMMARY")

print("=" * 90)

for budget in BUDGETS:

    budget_results = [

        r

        for r in results

        if r["budget"] == budget

    ]

    recoveries = np.asarray(

        [

            r["recovery_pct"]

            for r in budget_results

        ],

        dtype=np.float64

    )

    confirmations = np.asarray(

        [

            r["confirmation_rollout"]

            for r in budget_results

        ],

        dtype=np.float64

    )

    a_values = np.asarray(

        [

            r["a_rollout"]

            for r in budget_results

        ],

        dtype=np.float64

    )

    print()

    print(

        f"Budget = {budget}"

    )

    print(

        "  Repeats:",

        len(budget_results)

    )

    print(

        "  Confirmation mean:",

        f"{confirmations.mean():.8f}"

    )

    print(

        "  Confirmation std:",

        f"{confirmations.std(ddof=1) if len(confirmations) > 1 else 0.0:.8f}"

    )

    print(

        "  Recovery mean:",

        f"{recoveries.mean():+.3f}%"

    )

    print(

        "  Recovery std:",

        f"{recoveries.std(ddof=1) if len(recoveries) > 1 else 0.0:.3f}%"

    )

    print(

        "  Recovery min:",

        f"{recoveries.min():+.3f}%"

    )

    print(

        "  Recovery max:",

        f"{recoveries.max():+.3f}%"

    )

    print(

        "  Positive recovery:",

        f"{np.sum(recoveries > 0)}/{len(recoveries)}"

    )

    print(

        "  A rollout mean:",

        f"{a_values.mean():.8f}"

    )

print()

print("=" * 90)

print("DETAILED SUBSET RESULTS")

print("=" * 90)

header = (

    f"{'Budget':>8} | "

    f"{'Subset':>7} | "

    f"{'BestEp':>7} | "

    f"{'Val':>11} | "

    f"{'Confirm':>11} | "

    f"{'Recovery':>10} | "

    f"{'A Roll':>11}"

)

print(header)

print(

    "-" * len(header)

)

for r in results:

    print(

        f"{r['budget']:8d} | "

        f"{r['repeat']:7d} | "

        f"{r['best_epoch']:7d} | "

        f"{r['val_rollout']:11.8f} | "

        f"{r['confirmation_rollout']:11.8f} | "

        f"{r['recovery_pct']:+9.3f}% | "

        f"{r['a_rollout']:11.8f}"

    )

np.savez(

    "embodiment_recovery_v8_2_subset_stability_results.npz",

    budget=np.asarray(

        [

            r["budget"]

            for r in results

        ],

        dtype=np.int32

    ),

    repeat=np.asarray(

        [

            r["repeat"]

            for r in results

        ],

        dtype=np.int32

    ),

    best_epoch=np.asarray(

        [

            r["best_epoch"]

            for r in results

        ],

        dtype=np.int32

    ),

    val_rollout=np.asarray(

        [

            r["val_rollout"]

            for r in results

        ],

        dtype=np.float32

    ),

    confirmation_rollout=np.asarray(

        [

            r["confirmation_rollout"]

            for r in results

        ],

        dtype=np.float32

    ),

    recovery_pct=np.asarray(

        [

            r["recovery_pct"]

            for r in results

        ],

        dtype=np.float32

    ),

    a_rollout=np.asarray(

        [

            r["a_rollout"]

            for r in results

        ],

        dtype=np.float32

    ),

    zero_confirmation=np.float32(

        zero_confirmation_rollout

    ),

    mass_multiplier=np.float32(

        MASS_MULTIPLIER

    ),

)

print()

print("=" * 90)

print("SAVED")

print("=" * 90)

print(

    "embodiment_recovery_v8_2_subset_stability_results.npz"

)

print()

print("DONE")
