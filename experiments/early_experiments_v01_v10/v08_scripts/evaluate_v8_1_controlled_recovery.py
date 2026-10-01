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

ADAPT_SEEDS = [

    9101, 9102, 9103, 9104, 9105,

    9106, 9107, 9108, 9109, 9110,

    9111, 9112, 9113, 9114, 9115,

    9116, 9117, 9118, 9119, 9120,

]

VALIDATION_SEEDS = [

    9301,

    9302,

    9303,

    9304,

    9305,

]

TEST_SEEDS = [

    9201,

    9202,

    9203,

    9204,

    9205,

]

BUDGETS = [

    0,

    1,

    5,

    10,

    20,

]

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

        state_dict = (

            checkpoint[

                "model_state_dict"

            ]

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

            f"Expected one link5 body, "

            f"got {matches}"

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

    actual_mass = float(

        env.sim.model.body_mass[

            body_id

        ]

    )

    return actual_mass

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

            env=env,

            body_name=body_name,

            baseline_mass=baseline_mass

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

                f"{actual_mass} "

                f"vs "

                f"{expected_mass}"

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

            action = actions[

                step

            ]

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

            f"Collecting B trajectory "

            f"seed={seed}"

        )

        trajectory = (

            collect_b_trajectory(

                seed=seed,

                body_name=body_name,

                baseline_mass=baseline_mass

            )

        )

        pool.append(

            trajectory

        )

    return pool

def train_head_only(

    base_state_dict,

    train_trajectories,

    val_fn

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

    ).to(

        DEVICE

    )

    y = torch.from_numpy(

        targets

    ).to(

        DEVICE

    )

    dataset = torch.utils.data.TensorDataset(

        x,

        y

    )

    generator = torch.Generator()

    generator.manual_seed(

        50000 + len(train_trajectories)

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

    best_val = float(

        "inf"

    )

    best_epoch = 0

    patience_counter = 0

    print()

    print(

        "Trainable parameters:"

    )

    trainable = 0

    total = 0

    for param in model.parameters():

        n = param.numel()

        total += n

        if param.requires_grad:

            trainable += n

    print(

        "  trainable:",

        trainable

    )

    print(

        "  total:",

        total

    )

    print(

        "  trainable %:",

        f"{100 * trainable / total:.3f}%"

    )

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

            batch_n = (

                batch_x.shape[0]

            )

            running_loss += (

                loss.item()

                * batch_n

            )

            count += batch_n

        train_loss = (

            running_loss

            / max(count, 1)

        )

        val_rollout = val_fn(

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

                f"val_rollout={val_rollout:.8f} | "

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

            env=env,

            body_name=body_name,

            baseline_mass=baseline_mass

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

            action = actions[

                step

            ]

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

            action = actions[

                step

            ]

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

def evaluate_split(

    model,

    seeds,

    body_name,

    baseline_mass

):

    results = []

    for seed in seeds:

        results.append(

            evaluate_model_on_b(

                model=model,

                seed=seed,

                body_name=body_name,

                baseline_mass=baseline_mass

            )

        )

    return results

def summarize_results(results):

    return {

        "teacher":

            float(

                np.mean(

                    [

                        r["teacher_mae"]

                        for r in results

                    ]

                )

            ),

        "rollout":

            float(

                np.mean(

                    [

                        r["rollout_mae"]

                        for r in results

                    ]

                )

            ),

        "final":

            float(

                np.mean(

                    [

                        r["final"]

                        for r in results

                    ]

                )

            ),

        "max":

            float(

                np.mean(

                    [

                        r["max"]

                        for r in results

                    ]

                )

            ),

    }

print("=" * 90)

print(

    "V8.1 — CONTROLLED FEW-SHOT EMBODIMENT RECOVERY"

)

print("=" * 90)

print()

print(

    "Base model:",

    MODEL_FILE

)

print(

    "Robot B mass:",

    f"x{MASS_MULTIPLIER:.1f}"

)

print(

    "Adaptation seeds:",

    ADAPT_SEEDS

)

print(

    "Validation seeds:",

    VALIDATION_SEEDS

)

print(

    "Test seeds:",

    TEST_SEEDS

)

print(

    "Budgets:",

    BUDGETS

)

print(

    "Max epochs:",

    MAX_EPOCHS

)

print(

    "Patience:",

    PATIENCE

)

(

    body_name,

    baseline_mass,

    target_mass

) = get_mass_config()

print()

print("=" * 90)

print(

    "MASS CONFIGURATION"

)

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

print(

    "PHASE 1 — DATA"

)

print("=" * 90)

adaptation_pool = build_b_pool(

    seeds=ADAPT_SEEDS,

    body_name=body_name,

    baseline_mass=baseline_mass

)

validation_pool = build_b_pool(

    seeds=VALIDATION_SEEDS,

    body_name=body_name,

    baseline_mass=baseline_mass

)

print()

print(

    "Adaptation transitions:",

    len(adaptation_pool)

    * ROLLOUT_STEPS

)

print(

    "Validation transitions:",

    len(validation_pool)

    * ROLLOUT_STEPS

)

print()

print("=" * 90)

print(

    "PHASE 2 — ZERO-SHOT TEST BASELINE"

)

print("=" * 90)

zero_test_results = evaluate_split(

    model=base_model,

    seeds=TEST_SEEDS,

    body_name=body_name,

    baseline_mass=baseline_mass

)

zero_summary = summarize_results(

    zero_test_results

)

print()

print(

    "Zero-shot test:"

)

print(

    "  Teacher:",

    f"{zero_summary['teacher']:.8f}"

)

print(

    "  Rollout:",

    f"{zero_summary['rollout']:.8f}"

)

all_results = []

for budget in BUDGETS:

    print()

    print("=" * 90)

    print(

        f"BUDGET = {budget} TRAJECTORIES"

    )

    print("=" * 90)

    if budget == 0:

        model = load_v7_model()

        best_epoch = 0

        best_val = zero_summary[

            "rollout"

        ]

        train_loss = 0.0

    else:

        train_trajectories = (

            adaptation_pool[

                :budget

            ]

        )

        def validation_fn(candidate):

            candidate.eval()

            results = []

            with torch.no_grad():

                for seed in VALIDATION_SEEDS:

                    result = (

                        evaluate_model_on_b(

                            model=candidate,

                            seed=seed,

                            body_name=body_name,

                            baseline_mass=baseline_mass

                        )

                    )

                    results.append(

                        result[

                            "rollout_mae"

                        ]

                    )

            return float(

                np.mean(results)

            )

        model, history, best_epoch, best_val = (

            train_head_only(

                base_state_dict=base_state_dict,

                train_trajectories=train_trajectories,

                val_fn=validation_fn

            )

        )

        train_loss = history[-1][1]

        save_path = (

            f"v8_1_headonly_"

            f"{budget}traj.pt"

        )

        torch.save(

            model.state_dict(),

            save_path

        )

        print()

        print(

            "Best epoch:",

            best_epoch

        )

        print(

            "Best validation rollout:",

            f"{best_val:.8f}"

        )

        print(

            "Saved:",

            save_path

        )

    val_results = evaluate_split(

        model=model,

        seeds=VALIDATION_SEEDS,

        body_name=body_name,

        baseline_mass=baseline_mass

    )

    val_summary = summarize_results(

        val_results

    )

    test_results = evaluate_split(

        model=model,

        seeds=TEST_SEEDS,

        body_name=body_name,

        baseline_mass=baseline_mass

    )

    test_summary = summarize_results(

        test_results

    )

    a_errors = []

    for seed in TEST_SEEDS:

        a_errors.append(

            evaluate_model_on_a(

                model=model,

                seed=seed

            )

        )

    a_rollout = float(

        np.mean(

            a_errors

        )

    )

    reduction = (

        zero_summary["rollout"]

        - test_summary["rollout"]

    )

    recovery_pct = (

        reduction

        / max(

            zero_summary["rollout"],

            1e-12

        )

        * 100.0

    )

    result = {

        "budget":

            budget,

        "transitions":

            budget

            * ROLLOUT_STEPS,

        "best_epoch":

            best_epoch,

        "best_val":

            best_val,

        "train_loss":

            train_loss,

        "val_teacher":

            val_summary[

                "teacher"

            ],

        "val_rollout":

            val_summary[

                "rollout"

            ],

        "test_teacher":

            test_summary[

                "teacher"

            ],

        "test_rollout":

            test_summary[

                "rollout"

            ],

        "test_final":

            test_summary[

                "final"

            ],

        "test_max":

            test_summary[

                "max"

            ],

        "a_rollout":

            a_rollout,

        "reduction":

            reduction,

        "recovery_pct":

            recovery_pct,

    }

    all_results.append(

        result

    )

    print()

    print(

        "VALIDATION"

    )

    print(

        "  Teacher:",

        f"{result['val_teacher']:.8f}"

    )

    print(

        "  Rollout:",

        f"{result['val_rollout']:.8f}"

    )

    print()

    print(

        "HELD-OUT TEST B"

    )

    print(

        "  Teacher:",

        f"{result['test_teacher']:.8f}"

    )

    print(

        "  Rollout:",

        f"{result['test_rollout']:.8f}"

    )

    print(

        "  Final:",

        f"{result['test_final']:.8f}"

    )

    print(

        "  Max:",

        f"{result['test_max']:.8f}"

    )

    print()

    print(

        "RECOVERY"

    )

    print(

        "  Reduction:",

        f"{result['reduction']:+.8f}"

    )

    print(

        "  Recovery:",

        f"{result['recovery_pct']:+.3f}%"

    )

    print()

    print(

        "A RETENTION"

    )

    print(

        "  A rollout:",

        f"{result['a_rollout']:.8f}"

    )

print()

print("=" * 90)

print(

    "V8.1 CONTROLLED RECOVERY SUMMARY"

)

print("=" * 90)

print()

header = (

    f"{'Budget':>8} | "

    f"{'Trans':>7} | "

    f"{'BestEp':>7} | "

    f"{'ValRoll':>11} | "

    f"{'TestB':>11} | "

    f"{'Reduction':>11} | "

    f"{'Recovery':>10} | "

    f"{'A Roll':>11}"

)

print(header)

print(

    "-" * len(header)

)

for r in all_results:

    print(

        f"{r['budget']:8d} | "

        f"{r['transitions']:7d} | "

        f"{r['best_epoch']:7d} | "

        f"{r['val_rollout']:11.8f} | "

        f"{r['test_rollout']:11.8f} | "

        f"{r['reduction']:+11.8f} | "

        f"{r['recovery_pct']:+9.3f}% | "

        f"{r['a_rollout']:11.8f}"

    )

print()

print("=" * 90)

print(

    "RECOVERY CURVE CHECK"

)

print("=" * 90)

test_errors = np.asarray(

    [

        r["test_rollout"]

        for r in all_results

    ],

    dtype=np.float64

)

recoveries = np.asarray(

    [

        r["recovery_pct"]

        for r in all_results

    ],

    dtype=np.float64

)

print()

print(

    "Test rollout errors:"

)

for r in all_results:

    print(

        f"  {r['budget']:2d} traj: "

        f"{r['test_rollout']:.8f}"

    )

print()

print(

    "Recovery percentages:"

)

for r in all_results:

    print(

        f"  {r['budget']:2d} traj: "

        f"{r['recovery_pct']:+.3f}%"

    )

np.savez(

    "embodiment_recovery_v8_1_results.npz",

    budgets=np.asarray(

        [

            r["budget"]

            for r in all_results

        ],

        dtype=np.int32

    ),

    transitions=np.asarray(

        [

            r["transitions"]

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

    val_rollout=np.asarray(

        [

            r["val_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    test_teacher=np.asarray(

        [

            r["test_teacher"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    test_rollout=np.asarray(

        [

            r["test_rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    test_final=np.asarray(

        [

            r["test_final"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    test_max=np.asarray(

        [

            r["test_max"]

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

    zero_shot_test=np.float32(

        zero_summary[

            "rollout"

        ]

    ),

    baseline_mass=np.float32(

        baseline_mass

    ),

    target_mass=np.float32(

        target_mass

    ),

    mass_multiplier=np.float32(

        MASS_MULTIPLIER

    ),

)

print()

print("=" * 90)

print(

    "SAVED"

)

print("=" * 90)

print(

    "embodiment_recovery_v8_1_results.npz"

)

for budget in BUDGETS:

    print(

        f"v8_1_headonly_{budget}traj.pt"

    )

print()

print(

    "DONE"

)

print("=" * 90)
