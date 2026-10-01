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

EPOCHS = 100

BATCH_SIZE = 256

WEIGHT_DECAY = 1e-6

DEVICE = torch.device("cpu")

MODEL_SAVE_PREFIX = "v8_recovery"

logging.getLogger("robosuite").setLevel(

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

    return model

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

            amplitude = amplitudes[

                dim

            ]

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

            f"Expected one link5 body, "

            f"got {matches}"

        )

    return matches[0]

def apply_mass_shift(

    env,

    body_name,

    multiplier,

    baseline_mass

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

    target_mass,

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

            multiplier=MASS_MULTIPLIER,

            baseline_mass=baseline_mass

        )

        if not np.isclose(

            actual_mass,

            target_mass,

            atol=1e-7

        ):

            raise RuntimeError(

                f"Mass mismatch: "

                f"expected {target_mass}, "

                f"got {actual_mass}"

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

        trajectory_states = [

            state.copy()

        ]

        for step in range(

            ROLLOUT_STEPS

        ):

            action = actions[

                step

            ]

            x = np.concatenate(

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

            next_state = (

                extract_state(

                    next_obs

                )

            )

            inputs.append(

                x

            )

            targets.append(

                next_state.copy()

            )

            trajectory_states.append(

                next_state.copy()

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

            "actions": actions,

            "states": np.asarray(

                trajectory_states,

                dtype=np.float32

            ),

        }

    finally:

        env.close()

def build_adaptation_pool(

    body_name,

    target_mass,

    baseline_mass

):

    trajectories = []

    for seed in ADAPT_SEEDS:

        print(

            f"Collecting B adaptation "

            f"trajectory seed={seed}"

        )

        trajectory = (

            collect_b_trajectory(

                seed=seed,

                body_name=body_name,

                target_mass=target_mass,

                baseline_mass=baseline_mass

            )

        )

        trajectories.append(

            trajectory

        )

    return trajectories

def fine_tune_from_v7(

    base_state_dict,

    trajectories

):

    model = DynamicsModel().to(

        DEVICE

    )

    model.load_state_dict(

        copy.deepcopy(

            base_state_dict

        )

    )

    model.train()

    inputs = np.concatenate(

        [

            t["inputs"]

            for t in trajectories

        ],

        axis=0

    )

    targets = np.concatenate(

        [

            t["targets"]

            for t in trajectories

        ],

        axis=0

    )

    x_tensor = torch.from_numpy(

        inputs

    ).to(

        DEVICE

    )

    y_tensor = torch.from_numpy(

        targets

    ).to(

        DEVICE

    )

    dataset = torch.utils.data.TensorDataset(

        x_tensor,

        y_tensor

    )

    loader = torch.utils.data.DataLoader(

        dataset,

        batch_size=min(

            BATCH_SIZE,

            len(dataset)

        ),

        shuffle=True

    )

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    criterion = nn.MSELoss()

    history = []

    for epoch in range(

        EPOCHS

    ):

        total_loss = 0.0

        total_count = 0

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

            batch_size = (

                batch_x.shape[0]

            )

            total_loss += (

                loss.item()

                * batch_size

            )

            total_count += (

                batch_size

            )

        epoch_loss = (

            total_loss

            / max(

                total_count,

                1

            )

        )

        history.append(

            epoch_loss

        )

    model.eval()

    return model, history

def evaluate_model_on_b(

    model,

    seed,

    body_name,

    target_mass,

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

            multiplier=MASS_MULTIPLIER,

            baseline_mass=baseline_mass

        )

        if not np.isclose(

            actual_mass,

            target_mass,

            atol=1e-7

        ):

            raise RuntimeError(

                "B mass was not applied correctly."

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

        actual_states = []

        predicted_states = []

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

            actual_states.append(

                next_state.copy()

            )

            predicted_states.append(

                free_prediction.copy()

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

        actual_states = np.asarray(

            actual_states,

            dtype=np.float32

        )

        predicted_states = np.asarray(

            predicted_states,

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

            "rollout_final":

                float(

                    rollout_errors[-1].mean()

                ),

            "rollout_max":

                float(

                    rollout_errors.max()

                ),

            "teacher_dim":

                teacher_errors.mean(

                    axis=0

                ),

            "rollout_dim":

                rollout_errors.mean(

                    axis=0

                ),

            "actual_states":

                actual_states,

            "predicted_states":

                predicted_states,

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

print("=" * 90)

print(

    "V8 — FEW-SHOT EMBODIMENT RECOVERY"

)

print("=" * 90)

print()

print(

    "Base model:",

    MODEL_FILE

)

print(

    "Robot B mass multiplier:",

    MASS_MULTIPLIER

)

print(

    "Adaptation seeds:",

    ADAPT_SEEDS

)

print(

    "Held-out test seeds:",

    TEST_SEEDS

)

print(

    "Budgets:",

    BUDGETS

)

print(

    "Transitions / trajectory:",

    ROLLOUT_STEPS

)

print(

    "Epochs:",

    EPOCHS

)

print(

    "Learning rate:",

    LEARNING_RATE

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

    "PHASE 1 — COLLECT B ADAPTATION DATA"

)

print("=" * 90)

adaptation_pool = (

    build_adaptation_pool(

        body_name=body_name,

        target_mass=target_mass,

        baseline_mass=baseline_mass

    )

)

print()

print(

    "Collected trajectories:",

    len(adaptation_pool)

)

print(

    "Transitions:",

    len(adaptation_pool)

    * ROLLOUT_STEPS

)

print()

print("=" * 90)

print(

    "PHASE 2 — ZERO-SHOT ROBOT B"

)

print("=" * 90)

zero_shot_test_results = []

for seed in TEST_SEEDS:

    result = evaluate_model_on_b(

        model=base_model,

        seed=seed,

        body_name=body_name,

        target_mass=target_mass,

        baseline_mass=baseline_mass

    )

    zero_shot_test_results.append(

        result

    )

    print()

    print(

        f"Seed {seed}"

    )

    print(

        "  Teacher MAE:",

        f"{result['teacher_mae']:.8f}"

    )

    print(

        "  Rollout MAE:",

        f"{result['rollout_mae']:.8f}"

    )

    print(

        "  Final:",

        f"{result['rollout_final']:.8f}"

    )

    print(

        "  Max:",

        f"{result['rollout_max']:.8f}"

    )

zero_teacher = np.mean(

    [

        r["teacher_mae"]

        for r in zero_shot_test_results

    ]

)

zero_rollout = np.mean(

    [

        r["rollout_mae"]

        for r in zero_shot_test_results

    ]

)

all_results = []

for budget in BUDGETS:

    print()

    print("=" * 90)

    print(

        f"PHASE 3 — ADAPTATION BUDGET = "

        f"{budget} TRAJECTORIES"

    )

    print("=" * 90)

    if budget == 0:

        adapted_model = (

            load_v7_model()

        )

        train_history = [

            0.0

        ]

    else:

        adaptation_trajectories = (

            adaptation_pool[

                :budget

            ]

        )

        print()

        print(

            "Adaptation trajectories:",

            budget

        )

        print(

            "Adaptation transitions:",

            budget

            * ROLLOUT_STEPS

        )

        adapted_model, train_history = (

            fine_tune_from_v7(

                base_state_dict=base_state_dict,

                trajectories=adaptation_trajectories

            )

        )

        print(

            "Initial training loss:",

            f"{train_history[0]:.10f}"

        )

        print(

            "Final training loss:",

            f"{train_history[-1]:.10f}"

        )

        save_path = (

            f"{MODEL_SAVE_PREFIX}_"

            f"{budget}traj.pt"

        )

        torch.save(

            adapted_model.state_dict(),

            save_path

        )

        print(

            "Saved:",

            save_path

        )

    budget_results_b = []

    for seed in TEST_SEEDS:

        result = evaluate_model_on_b(

            model=adapted_model,

            seed=seed,

            body_name=body_name,

            target_mass=target_mass,

            baseline_mass=baseline_mass

        )

        budget_results_b.append(

            result

        )

    budget_results_a = []

    for seed in TEST_SEEDS:

        error_a = evaluate_model_on_a(

            model=adapted_model,

            seed=seed

        )

        budget_results_a.append(

            error_a

        )

    b_teacher = np.mean(

        [

            r["teacher_mae"]

            for r in budget_results_b

        ]

    )

    b_rollout = np.mean(

        [

            r["rollout_mae"]

            for r in budget_results_b

        ]

    )

    b_final = np.mean(

        [

            r["rollout_final"]

            for r in budget_results_b

        ]

    )

    b_max = np.mean(

        [

            r["rollout_max"]

            for r in budget_results_b

        ]

    )

    a_rollout = np.mean(

        budget_results_a

    )

    error_reduction = (

        zero_rollout

        - b_rollout

    )

    if abs(zero_rollout) > 1e-12:

        recovery_pct = (

            error_reduction

            / zero_rollout

            * 100.0

        )

    else:

        recovery_pct = 0.0

    all_results.append({

        "budget":

            budget,

        "transitions":

            budget

            * ROLLOUT_STEPS,

        "teacher":

            b_teacher,

        "rollout":

            b_rollout,

        "final":

            b_final,

        "max":

            b_max,

        "a_rollout":

            a_rollout,

        "error_reduction":

            error_reduction,

        "recovery_pct":

            recovery_pct,

    })

    print()

    print(

        "HELD-OUT B RESULTS"

    )

    print(

        "  Teacher MAE:",

        f"{b_teacher:.8f}"

    )

    print(

        "  Rollout MAE:",

        f"{b_rollout:.8f}"

    )

    print(

        "  Final:",

        f"{b_final:.8f}"

    )

    print(

        "  Max:",

        f"{b_max:.8f}"

    )

    print(

        "  Error reduction:",

        f"{error_reduction:+.8f}"

    )

    print(

        "  Recovery:",

        f"{recovery_pct:+.3f}%"

    )

    print()

    print(

        "HELD-OUT A RETENTION"

    )

    print(

        "  A rollout MAE:",

        f"{a_rollout:.8f}"

    )

print()

print("=" * 90)

print(

    "V8 FEW-SHOT RECOVERY SUMMARY"

)

print("=" * 90)

print()

print(

    f"{'Budget':>8} | "

    f"{'Trans':>7} | "

    f"{'B Teacher':>11} | "

    f"{'B Rollout':>11} | "

    f"{'Reduction':>11} | "

    f"{'Recovery':>10} | "

    f"{'A Rollout':>11}"

)

print(

    "-" * 90

)

for r in all_results:

    print(

        f"{r['budget']:8d} | "

        f"{r['transitions']:7d} | "

        f"{r['teacher']:11.8f} | "

        f"{r['rollout']:11.8f} | "

        f"{r['error_reduction']:+11.8f} | "

        f"{r['recovery_pct']:+9.3f}% | "

        f"{r['a_rollout']:11.8f}"

    )

print()

print("=" * 90)

print(

    "RECOVERY CHECK"

)

print("=" * 90)

zero_result = all_results[0]

print()

print(

    "Zero-shot B rollout:",

    f"{zero_result['rollout']:.8f}"

)

for result in all_results[1:]:

    improvement = (

        zero_result["rollout"]

        - result["rollout"]

    )

    print()

    print(

        f"{result['budget']} trajectories:"

    )

    print(

        "  B rollout:",

        f"{result['rollout']:.8f}"

    )

    print(

        "  Improvement:",

        f"{improvement:+.8f}"

    )

    print(

        "  Recovery:",

        f"{result['recovery_pct']:+.3f}%"

    )

np.savez(

    "embodiment_recovery_v8_results.npz",

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

    b_teacher=np.asarray(

        [

            r["teacher"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_rollout=np.asarray(

        [

            r["rollout"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_final=np.asarray(

        [

            r["final"]

            for r in all_results

        ],

        dtype=np.float32

    ),

    b_max=np.asarray(

        [

            r["max"]

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

    error_reduction=np.asarray(

        [

            r["error_reduction"]

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

    zero_shot_rollout=np.float32(

        zero_rollout

    ),

    zero_shot_teacher=np.float32(

        zero_teacher

    ),

    baseline_mass=np.float32(

        baseline_mass

    ),

    target_mass=np.float32(

        target_mass

    ),

    mass_multiplier=np.float32(

        MASS_MULTIPLIER

    )

)

print()

print("=" * 90)

print(

    "SAVED"

)

print("=" * 90)

print(

    "embodiment_recovery_v8_results.npz"

)

print(

    "v8_recovery_0traj.pt"

)

print(

    "v8_recovery_1traj.pt"

)

print(

    "v8_recovery_5traj.pt"

)

print(

    "v8_recovery_10traj.pt"

)

print(

    "v8_recovery_20traj.pt"

)

print()

print(

    "DONE"

)

print("=" * 90)
