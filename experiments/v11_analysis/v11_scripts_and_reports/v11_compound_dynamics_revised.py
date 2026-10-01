import copy

import logging

import os

import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

import mujoco

from robosuite.utils.mjmod import DynamicsModder

ROLLOUT_STEPS = 120

MODEL_FILE = "dynamics_model_v7.pt"

RESULTS_FILE = "embodiment_recovery_v11_compound_dynamics_results.npz"

SHIFT_CONFIGS = [

    (1.0, 1.0),

    (3.0, 1.0),

    (1.0, 3.0),

    (2.0, 2.0),

    (3.0, 3.0),

    (3.0, 5.0),

    (5.0, 3.0),

    (5.0, 5.0),

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

    9601,

    9602,

    9603,

    9604,

    9605,

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

            24,

            128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                128

            ),

            nn.ReLU(),

            nn.Linear(

                128,

                17

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

            0.18,

            0.18,

            0.18,

            0.12,

            0.12,

            0.12,

        ],

        dtype=np.float32

    )

    dimensions = np.arange(6)

    rng.shuffle(

        dimensions

    )

    ROLLOUT_STEPS = 120

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

def get_physical_config():

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

        baseline_inertia = np.asarray(

            env.sim.model.body_inertia[

                body_id

            ],

            dtype=np.float64

        ).copy()

        if baseline_mass <= 0.0:

            raise RuntimeError(

                f"Invalid baseline mass: "

                f"{baseline_mass}"

            )

        if np.any(

            baseline_inertia <= 0.0

        ):

            raise RuntimeError(

                f"Invalid baseline inertia: "

                f"{baseline_inertia}"

            )

        return (

            body_name,

            baseline_mass,

            baseline_inertia

        )

    finally:

        env.close()

def apply_compound_shift(

    env,

    body_name,

    baseline_mass,

    baseline_inertia,

    mass_multiplier,

    inertia_multiplier

):

    target_mass = (

        baseline_mass

        * mass_multiplier

    )

    target_inertia = (

        baseline_inertia

        * inertia_multiplier

    )

    modder = DynamicsModder(

        env.sim,

        randomize_density=False,

        randomize_viscosity=False,

        randomize_position=False,

        randomize_quaternion=False,

        randomize_inertia=False,

        randomize_mass=False,

        randomize_friction=False,

        randomize_solref=False,

        randomize_solimp=False,

        randomize_stiffness=False,

        randomize_frictionloss=False,

        randomize_damping=False,

        randomize_armature=False,

    )

    modder.mod_mass(

        body_name,

        float(target_mass)

    )

    modder.mod_inertia(

        body_name,

        target_inertia.astype(

            np.float64

        )

    )

    modder.update()

    env.sim.forward()

    model = env.sim.model._model

    body_id = mujoco.mj_name2id(

        model,

        mujoco.mjtObj.mjOBJ_BODY,

        body_name

    )

    actual_mass = float(

        env.sim.model.body_mass[

            body_id

        ]

    )

    actual_inertia = np.asarray(

        env.sim.model.body_inertia[

            body_id

        ],

        dtype=np.float64

    ).copy()

    if not np.isclose(

        actual_mass,

        target_mass,

        atol=1e-7,

        rtol=1e-7

    ):

        raise RuntimeError(

            "Mass mismatch:\n"

            f"Actual:   {actual_mass}\n"

            f"Expected: {target_mass}"

        )

    if not np.allclose(

        actual_inertia,

        target_inertia,

        atol=1e-10,

        rtol=1e-7

    ):

        raise RuntimeError(

            "Inertia mismatch:\n"

            f"Actual:   {actual_inertia}\n"

            f"Expected: {target_inertia}"

        )

    return (

        actual_mass,

        actual_inertia

    )

def collect_b_trajectory(

    seed,

    body_name,

    baseline_mass,

    baseline_inertia,

    mass_multiplier,

    inertia_multiplier

):

    env = make_env(

        seed

    )

    try:

        env.reset()

        apply_compound_shift(

            env=env,

            body_name=body_name,

            baseline_mass=baseline_mass,

            baseline_inertia=baseline_inertia,

            mass_multiplier=mass_multiplier,

            inertia_multiplier=inertia_multiplier

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

    baseline_inertia,

    mass_multiplier,

    inertia_multiplier

):

    pool = []

    for seed in seeds:

        print(

            f"Collecting "

            f"M×{mass_multiplier:.1f} "

            f"I×{inertia_multiplier:.1f} "

            f"seed={seed}"

        )

        pool.append(

            collect_b_trajectory(

                seed=seed,

                body_name=body_name,

                baseline_mass=baseline_mass,

                baseline_inertia=baseline_inertia,

                mass_multiplier=mass_multiplier,

                inertia_multiplier=inertia_multiplier

            )

        )

    return pool

def evaluate_model_on_b(

    model,

    seed,

    body_name,

    baseline_mass,

    baseline_inertia,

    mass_multiplier,

    inertia_multiplier,

    ROLLOUT_STEPS=120

):

    env = make_env(

        seed

    )

    try:

        env.reset()

        apply_compound_shift(

            env=env,

            body_name=body_name,

            baseline_mass=baseline_mass,

            baseline_inertia=baseline_inertia,

            mass_multiplier=mass_multiplier,

            inertia_multiplier=inertia_multiplier

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

    run_seed,

    progress_path=None

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

    best_val_teacher = float("inf")

    best_epoch = 0

    patience_counter = 0

    start_epoch = 1

    if (

        progress_path is not None

        and os.path.exists(progress_path)

    ):

        progress = torch.load(

            progress_path,

            map_location=DEVICE

        )

        model.load_state_dict(

            progress["model_state"]

        )

        optimizer.load_state_dict(

            progress["optimizer_state"]

        )

        generator.set_state(

            progress["generator_state"]

        )

        best_state = progress["best_state"]

        best_val = progress["best_val"]

        best_val_teacher = progress["best_val_teacher"]

        best_epoch = progress["best_epoch"]

        patience_counter = progress["patience_counter"]

        start_epoch = progress["epoch"] + 1

        print(

            f"Resuming mid-training from epoch "

            f"{start_epoch} (progress checkpoint found)."

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

    for epoch in range(

        start_epoch,

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

                loss.item()

                * n

            )

            count += n

        train_loss = (

            running_loss

            / max(count, 1)

        )

        val_teacher, val_rollout = validation_fn(

            model

        )

        improved = (

            val_rollout

            < best_val - MIN_DELTA

        )

        if improved:

            best_val = val_rollout

            best_val_teacher = val_teacher

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

                f"val_teacher={val_teacher:.8f} | "

                f"val_rollout={val_rollout:.8f} | "

                f"best={best_val:.8f}"

            )

        if progress_path is not None:

            atomic_torch_save(

                {

                    "model_state": model.state_dict(),

                    "optimizer_state": optimizer.state_dict(),

                    "generator_state": generator.get_state(),

                    "best_state": best_state,

                    "best_val": best_val,

                    "best_val_teacher": best_val_teacher,

                    "best_epoch": best_epoch,

                    "patience_counter": patience_counter,

                    "epoch": epoch,

                },

                progress_path

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

    if (

        progress_path is not None

        and os.path.exists(progress_path)

    ):

        os.remove(

            progress_path

        )

    return (

        model,

        best_epoch,

        best_val,

        best_val_teacher

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

import hashlib

import inspect

REQUIRED_ID_FIELDS = [

    "mass_multiplier",

    "inertia_multiplier",

    "budget",

    "repeat",

]

METRIC_FIELDS = [

    "best_epoch",

    "best_val",

    "val_rollout",

    "zero_rollout",

    "confirmation_rollout",

    "confirmation_std",

    "reduction",

    "recovery_pct",

]

def experiment_key(

    mass_multiplier,

    inertia_multiplier,

    budget,

    repeat

):

    return (

        round(float(mass_multiplier), 4),

        round(float(inertia_multiplier), 4),

        int(budget),

        int(repeat),

    )

def checkpoint_path_for(

    mass_multiplier,

    inertia_multiplier,

    budget,

    repeat

):

    mass_tag = f"{mass_multiplier:.0f}"

    inertia_tag = f"{inertia_multiplier:.0f}"

    return (

        "v11_compound_"

        f"Mx{mass_tag}_"

        f"Ix{inertia_tag}_"

        f"{budget}traj_"

        f"subset{repeat}.pt"

    )

def progress_checkpoint_path_for(

    mass_multiplier,

    inertia_multiplier,

    budget,

    repeat

):

    return checkpoint_path_for(

        mass_multiplier,

        inertia_multiplier,

        budget,

        repeat

    ).replace(

        ".pt",

        "_inprogress.pt"

    )

def compute_experiment_version():

    relevant_config = {

        "MODEL_FILE": MODEL_FILE,

        "INPUT_DIM": INPUT_DIM,

        "OUTPUT_DIM": OUTPUT_DIM,

        "HIDDEN": HIDDEN,

        "POS_AMPLITUDE": POS_AMPLITUDE,

        "ROT_AMPLITUDE": ROT_AMPLITUDE,

        "ROLLOUT_STEPS": ROLLOUT_STEPS,

        "LEARNING_RATE": LEARNING_RATE,

        "MAX_EPOCHS": MAX_EPOCHS,

        "PATIENCE": PATIENCE,

        "BATCH_SIZE": BATCH_SIZE,

        "WEIGHT_DECAY": WEIGHT_DECAY,

        "MIN_DELTA": MIN_DELTA,

        "ADAPT_SEEDS": ADAPT_SEEDS,

        "VALIDATION_SEEDS": VALIDATION_SEEDS,

        "CONFIRMATION_SEEDS": CONFIRMATION_SEEDS,

        "SUBSET_REPEATS": SUBSET_REPEATS,

    }

    fingerprint = repr(sorted(relevant_config.items()))

    fingerprint += inspect.getsource(DynamicsModel)

    fingerprint += inspect.getsource(train_head_only)

    return hashlib.sha256(

        fingerprint.encode("utf-8")

    ).hexdigest()[:16]

EXPERIMENT_VERSION = compute_experiment_version()

def atomic_savez(path, **arrays):

    tmp_path = path + ".tmp.npz"

    np.savez(

        tmp_path,

        **arrays

    )

    os.replace(

        tmp_path,

        path

    )

def atomic_torch_save(obj, path):

    tmp_path = path + ".tmp"

    torch.save(

        obj,

        tmp_path

    )

    os.replace(

        tmp_path,

        path

    )

def save_results(results):

    atomic_savez(

        RESULTS_FILE,

        mass_multiplier=np.asarray(

            [r["mass_multiplier"] for r in results],

            dtype=np.float32

        ),

        inertia_multiplier=np.asarray(

            [r["inertia_multiplier"] for r in results],

            dtype=np.float32

        ),

        budget=np.asarray(

            [r["budget"] for r in results],

            dtype=np.int32

        ),

        repeat=np.asarray(

            [r["repeat"] for r in results],

            dtype=np.int32

        ),

        best_epoch=np.asarray(

            [r["best_epoch"] for r in results],

            dtype=np.int32

        ),

        best_val=np.asarray(

            [r["best_val"] for r in results],

            dtype=np.float32

        ),

        val_rollout=np.asarray(

            [r["val_rollout"] for r in results],

            dtype=np.float32

        ),

        zero_rollout=np.asarray(

            [r["zero_rollout"] for r in results],

            dtype=np.float32

        ),

        confirmation_rollout=np.asarray(

            [r["confirmation_rollout"] for r in results],

            dtype=np.float32

        ),

        confirmation_std=np.asarray(

            [r["confirmation_std"] for r in results],

            dtype=np.float32

        ),

        reduction=np.asarray(

            [r["reduction"] for r in results],

            dtype=np.float32

        ),

        recovery_pct=np.asarray(

            [r["recovery_pct"] for r in results],

            dtype=np.float32

        ),

        experiment_version=np.asarray(

            [r["experiment_version"] for r in results],

            dtype="<U16"

        ),

        baseline_mass=np.float32(baseline_mass),

        baseline_inertia=baseline_inertia.astype(np.float32),

    )

def load_existing_results():

    if not os.path.exists(RESULTS_FILE):

        return []

    data = np.load(RESULTS_FILE)

    missing_id_fields = [

        f for f in REQUIRED_ID_FIELDS if f not in data.files

    ]

    if missing_id_fields:

        print()

        print("=" * 105)

        print(

            f"WARNING: {RESULTS_FILE} is missing required field(s) "

            f"{missing_id_fields} — it doesn't match this script's "

            f"result schema. Treating it as incompatible and "

            f"starting fresh (rename/move the old file first if you "

            f"want to keep it for reference)."

        )

        print("=" * 105)

        return []

    n = len(data["mass_multiplier"])

    loaded = []

    for i in range(n):

        entry = {

            "mass_multiplier":

                float(data["mass_multiplier"][i]),

            "inertia_multiplier":

                float(data["inertia_multiplier"][i]),

            "budget":

                int(data["budget"][i]),

            "repeat":

                int(data["repeat"][i]),

        }

        for field in METRIC_FIELDS:

            if field in data.files:

                value = data[field][i]

                entry[field] = (

                    int(value)

                    if field == "best_epoch"

                    else float(value)

                )

            else:

                entry[field] = float("nan")

        if "experiment_version" in data.files:

            entry["experiment_version"] = str(

                data["experiment_version"][i]

            )

        else:

            entry["experiment_version"] = None

        entry["zero_error"] = entry["zero_rollout"]

        entry["adapted_error"] = entry["confirmation_rollout"]

        loaded.append(entry)

    return loaded

def is_result_valid(r):

    if r["experiment_version"] != EXPERIMENT_VERSION:

        return False

    metric_values = [

        r["best_val"],

        r["val_rollout"],

        r["zero_rollout"],

        r["confirmation_rollout"],

        r["confirmation_std"],

        r["reduction"],

        r["recovery_pct"],

    ]

    if any(

        not np.isfinite(v)

        for v in metric_values

    ):

        return False

    checkpoint_name = checkpoint_path_for(

        r["mass_multiplier"],

        r["inertia_multiplier"],

        r["budget"],

        r["repeat"],

    )

    if not os.path.exists(checkpoint_name):

        return False

    return True

def build_completed_keys(results):

    completed = set()

    invalid_count = 0

    version_mismatch_count = 0

    for r in results:

        key = experiment_key(

            r["mass_multiplier"],

            r["inertia_multiplier"],

            r["budget"],

            r["repeat"],

        )

        if is_result_valid(r):

            completed.add(key)

        else:

            invalid_count += 1

            if r["experiment_version"] != EXPERIMENT_VERSION:

                version_mismatch_count += 1

    if invalid_count:

        print()

        print("=" * 105)

        print(

            f"WARNING: {invalid_count} entr(y/ies) in {RESULTS_FILE} "

            f"will be re-run instead of being trusted as done "

            f"({version_mismatch_count} from a different "

            f"model/training version, "

            f"{invalid_count - version_mismatch_count} with "

            f"non-finite metrics or a missing checkpoint file)."

        )

        print("=" * 105)

    return completed

print("=" * 105)

print(

    "V11 — COMPOUND DYNAMICS RECOVERY"

)

print("=" * 105)

print()

print(

    "Shift configs:",

    SHIFT_CONFIGS

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

    "Validation seeds:",

    VALIDATION_SEEDS

)

print(

    "Fresh confirmation:",

    CONFIRMATION_SEEDS

)

(

    body_name,

    baseline_mass,

    baseline_inertia

) = get_physical_config()

print()

print("=" * 105)

print(

    "PHYSICAL BASELINE"

)

print("=" * 105)

print(

    "Body:",

    body_name

)

print(

    "Baseline mass:",

    f"{baseline_mass:.8f}"

)

print(

    "Baseline inertia:",

    np.array2string(

        baseline_inertia,

        precision=10

    )

)

base_model = load_v7_model()

base_state_dict = copy.deepcopy(

    base_model.state_dict()

)

subset_specs = build_subset_specs()

print()

print("=" * 105)

print(

    "FIXED ADAPTATION SUBSETS"

)

print("=" * 105)

for spec in subset_specs:

    print(

        f"Budget {spec['budget']:2d} | "

        f"Subset {spec['repeat']} | "

        f"{spec['seeds']}"

    )

raw_loaded_results = load_existing_results()

completed_keys = build_completed_keys(

    raw_loaded_results

)

all_results = [

    r

    for r in raw_loaded_results

    if experiment_key(

        r["mass_multiplier"],

        r["inertia_multiplier"],

        r["budget"],

        r["repeat"]

    ) in completed_keys

]

if all_results:

    print()

    print("=" * 105)

    print(

        f"RESUME: found {len(all_results)} previously completed "

        f"experiment(s) in {RESULTS_FILE}"

    )

    print("=" * 105)

for (

    mass_multiplier,

    inertia_multiplier

) in SHIFT_CONFIGS:

    print()

    print("#" * 105)

    print(

        f"COMPOUND SHIFT "

        f"M×{mass_multiplier:.1f} "

        f"I×{inertia_multiplier:.1f}"

    )

    print("#" * 105)

    pending_specs = [

        spec

        for spec in subset_specs

        if experiment_key(

            mass_multiplier,

            inertia_multiplier,

            spec["budget"],

            spec["repeat"]

        ) not in completed_keys

    ]

    if not pending_specs:

        print()

        print(

            f"SKIP — all subsets for M×{mass_multiplier:.1f} "

            f"I×{inertia_multiplier:.1f} already completed, "

            f"nothing to adapt or evaluate here."

        )

        continue

    skipped_specs = [

        spec

        for spec in subset_specs

        if spec not in pending_specs

    ]

    for spec in skipped_specs:

        print(

            f"SKIP (already completed): "

            f"M×{mass_multiplier:.1f} "

            f"I×{inertia_multiplier:.1f} | "

            f"Budget {spec['budget']} | "

            f"Subset {spec['repeat']}"

        )

    target_mass = (

        baseline_mass

        * mass_multiplier

    )

    target_inertia = (

        baseline_inertia

        * inertia_multiplier

    )

    print()

    print(

        "Target mass:",

        f"{target_mass:.8f}"

    )

    print(

        "Target inertia:",

        np.array2string(

            target_inertia,

            precision=10

        )

    )

    adaptation_pool = build_b_pool(

        seeds=ADAPT_SEEDS,

        body_name=body_name,

        baseline_mass=baseline_mass,

        baseline_inertia=baseline_inertia,

        mass_multiplier=mass_multiplier,

        inertia_multiplier=inertia_multiplier

    )

    seed_to_trajectory = {

        t["seed"]: t

        for t in adaptation_pool

    }

    print()

    print("=" * 105)

    print(

        f"ZERO-SHOT CONFIRMATION "

        f"M×{mass_multiplier:.1f} "

        f"I×{inertia_multiplier:.1f}"

    )

    print("=" * 105)

    zero_results = []

    for seed in CONFIRMATION_SEEDS:

        result = evaluate_model_on_b(

            model=base_model,

            seed=seed,

            body_name=body_name,

            baseline_mass=baseline_mass,

            baseline_inertia=baseline_inertia,

            mass_multiplier=mass_multiplier,

            inertia_multiplier=inertia_multiplier

        )

        zero_results.append(

            result

        )

        print(

            f"Seed {seed}: "

            f"teacher={result['teacher']:.8f} "

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

    for spec in pending_specs:

        budget = spec["budget"]

        repeat = spec["repeat"]

        selected_seeds = spec["seeds"]

        print()

        print("=" * 105)

        print(

            f"M×{mass_multiplier:.1f} "

            f"I×{inertia_multiplier:.1f} | "

            f"Budget {budget} | "

            f"Subset {repeat}"

        )

        print("=" * 105)

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

            teacher_values = []

            rollout_values = []

            for seed in VALIDATION_SEEDS:

                result = evaluate_model_on_b(

                    model=candidate,

                    seed=seed,

                    body_name=body_name,

                    baseline_mass=baseline_mass,

                    baseline_inertia=baseline_inertia,

                    mass_multiplier=mass_multiplier,

                    inertia_multiplier=inertia_multiplier

                )

                teacher_values.append(

                    result["teacher"]

                )

                rollout_values.append(

                    result["rollout"]

                )

            return (

                float(np.mean(teacher_values)),

                float(np.mean(rollout_values))

            )

        run_seed = (

            1100000

            + int(mass_multiplier * 10000)

            + int(inertia_multiplier * 1000)

            + budget * 100

            + repeat

        )

        progress_path = progress_checkpoint_path_for(

            mass_multiplier,

            inertia_multiplier,

            budget,

            repeat

        )

        (

            model,

            best_epoch,

            best_val,

            best_val_teacher

        ) = train_head_only(

            base_state_dict=

                base_state_dict,

            train_trajectories=

                train_trajectories,

            validation_fn=

                validation_fn,

            run_seed=

                run_seed,

            progress_path=

                progress_path

        )

        val_results = []

        for seed in VALIDATION_SEEDS:

            val_results.append(

                evaluate_model_on_b(

                    model=model,

                    seed=seed,

                    body_name=body_name,

                    baseline_mass=baseline_mass,

                    baseline_inertia=baseline_inertia,

                    mass_multiplier=mass_multiplier,

                    inertia_multiplier=inertia_multiplier

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

                    model=model,

                    seed=seed,

                    body_name=body_name,

                    baseline_mass=baseline_mass,

                    baseline_inertia=baseline_inertia,

                    mass_multiplier=mass_multiplier,

                    inertia_multiplier=inertia_multiplier

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

        confirmation_values = np.asarray(

            [

                r["rollout"]

                for r in confirmation_results

            ],

            dtype=np.float64

        )

        confirmation_std = float(

            confirmation_values.std(

                ddof=1

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

        result = {

            "mass_multiplier":

                mass_multiplier,

            "inertia_multiplier":

                inertia_multiplier,

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

                confirmation_std,

            "reduction":

                reduction,

            "recovery_pct":

                recovery_pct,

            "zero_error":

                zero_rollout,

            "adapted_error":

                confirmation_rollout,

            "experiment_version":

                EXPERIMENT_VERSION,

        }

        all_results.append(

            result

        )

        completed_keys.add(

            experiment_key(

                mass_multiplier,

                inertia_multiplier,

                budget,

                repeat

            )

        )

        checkpoint_name = checkpoint_path_for(

            mass_multiplier,

            inertia_multiplier,

            budget,

            repeat

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

            f"{confirmation_std:.8f}"

        )

        print(

            "  Saved:",

            checkpoint_name

        )

        save_results(

            all_results

        )

print()

print("=" * 105)

print(

    "V11 COMPOUND DYNAMICS SUMMARY"

)

print("=" * 105)

for (

    mass_multiplier,

    inertia_multiplier

) in SHIFT_CONFIGS:

    print()

    print(

        f"M×{mass_multiplier:.1f} "

        f"I×{inertia_multiplier:.1f}"

    )

    for budget in BUDGETS:

        subset_results = [

            r

            for r in all_results

            if (

                r["mass_multiplier"]

                == mass_multiplier

                and

                r["inertia_multiplier"]

                == inertia_multiplier

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

            f"mean="

            f"{recoveries.mean():+.3f}% | "

            f"std="

            f"{recoveries.std(ddof=1):.3f}% | "

            f"min="

            f"{recoveries.min():+.3f}% | "

            f"max="

            f"{recoveries.max():+.3f}% | "

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

print("=" * 105)

print(

    "CROSS-COMPOUND RECOVERY TABLE"

)

print("=" * 105)

header = (

    f"{'Mass':>7} | "

    f"{'Inertia':>8} | "

    f"{'Budget':>7} | "

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

for (

    mass_multiplier,

    inertia_multiplier

) in SHIFT_CONFIGS:

    for budget in BUDGETS:

        subset_results = [

            r

            for r in all_results

            if (

                r["mass_multiplier"]

                == mass_multiplier

                and

                r["inertia_multiplier"]

                == inertia_multiplier

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

            f"{mass_multiplier:7.1f} | "

            f"{inertia_multiplier:8.1f} | "

            f"{budget:7d} | "

            f"{recoveries.mean():+9.3f}% | "

            f"{recoveries.std(ddof=1):8.3f} | "

            f"{recoveries.min():+9.3f}% | "

            f"{recoveries.max():+9.3f}% | "

            f"{np.sum(recoveries > 0):4d}/"

            f"{len(recoveries):<4d}"

        )

print()

print("=" * 105)

print(

    "RAW ERROR TABLE — USE THIS FOR COMPOUND INTERACTION ANALYSIS"

)

print("=" * 105)

raw_header = (

    f"{'Mass':>7} | "

    f"{'Inertia':>8} | "

    f"{'Budget':>7} | "

    f"{'ZeroErr':>12} | "

    f"{'AdaptErr':>12} | "

    f"{'Reduction':>12}"

)

print(raw_header)

print("-" * len(raw_header))

for (

    mass_multiplier,

    inertia_multiplier

) in SHIFT_CONFIGS:

    for budget in BUDGETS:

        subset_results = [

            r

            for r in all_results

            if (

                r["mass_multiplier"] == mass_multiplier

                and

                r["inertia_multiplier"] == inertia_multiplier

                and

                r["budget"] == budget

            )

        ]

        zero_values = np.asarray(

            [

                r["zero_error"]

                for r in subset_results

            ],

            dtype=np.float64

        )

        adapted_values = np.asarray(

            [

                r["adapted_error"]

                for r in subset_results

            ],

            dtype=np.float64

        )

        reduction_values = zero_values - adapted_values

        print(

            f"{mass_multiplier:7.1f} | "

            f"{inertia_multiplier:8.1f} | "

            f"{budget:7d} | "

            f"{zero_values.mean():12.8f} | "

            f"{adapted_values.mean():12.8f} | "

            f"{reduction_values.mean():+12.8f}"

        )

save_results(

    all_results

)

print()

print("=" * 105)

print(

    "SAVED"

)

print("=" * 105)

print(

    RESULTS_FILE

)

print()

print(

    "DONE"

)
