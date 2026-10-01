import numpy as np

import robosuite as suite

NUM_TRAJECTORIES = 50

STEPS_PER_TRAJECTORY = 400

OUTPUT_FILE = "multi_trajectory_v3.npz"

BASE_SEED = 3000

def get_state(obs):

    joint_pos = np.asarray(

        obs["robot0_joint_pos"],

        dtype=np.float32

    )

    eef_pos = np.asarray(

        obs["robot0_eef_pos"],

        dtype=np.float32

    )

    return np.concatenate([

        joint_pos,

        eef_pos

    ]).astype(np.float32)

all_states = []

all_actions = []

all_next_states = []

for traj in range(NUM_TRAJECTORIES):

    seed = BASE_SEED + traj

    print(

        f"\n[TRAJECTORY {traj + 1}/{NUM_TRAJECTORIES}] "

        f"seed={seed}"

    )

    np.random.seed(seed)

    env = suite.make(

        env_name="Lift",

        robots="Panda",

        has_renderer=False,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        control_freq=20,

    )

    obs = env.reset()

    rng = np.random.default_rng(seed)

    for step in range(STEPS_PER_TRAJECTORY):

        state = get_state(obs)

        action = np.zeros(7, dtype=np.float32)

        action_idx = rng.integers(0, 7)

        direction = rng.choice([-1.0, 1.0])

        magnitude = rng.uniform(0.05, 0.20)

        action[action_idx] = (

            direction * magnitude

        )

        next_obs, reward, done, info = env.step(action)

        next_state = get_state(next_obs)

        all_states.append(state)

        all_actions.append(action)

        all_next_states.append(next_state)

        obs = next_obs

        if done:

            print(

                f"  Episode ended at step {step + 1}"

            )

            break

    env.close()

states = np.asarray(

    all_states,

    dtype=np.float32

)

actions = np.asarray(

    all_actions,

    dtype=np.float32

)

next_states = np.asarray(

    all_next_states,

    dtype=np.float32

)

assert states.ndim == 2

assert actions.ndim == 2

assert next_states.ndim == 2

assert states.shape[0] == actions.shape[0]

assert states.shape[0] == next_states.shape[0]

assert states.shape[1] == 10

assert actions.shape[1] == 7

assert next_states.shape[1] == 10

assert not np.isnan(states).any()

assert not np.isnan(actions).any()

assert not np.isnan(next_states).any()

np.savez(

    OUTPUT_FILE,

    states=states,

    actions=actions,

    next_states=next_states

)

print("\n")

print("=" * 60)

print("DATASET V3 COMPLETE")

print("=" * 60)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

print("\nState range:")

print(f"Min : {states.min():.6f}")

print(f"Max : {states.max():.6f}")

print("\nAction range:")

print(f"Min : {actions.min():.6f}")

print(f"Max : {actions.max():.6f}")

print("\nAction dimension usage:")

for i in range(7):

    count = np.count_nonzero(

        actions[:, i]

    )

    percentage = (

        count / len(actions) * 100

    )

    print(

        f"Action {i + 1}: "

        f"{count:5d} samples "

        f"({percentage:6.2f}%)"

    )

print("\nJoint position coverage:")

for i in range(7):

    joint = states[:, i]

    print(

        f"Joint {i + 1}: "

        f"min={joint.min(): .6f}  "

        f"max={joint.max(): .6f}  "

        f"mean={joint.mean(): .6f}  "

        f"std={joint.std(): .6f}"

    )

print("\nEEF position coverage:")

eef_names = ["X", "Y", "Z"]

for i, name in enumerate(eef_names):

    eef = states[:, 7 + i]

    print(

        f"EEF {name}: "

        f"min={eef.min(): .6f}  "

        f"max={eef.max(): .6f}  "

        f"mean={eef.mean(): .6f}  "

        f"std={eef.std(): .6f}"

    )

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 60)
