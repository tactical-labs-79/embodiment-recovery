import numpy as np

import robosuite as suite

ENV_NAME = "Lift"

ROBOT = "Panda"

NUM_TRAJECTORIES = 50

TRAJ_LEN = 400

OUTPUT_FILE = "targeted_dataset_v5.npz"

CONTROL_FREQ = 20

RNG_SEED = 5050

TARGET_POINTS = [

    (-2.52, 2.82),

    (-2.48, 2.78),

    (-2.44, 2.74),

    (-2.40, 2.70),

    (-2.38, 2.68),

    (-2.35, 2.65),

    (-2.32, 2.62),

    (-2.30, 2.60),

    (-2.35, 2.58),

    (-2.40, 2.55),

]

MIN_ACTION = 0.05

MAX_ACTION = 0.15

LARGE_ACTION_PROB = 0.15

rng = np.random.default_rng(RNG_SEED)

env = suite.make(

    env_name=ENV_NAME,

    robots=ROBOT,

    has_renderer=False,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    use_object_obs=False,

    control_freq=CONTROL_FREQ,

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

    state = np.concatenate([

        joint_pos,

        joint_vel,

        eef_pos,

    ])

    assert state.shape == (17,)

    return state

states = []

actions = []

next_states = []

print("=" * 75)

print("V5 TARGETED COVERAGE DATASET")

print("=" * 75)

print()

print(f"Trajectories : {NUM_TRAJECTORIES}")

print(f"Steps/traj   : {TRAJ_LEN}")

print(f"Total target : {NUM_TRAJECTORIES * TRAJ_LEN}")

print()

print("Target region:")

print("J4 [-2.40, -2.25]")

print("J6 [ 2.50,  2.70]")

for traj_id in range(NUM_TRAJECTORIES):

    obs = env.reset()

    robot = env.robots[0]

    target_j4, target_j6 = TARGET_POINTS[

        traj_id % len(TARGET_POINTS)

    ]

    target_qpos = np.asarray(

        obs["robot0_joint_pos"],

        dtype=np.float64

    ).copy()

    target_qpos[3] = target_j4

    target_qpos[5] = target_j6

    robot.set_robot_joint_positions(

        target_qpos

    )

    arm_qpos_idx = robot.arm_joint_indexes

    env.sim.data.qvel[

        arm_qpos_idx

    ] = 0.0

    env.sim.forward()

    obs = env._get_observations(force_update=True)

    state = extract_state(obs)

    if traj_id < 10:

        print(

            f"[TRAJ {traj_id:02d}] "

            f"Start J4={state[3]: .5f} "

            f"J6={state[5]: .5f}"

        )

    for step in range(TRAJ_LEN):

        action = np.zeros(

            env.action_dim,

            dtype=np.float32

        )

        if rng.random() < 0.70:

            action_idx = int(

                rng.choice([3, 5])

            )

        else:

            action_idx = int(

                rng.integers(0, 7)

            )

        magnitude = rng.uniform(

            MIN_ACTION,

            MAX_ACTION

        )

        if rng.random() < LARGE_ACTION_PROB:

            magnitude = rng.uniform(

                0.12,

                0.20

            )

        sign = rng.choice(

            [-1.0, 1.0]

        )

        action[action_idx] = (

            sign * magnitude

        )

        next_obs, reward, done, info = env.step(

            action

        )

        next_state = extract_state(

            next_obs

        )

        states.append(

            state.copy()

        )

        actions.append(

            action.copy()

        )

        next_states.append(

            next_state.copy()

        )

        state = next_state

        obs = next_obs

        if done:

            print(

                f"[WARNING] "

                f"Trajectory {traj_id} "

                f"terminated early at step {step + 1}"

            )

            break

env.close()

states = np.asarray(

    states,

    dtype=np.float32

)

actions = np.asarray(

    actions,

    dtype=np.float32

)

next_states = np.asarray(

    next_states,

    dtype=np.float32

)

print()

print("=" * 75)

print("DATASET CHECK")

print("=" * 75)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

assert states.shape[1] == 17

assert actions.shape[1] == 7

assert next_states.shape[1] == 17

j4 = states[:, 3]

j6 = states[:, 5]

broad_mask = (

    (j4 >= -2.55) &

    (j4 <= -2.25) &

    (j6 >= 2.50) &

    (j6 <= 2.85)

)

core_mask = (

    (j4 >= -2.40) &

    (j4 <= -2.25) &

    (j6 >= 2.50) &

    (j6 <= 2.70)

)

print()

print("=" * 75)

print("TARGETED COVERAGE RESULT")

print("=" * 75)

print(

    f"Broad region : "

    f"{broad_mask.sum()} / {len(states)} "

    f"({broad_mask.mean() * 100:.2f}%)"

)

print(

    f"Core region  : "

    f"{core_mask.sum()} / {len(states)} "

    f"({core_mask.mean() * 100:.2f}%)"

)

print()

print("State ranges:")

print(

    f"J4 : "

    f"{j4.min(): .6f} "

    f"→ "

    f"{j4.max(): .6f}"

)

print(

    f"J6 : "

    f"{j6.min(): .6f} "

    f"→ "

    f"{j6.max(): .6f}"

)

np.savez(

    OUTPUT_FILE,

    states=states,

    actions=actions,

    next_states=next_states,

)

print()

print("=" * 75)

print("SAVED")

print("=" * 75)

print(

    f"File : {OUTPUT_FILE}"

)

print("=" * 75)

print("DONE")

print("=" * 75)
