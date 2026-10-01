import robosuite as suite

import numpy as np

import os

NUM_TRAJECTORIES = 50

STEPS_PER_TRAJECTORY = 400

BASE_SEED = 3000

OUTPUT_FILE = "multi_trajectory_v4.npz"

def get_state(obs):

    required_keys = [

        "robot0_joint_pos",

        "robot0_joint_vel",

        "robot0_eef_pos",

    ]

    for key in required_keys:

        if key not in obs:

            raise KeyError(

                f"Observation key '{key}' tidak ditemukan. "

                f"Keys tersedia: {list(obs.keys())}"

            )

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

        eef_pos

    ])

    if state.shape != (17,):

        raise ValueError(

            f"Dimensi state salah: {state.shape}, "

            f"seharusnya (17,)"

        )

    return state.astype(np.float32)

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=False,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    control_freq=20,

)

all_states = []

all_actions = []

all_next_states = []

print("=" * 60)

print("COLLECTING DATASET V4")

print("State: 7 joint pos + 7 joint vel + 3 EEF pos = 17")

print("Action: 7 controller commands")

print("=" * 60)

for traj in range(NUM_TRAJECTORIES):

    seed = BASE_SEED + traj

    np.random.seed(seed)

    obs = env.reset()

    state = get_state(obs)

    for step in range(STEPS_PER_TRAJECTORY):

        action = np.zeros(7, dtype=np.float32)

        action_index = np.random.randint(0, 7)

        action_sign = np.random.choice([-1.0, 1.0])

        action_magnitude = np.random.uniform(0.05, 0.20)

        action[action_index] = (

            action_sign * action_magnitude

        )

        next_obs, reward, done, info = env.step(action)

        next_state = get_state(next_obs)

        all_states.append(state)

        all_actions.append(action)

        all_next_states.append(next_state)

        state = next_state

    print(

        f"Trajectory {traj + 1:02d}/{NUM_TRAJECTORIES} "

        f"selesai | seed={seed}"

    )

env.close()

states = np.asarray(all_states, dtype=np.float32)

actions = np.asarray(all_actions, dtype=np.float32)

next_states = np.asarray(all_next_states, dtype=np.float32)

assert states.ndim == 2

assert actions.ndim == 2

assert next_states.ndim == 2

assert states.shape[1] == 17

assert actions.shape[1] == 7

assert next_states.shape[1] == 17

assert states.shape[0] == actions.shape[0]

assert states.shape[0] == next_states.shape[0]

assert not np.isnan(states).any()

assert not np.isnan(actions).any()

assert not np.isnan(next_states).any()

np.savez(

    OUTPUT_FILE,

    states=states,

    actions=actions,

    next_states=next_states

)

print("\n" + "=" * 60)

print("DATASET V4 SELESAI")

print("=" * 60)

print(f"File       : {OUTPUT_FILE}")

print(f"States     : {states.shape}")

print(f"Actions    : {actions.shape}")

print(f"Next states: {next_states.shape}")

print(f"State min  : {states.min():.6f}")

print(f"State max  : {states.max():.6f}")

print(f"Action min : {actions.min():.6f}")

print(f"Action max : {actions.max():.6f}")

print("=" * 60)
