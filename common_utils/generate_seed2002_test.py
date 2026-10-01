import numpy as np

import robosuite as suite

SEED = 2002

STEPS = 400

OUTPUT_FILE = "random_test_2002.npz"

def get_state(obs):

    joint_pos = np.asarray(

        obs["robot0_joint_pos"],

        dtype=np.float32

    )

    eef_pos = np.asarray(

        obs["robot0_eef_pos"],

        dtype=np.float32

    )

    return np.concatenate(

        [joint_pos, eef_pos]

    ).astype(np.float32)

np.random.seed(SEED)

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=False,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    control_freq=20,

)

obs = env.reset()

rng = np.random.default_rng(SEED)

states = []

actions = []

next_states = []

for step in range(STEPS):

    state = get_state(obs)

    action = np.zeros(7, dtype=np.float32)

    action_idx = rng.integers(0, 7)

    direction = rng.choice([-1.0, 1.0])

    magnitude = rng.uniform(0.05, 0.20)

    action[action_idx] = direction * magnitude

    next_obs, reward, done, info = env.step(action)

    next_state = get_state(next_obs)

    states.append(state)

    actions.append(action)

    next_states.append(next_state)

    obs = next_obs

    if done:

        print(f"Episode ended at step {step + 1}")

        break

env.close()

states = np.asarray(states, dtype=np.float32)

actions = np.asarray(actions, dtype=np.float32)

next_states = np.asarray(next_states, dtype=np.float32)

np.savez(

    OUTPUT_FILE,

    states=states,

    actions=actions,

    next_states=next_states

)

print("=" * 60)

print("SEED 2002 TEST DATA GENERATED")

print("=" * 60)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

print(f"Saved       : {OUTPUT_FILE}")

print("=" * 60)
