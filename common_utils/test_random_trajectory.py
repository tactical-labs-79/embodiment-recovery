import numpy as np

import robosuite as suite

STEPS = 400

SEED = 999

rng = np.random.default_rng(SEED)

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=True,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    control_freq=20,

)

obs = env.reset()

observations = []

actions = []

print("================================")

print("RANDOM TEST TRAJECTORY")

print("================================")

print(f"Steps : {STEPS}")

print(f"Seed  : {SEED}")

for step in range(STEPS):

    action = np.zeros(7, dtype=np.float32)

    joint = rng.integers(0, 4)

    direction = rng.choice(

        [-1.0, 1.0]

    )

    magnitude = rng.uniform(

        0.05,

        0.20

    )

    action[joint] = (

        direction * magnitude

    )

    state = np.concatenate([

        obs["robot0_joint_pos"],

        obs["robot0_eef_pos"]

    ]).astype(np.float32)

    observations.append(state)

    actions.append(action.copy())

    obs, reward, done, info = env.step(

        action

    )

    env.render()

    if done:

        break

env.close()

observations = np.array(

    observations,

    dtype=np.float32

)

actions = np.array(

    actions,

    dtype=np.float32

)

np.savez(

    "random_test.npz",

    states=observations,

    actions=actions

)

print("\n================================")

print("RANDOM TEST COMPLETE")

print("================================")

print(

    "States :",

    observations.shape

)

print(

    "Actions:",

    actions.shape

)

print(

    "\nSaved as: random_test.npz"

)
