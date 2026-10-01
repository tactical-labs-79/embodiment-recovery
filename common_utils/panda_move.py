import numpy as np

import robosuite as suite

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

print("Recording started...")

for step in range(800):

    action = np.zeros(7)

    if step < 200:

        action[0] = 0.2

    elif step < 400:

        action[0] = -0.2

    elif step < 600:

        action[1] = 0.2

    else:

        action[1] = -0.2

    observations.append(obs)

    actions.append(action.copy())

    obs, reward, done, info = env.step(action)

    env.render()

    if done:

        obs = env.reset()

env.close()

np.savez(

    "trajectory_varied.npz",

    observations=np.array(observations, dtype=object),

    actions=np.array(actions),

)

print("Recording selesai!")

print(f"Jumlah timestep: {len(actions)}")

print("Data disimpan sebagai: trajectory_varied.npz")
