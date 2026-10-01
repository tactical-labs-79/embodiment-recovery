import numpy as np

import robosuite as suite

NUM_TRAJECTORIES = 20

STEPS_PER_TRAJECTORY = 400

OUTPUT_FILE = "multi_trajectory.npz"

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=True,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    control_freq=20,

)

all_states = []

all_actions = []

all_next_states = []

print("================================")

print("MULTI TRAJECTORY DATA COLLECTION")

print("================================")

print(f"Trajectories : {NUM_TRAJECTORIES}")

print(f"Steps each   : {STEPS_PER_TRAJECTORY}")

for trajectory in range(NUM_TRAJECTORIES):

    obs = env.reset()

    print(

        f"\nTrajectory "

        f"{trajectory + 1}/{NUM_TRAJECTORIES}"

    )

    states = []

    actions = []

    for step in range(STEPS_PER_TRAJECTORY):

        action = np.zeros(7)

        joint = np.random.randint(0, 4)

        direction = np.random.choice(

            [-1.0, 1.0]

        )

        magnitude = np.random.uniform(

            0.05,

            0.20

        )

        action[joint] = (

            direction * magnitude

        )

        state = np.concatenate([

            obs["robot0_joint_pos"],

            obs["robot0_eef_pos"]

        ])

        states.append(state)

        actions.append(action.copy())

        obs, reward, done, info = env.step(

            action

        )

        env.render()

        if done:

            break

    states = np.array(

        states,

        dtype=np.float32

    )

    actions = np.array(

        actions,

        dtype=np.float32

    )

    if len(states) > 1:

        all_states.append(

            states[:-1]

        )

        all_actions.append(

            actions[:-1]

        )

        all_next_states.append(

            states[1:]

        )

env.close()

states = np.concatenate(

    all_states,

    axis=0

)

actions = np.concatenate(

    all_actions,

    axis=0

)

next_states = np.concatenate(

    all_next_states,

    axis=0

)

np.savez(

    OUTPUT_FILE,

    states=states,

    actions=actions,

    next_states=next_states

)

print("\n================================")

print("DATASET COLLECTION COMPLETE")

print("================================")

print(

    "States      :",

    states.shape

)

print(

    "Actions     :",

    actions.shape

)

print(

    "Next states :",

    next_states.shape

)

print(

    "\nSaved as:",

    OUTPUT_FILE

)
