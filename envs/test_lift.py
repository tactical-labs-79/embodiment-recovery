import robosuite as suite

from robosuite.controllers import load_composite_controller_config

controller_config = load_composite_controller_config(

    controller="BASIC",

    robot="Panda",

)

env = suite.make(

    env_name="Lift",

    robots="Panda",

    controller_configs=controller_config,

    has_renderer=True,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    control_freq=20,

)

obs = env.reset()

print("Simulator berhasil dimulai!")

print("Action space:", env.action_space)

for step in range(500):

    action = [0.0] * env.action_dim

    obs, reward, done, info = env.step(action)

    env.render()

    if step % 100 == 0:

        print(f"step={step}, reward={reward}, done={done}")

    if done:

        obs = env.reset()

env.close()
