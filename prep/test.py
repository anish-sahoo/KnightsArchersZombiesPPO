from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    spawn_rate=20,
    vector_state=False,
    num_archers=4,
    num_knights=4,
    max_zombies=50,
)

# observations, infos = env.reset()
# while env.agents:
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     observations, rewards, terminations, truncations, infos = env.step(actions)
#     # rest of the loop

# env.close()

env = knights_archers_zombies_v10.parallel_env(
    render_mode="human",
    spawn_rate=15,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=10,
)

import supersuit as ss

# Apply SuperSuit wrappers to make observations smaller and greyscale
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.color_reduction_v0(env, mode='full')
env = ss.frame_stack_v2(env, stack_size=4)

observations, infos = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    import matplotlib.pyplot as plt
    i=0
    for agent, obs in observations.items():
        print(agent, obs.shape)
        plt.imshow(obs)
        plt.title(f"Observation for {agent}")
        plt.savefig(f'observation{i}.png')
        plt.axis('off')
        i+=1
    break

env.close()