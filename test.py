from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode="human",
    num_archers=4,
    num_knights=0,
    max_zombies=300,
)
observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
