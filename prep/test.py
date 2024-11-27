from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    spawn_rate=20,
    vector_state=False,
    num_archers=4,
    num_knights=4,
    max_zombies=50,
)

observations, infos = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # rest of the loop

env.close()

