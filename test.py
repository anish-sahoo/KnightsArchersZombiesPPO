from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    vector_state=True,
    num_archers=2,
    num_knights=2,
    max_zombies=10,
)
observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(len(observations))
    for key, obs in observations.items():
        print(key, obs.shape)
    break
env.close()
