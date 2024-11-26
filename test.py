from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    vector_state=False,
    num_archers=4,
    num_knights=4,
    max_zombies=50,
)
observations, infos = env.reset()
a = 0
reward = 0
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    sum_rewards = sum(rewards.values())
    reward += sum_rewards
    
    if reward > 1 and sum_rewards > 0:
        print(f"Step {a}: {reward}")
    a += 1
        
env.close()

print(reward)
