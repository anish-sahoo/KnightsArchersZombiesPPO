from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.parallel_env(
    render_mode="human",
    vector_state=False,
    num_archers=4,
    num_knights=4,
    max_zombies=10,
    
)
observations, infos = env.reset()
a = 0
reward = 0
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for agent, done in terminations.items():
        if done:
            print(f"Agent {agent} died at step {a}")
    sum_rewards = sum(rewards.values())
    reward += sum_rewards
    
    if sum_rewards > 0:
        print(f"Step {a}: {reward}")
    a += 1
    
print(f"Final reward: {reward}")
print(f"Final step: {a}")
env.close()

print(reward)
