Knights Archers Zombies + Multi Agent PPO


## Goals
- implement the mappo algorithm with recurrent neural networks
- train it for this specific configuration for standardization:
```python
knights_archers_zombies_v10.parallel_env(
    render_mode="human",
    spawn_rate=15,
    vector_state=False,
    num_archers=2,
    num_knights=2,
    max_zombies=30,
)
```
- train and visualize with matplotlib and tensorboard
- run this on some other environments if possible
- run ray mappo for baseline
- write the paper