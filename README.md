Knights Archers Zombies + Multi Agent PPO


## Goals
- implement the mappo algorithm with recurrent neural networks
- train it for this specific configuration for standardization:
```python
env = knights_archers_zombies_v10.parallel_env(
    render_mode=None,
    vector_state=False,
    num_archers=4,
    num_knights=4,
    max_zombies=50,
)
```
- train and visualize with matplotlib and tensorboard
- run this on some other environments if possible
- run ray mappo for baseline
- write the paper