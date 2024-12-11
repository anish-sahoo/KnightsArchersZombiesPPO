# Knights Archers Zombies PPO

**CS 4180 Reinforcement Learning** Final Project - Implementing PPO in a multi-agent environment

Read the paper [here](paper/paper.pdf).

## Environment configuration
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

## Running the code

**NOTE: PettingZoo does not work on Windows, so Linux is recommended.**

First, create a virtualenv and install the required libraries/packages.

```bash
python3 -m venv .venv # this creates a virtual environment called .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

To run the code,
```bash
python3 mappo.py
```

## References

1. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). [Proximal Policy Optimization Algorithms](http://arxiv.org/abs/1707.06347). *CoRR*, abs/1707.06347. Retrieved from [arXiv](http://arxiv.org/abs/1707.06347).

2. **Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A. M., & Wu, Y.** (2021). [The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955). *CoRR*, abs/2103.01955. Retrieved from [arXiv](https://arxiv.org/abs/2103.01955).

3. **Terry, J. K., Black, B., & Hari, A.** (2020). [SuperSuit: Simple Microwrappers for Reinforcement Learning Environments](https://arxiv.org/abs/2008.08932). *arXiv preprint arXiv:2008.08932*. Retrieved from [arXiv](https://arxiv.org/abs/2008.08932).


