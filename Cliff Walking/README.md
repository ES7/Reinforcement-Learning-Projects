# Reinforcement Learning — From Scratch to Agentic AI

A collection of hands-on RL projects built alongside my [Reinforcement Learning From Scratch](https://medium.com/@sayedebad.777/reinforcement-learning-from-scratch-part-1-understanding-the-agent-environment-loop-e21c580e0af6) series on Medium. Each project is a self-contained implementation that accompanies an article — from tabular Q-Learning all the way to an LLM agent trained with RL to use tools.

---

## Projects

| # | Project | Algorithm | Concepts |
|---|---------|-----------|----------|
| 01 | [Grid World Navigator](./project-01-gridworld/) | Q-Learning | Custom Gym env, reward shaping, epsilon-greedy |
| 02 | [Blackjack Strategy Learner](./project-02-blackjack/) | Monte Carlo | First-visit MC, model-free RL, value function |
| 03 | [CliffWalking: TD vs MC](./project-03-cliffwalking-td-mc/) | TD(0) vs MC | Temporal difference, bias-variance tradeoff |
| 04 | [CliffWalking: SARSA vs Q-Learning](./project-04-cliffwalking-sarsa-qlearning/) | SARSA, Q-Learning | On-policy vs off-policy, path visualization |
| 05 | [LunarLander with DQN](./project-05-lunarlander-dqn/) | DQN | Deep RL, replay buffer, target network |
| 06 | [MountainCar with A2C](./project-06-mountaincar-a2c/) | A2C | Actor-Critic, continuous action space |
| 07 | [BipedalWalker with PPO](./project-07-bipedalwalker-ppo/) | PPO | Clip ratio, surrogate objective |
| 08 | [LLM Tool-Use Agent](./project-08-llm-tool-agent/) | PPO + LLM | Agentic AI, tool selection, custom Gym env |

Each folder is fully self-contained. Install that project's dependencies and run it independently.

---

## Setup

```bash
git clone https://github.com/yourusername/reinforcement-learning-projects.git
cd reinforcement-learning-projects

# Go into any project
cd project-01-gridworld
pip install -r requirements.txt
python train.py
```

Python 3.10+ recommended.

---

## Tech Stack

- [Gymnasium](https://gymnasium.farama.org/) — RL environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO, A2C implementations
- [PyTorch](https://pytorch.org/) — Neural networks for deep RL projects
- [NumPy](https://numpy.org/) & [Matplotlib](https://matplotlib.org/) — Computation and visualization

---
## Author

**Ebad Sayed** — Final year, IIT (ISM) Dhanbad, Co-founder of Voke AI

Connect: [LinkedIn](https://www.linkedin.com/in/ebad-sayed-0861a6227/) · [GitHub](https://github.com/ES7) · [X](https://x.com/EbadOnAI)
