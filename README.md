# Reinforcement Learning — From Scratch to Agentic AI

A collection of hands-on RL projects built alongside my [Reinforcement Learning From Scratch](https://medium.com/@sayedebad.777/reinforcement-learning-from-scratch-part-1-understanding-the-agent-environment-loop-e21c580e0af6) series on Medium. Each project is a self-contained implementation that accompanies an article — from tabular Q-Learning all the way to a miniature RLHF pipeline.

---

## Projects

| # | Project | Algorithm | Concepts | Article |
|---|---------|-----------|----------|---------|
| 01 | [Grid World Navigator](./project-01-gridworld/) | Q-Learning | Custom Gym env, reward shaping, epsilon-greedy | [Read →](https://medium.com/@sayedebad.777/building-a-grid-world-navigator-with-q-learning-and-gymnasium-2b5aa06bb597) |
| 02 | [Blackjack Strategy Learner](./project-02-blackjack/) | Monte Carlo | First-visit MC, model-free RL, value function | [Read →](https://medium.com/@sayedebad.777/building-a-blackjack-strategy-learner-with-monte-carlo-rl-1e3a66d153c5) |
| 03 | [CliffWalking: TD vs MC](./project-03-cliffwalking-td-mc/) | TD(0) vs MC | Temporal difference, bias-variance tradeoff | [Read →](https://medium.com/@sayedebad.777/td-vs-monte-carlo-on-cliffwalking-a-head-to-head-comparison-b2d83aaee71f) |
| 04 | [CliffWalking: SARSA vs Q-Learning](./project-04-cliffwalking-sarsa-qlearning/) | SARSA, Q-Learning | On-policy vs off-policy, path visualization | [Read →](https://medium.com/@sayedebad.777/sarsa-vs-q-learning-on-cliffwalking-on-policy-vs-off-policy-cb9917da0262) |
| 05 | [LunarLander with DQN](./project-05-lunarlander-dqn/) | DQN | Deep RL, replay buffer, target network | [Read →](https://medium.com/@sayedebad.777/lunarlander-with-deep-q-networks-from-scratch-no-libraries-be6632a2ac2d) |
| 06 | [MountainCar with A2C](./project-06-mountaincar-a2c/) | A2C | Actor-Critic, continuous action space | [Read →](http://medium.com/@sayedebad.777/mountaincar-with-actor-critic-a2c-continuous-actions-from-scratch-60679c68d051) |
| 07 | [BipedalWalker with PPO](./project-07-bipedalwalker-ppo/) | PPO | Clip ratio, surrogate objective, GAE | [Read →](https://medium.com/@sayedebad.777/bipedalwalker-with-ppo-the-clip-ratio-explained-61a644e713cd) |
| 08 | [LLM Tool-Use Agent](./project-08-llm-tool-agent/) | PPO + LLM | Agentic AI, tool selection, custom Gym env | [Read →](https://medium.com/@sayedebad.777/building-an-rl-trained-tool-use-agent-from-scratch-542d325ad205) |
| 09 | [Miniature RLHF Pipeline](./project-09-rlhf/) | Reward Model + PPO | Human preferences, Bradley-Terry loss, KL penalty | [Read →](https://medium.com/@sayedebad.777/building-a-miniature-rlhf-pipeline-from-scratch-2cea3e701878?postPublishedType=initial) |

Each folder is fully self-contained. Install that project's dependencies and run it independently.

---

## Setup

```bash
git clone https://github.com/ES7/reinforcement-learning-projects.git
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
- [PyTorch](https://pytorch.org/) — Neural networks for deep RL projects
- [NumPy](https://numpy.org/) & [Matplotlib](https://matplotlib.org/) — Computation and visualization

---

## Progress

- [x] Project 01 — Grid World Navigator
- [x] Project 02 — Blackjack Strategy Learner
- [x] Project 03 — CliffWalking TD vs MC
- [x] Project 04 — CliffWalking SARSA vs Q-Learning
- [x] Project 05 — LunarLander DQN
- [x] Project 06 — MountainCar A2C
- [x] Project 07 — BipedalWalker PPO
- [x] Project 08 — LLM Tool-Use Agent
- [x] Project 09 — Miniature RLHF Pipeline

---

## Author

**Ebad Sayed** — Final year, IIT (ISM) Dhanbad, Co-founder of Voke AI

Connect: [LinkedIn](https://www.linkedin.com/in/ebad-sayed-0861a6227/) · [GitHub](https://github.com/ES7) · [X](https://x.com/EbadOnAI)
