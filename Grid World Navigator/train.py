import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from grid_env import GridWorldEnv
from agent import QLearningAgent


def train(n_episodes=2000, render_every=None):
    env   = GridWorldEnv()
    agent = QLearningAgent(
        n_states=64, n_actions=4,
        alpha=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995
    )

    rewards_log  = []
    steps_log    = []
    success_log  = []

    print("Training Q-Learning agent on Grid World...")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} {'Epsilon':>9} {'Success':>9}")
    print("-" * 50)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        rewards_log.append(total_reward)
        steps_log.append(env.steps)
        success_log.append(1 if tuple(env.agent_pos) == env.goal else 0)

        if ep % 200 == 0:
            avg_r = np.mean(rewards_log[-200:])
            avg_s = np.mean(steps_log[-200:])
            sr    = np.mean(success_log[-200:]) * 100
            print(f"{ep:>8} {avg_r:>10.3f} {avg_s:>7.1f} {agent.epsilon:>9.3f} {sr:>8.1f}%")

    env.close()
    agent.save("q_table.npy")
    return agent, rewards_log, steps_log, success_log


def plot_training(rewards_log, steps_log, success_log):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Grid World — Q-Learning Training", fontsize=14, fontweight='bold')

    window = 50

    # Smoothed rewards
    smoothed = np.convolve(rewards_log, np.ones(window)/window, mode='valid')
    axes[0].plot(rewards_log, alpha=0.2, color='#7F77DD', linewidth=0.8)
    axes[0].plot(range(window-1, len(rewards_log)), smoothed, color='#534AB7', linewidth=2)
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")

    # Steps per episode
    smoothed_steps = np.convolve(steps_log, np.ones(window)/window, mode='valid')
    axes[1].plot(steps_log, alpha=0.2, color='#5DCAA5', linewidth=0.8)
    axes[1].plot(range(window-1, len(steps_log)), smoothed_steps, color='#0F6E56', linewidth=2)
    axes[1].set_title("Steps per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    # Success rate rolling
    sr = np.convolve(success_log, np.ones(window)/window, mode='valid') * 100
    axes[2].plot(range(window-1, len(success_log)), sr, color='#D85A30', linewidth=2)
    axes[2].axhline(100, linestyle='--', color='#993C1D', alpha=0.4)
    axes[2].set_title("Success rate (%)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("% episodes reaching goal")
    axes[2].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → training_curves.png")
    plt.show()


def plot_policy_and_values(agent):
    """Visualize Q-values and the learned greedy policy on the grid."""
    env = GridWorldEnv()
    grid = env.grid_size
    walls = env.walls

    V = np.max(agent.Q, axis=1).reshape(grid, grid)
    policy = agent.get_policy().reshape(grid, grid)
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Grid World — Learned Policy & Value Function", fontsize=13, fontweight='bold')

    # --- Value heatmap ---
    V_plot = V.copy()
    mask = np.zeros((grid, grid), dtype=bool)
    for (r, c) in walls:
        mask[r, c] = True

    V_masked = np.ma.array(V_plot, mask=mask)
    im = axes[0].imshow(V_masked, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(im, ax=axes[0], label='V(s) = max_a Q(s,a)')

    for r in range(grid):
        for c in range(grid):
            if (r, c) in walls:
                axes[0].add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#323238'))
            elif (r, c) == env.goal:
                axes[0].text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            elif (r, c) == env.start:
                axes[0].text(c, r, 'S', ha='center', va='center', fontsize=11, fontweight='bold', color='#323238')

    axes[0].set_title("Value function V(s)")
    axes[0].set_xticks(range(grid))
    axes[0].set_yticks(range(grid))

    # --- Policy arrows ---
    axes[1].set_facecolor('#F5F5F2')
    for r in range(grid):
        for c in range(grid):
            if (r, c) in walls:
                axes[1].add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#323238', zorder=2))
            elif (r, c) == env.goal:
                axes[1].add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#1D9E75', zorder=2))
                axes[1].text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=3)
            elif (r, c) == env.start:
                axes[1].add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#F0997B', zorder=2))
                axes[1].text(c, r, 'S', ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=3)
            else:
                axes[1].text(c, r, arrows[policy[r, c]], ha='center', va='center', fontsize=18, color='#534AB7', zorder=3)

    axes[1].set_xlim(-0.5, grid - 0.5)
    axes[1].set_ylim(grid - 0.5, -0.5)
    axes[1].set_xticks(range(grid))
    axes[1].set_yticks(range(grid))
    axes[1].grid(True, color='#CCCCCC', linewidth=0.5)
    axes[1].set_title("Greedy policy π(s)")

    legend = [
        mpatches.Patch(color='#F0997B', label='Start'),
        mpatches.Patch(color='#1D9E75', label='Goal'),
        mpatches.Patch(color='#323238', label='Wall'),
    ]
    axes[1].legend(handles=legend, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig("policy_values.png", dpi=150, bbox_inches='tight')
    print("Saved → policy_values.png")
    plt.show()


if __name__ == "__main__":
    agent, rewards, steps, successes = train(n_episodes=2000)
    plot_training(rewards, steps, successes)
    plot_policy_and_values(agent)
    print("\nDone! Check training_curves.png and policy_values.png")
