import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import gymnasium as gym
from agents import SARSAAgent, QLearningAgent


ROWS, COLS   = 4, 12
CLIFF_CELLS  = set((3, c) for c in range(1, 11))
START        = (3, 0)
GOAL         = (3, 11)

SARSA_COLOR  = '#534AB7'   # purple
QL_COLOR     = '#D85A30'   # coral


# ─────────────────────────────────────────────
# Episode runners
# ─────────────────────────────────────────────

def run_sarsa_episode(env, agent):
    """
    SARSA needs to know next_action BEFORE the update.
    So we select a' inside the loop before stepping.
    """
    obs, _ = env.reset()
    action = agent.select_action(obs)
    total_reward = 0
    done = False

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = agent.select_action(next_obs)
        agent.update(obs, action, reward, next_obs, next_action, done)
        obs    = next_obs
        action = next_action
        total_reward += reward

    agent.decay_epsilon()
    return total_reward


def run_qlearning_episode(env, agent):
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
    return total_reward


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(n_episodes=500):
    env_s = gym.make("CliffWalking-v1")
    env_q = gym.make("CliffWalking-v1")

    n_states  = env_s.observation_space.n
    n_actions = env_s.action_space.n

    sarsa_agent = SARSAAgent(n_states, n_actions,
                             alpha=0.1, gamma=0.99,
                             epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

    ql_agent = QLearningAgent(n_states, n_actions,
                              alpha=0.1, gamma=0.99,
                              epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

    sarsa_rewards = []
    ql_rewards    = []

    print("Training SARSA and Q-Learning agents on CliffWalking...")
    print(f"{'Episode':>8} {'SARSA Reward':>14} {'Q-Learning Reward':>18}")
    print("-" * 46)

    for ep in range(1, n_episodes + 1):
        sr = run_sarsa_episode(env_s, sarsa_agent)
        qr = run_qlearning_episode(env_q, ql_agent)
        sarsa_rewards.append(sr)
        ql_rewards.append(qr)

        if ep % 50 == 0:
            print(f"{ep:>8} {np.mean(sarsa_rewards[-50:]):>14.2f} {np.mean(ql_rewards[-50:]):>18.2f}")

    env_s.close()
    env_q.close()

    np.save("sarsa_q.npy", sarsa_agent.Q)
    np.save("ql_q.npy",    ql_agent.Q)
    print("\nQ-tables saved.")

    return sarsa_agent, ql_agent, sarsa_rewards, ql_rewards


# ─────────────────────────────────────────────
# Shared grid drawing helper
# ─────────────────────────────────────────────

def draw_grid(ax):
    ax.set_facecolor('#F5F5F2')
    ax.set_autoscale_on(False)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True, color='#CCCCCC', linewidth=0.5, zorder=0)

    for (r, c) in CLIFF_CELLS:
        ax.add_patch(plt.Rectangle(
            (float(c)-0.5, float(r)-0.5), 1.0, 1.0,
            color='#323238', zorder=2, transform=ax.transData))
        ax.text(c, r, 'X', ha='center', va='center',
                fontsize=7, color='#888', zorder=3)

    # Start
    ax.add_patch(plt.Rectangle(
        (float(START[1])-0.5, float(START[0])-0.5), 1.0, 1.0,
        color='#F0997B', zorder=2, transform=ax.transData))
    ax.text(START[1], START[0], 'S', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=3)

    # Goal
    ax.add_patch(plt.Rectangle(
        (float(GOAL[1])-0.5, float(GOAL[0])-0.5), 1.0, 1.0,
        color='#1D9E75', zorder=2, transform=ax.transData))
    ax.text(GOAL[1], GOAL[0], 'G', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=3)

    # Set limits AFTER patches
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)


# ─────────────────────────────────────────────
# Get greedy path
# ─────────────────────────────────────────────

def get_greedy_path(Q, env, max_steps=200):
    obs, _ = env.reset()
    path = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        row, col = divmod(obs, COLS)
        path.append((row, col))
        action = int(np.argmax(Q[obs]))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    row, col = divmod(obs, COLS)
    path.append((row, col))
    return path


# ─────────────────────────────────────────────
# Plot 1 — Reward curves
# ─────────────────────────────────────────────

def plot_reward_curves(sarsa_rewards, ql_rewards, window=20):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("CliffWalking — SARSA vs Q-Learning", fontsize=13, fontweight='bold')

    episodes = np.arange(1, len(sarsa_rewards) + 1)

    # Raw (faded)
    axes[0].plot(episodes, sarsa_rewards, alpha=0.2, color=SARSA_COLOR, linewidth=0.8)
    axes[0].plot(episodes, ql_rewards,    alpha=0.2, color=QL_COLOR,    linewidth=0.8)

    # Smoothed
    s_smooth = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    q_smooth = np.convolve(ql_rewards,    np.ones(window)/window, mode='valid')
    x_smooth = np.arange(window, len(sarsa_rewards) + 1)

    axes[0].plot(x_smooth, s_smooth, color=SARSA_COLOR, linewidth=2, label='SARSA (on-policy)')
    axes[0].plot(x_smooth, q_smooth, color=QL_COLOR,    linewidth=2, label='Q-Learning (off-policy)')
    axes[0].axhline(-13, linestyle='--', color='#1D9E75', alpha=0.7,
                    linewidth=1.2, label='Optimal (−13)')
    axes[0].set_title("Episode reward over training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-150, 5)

    # Avg + std last 100 episodes
    last_s = sarsa_rewards[-100:]
    last_q = ql_rewards[-100:]
    means  = [np.mean(last_s), np.mean(last_q)]
    stds   = [np.std(last_s),  np.std(last_q)]
    colors = [SARSA_COLOR, QL_COLOR]
    labels = ['SARSA', 'Q-Learning']

    bars = axes[1].bar(labels, means, color=colors, edgecolor='white',
                       width=0.4, yerr=stds, capsize=8,
                       error_kw=dict(ecolor='#555', elinewidth=1.5))
    axes[1].set_title("Avg reward ± std (last 100 episodes)")
    axes[1].set_ylabel("Reward")

    for bar, mean, std in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     mean + std + 1,
                     f'{mean:.1f} ± {std:.1f}',
                     ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("reward_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → reward_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 2 — Side by side path comparison
# ─────────────────────────────────────────────

def plot_paths_sidebyside(sarsa_agent, ql_agent):
    env = gym.make("CliffWalking-v1")
    sarsa_path = get_greedy_path(sarsa_agent.Q, env)
    env.reset()
    ql_path = get_greedy_path(ql_agent.Q, env)
    env.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Learned Paths — SARSA vs Q-Learning", fontsize=13, fontweight='bold')

    for ax, path, title, color in zip(
        axes,
        [sarsa_path, ql_path],
        [f'SARSA — safe route  ({len(sarsa_path)-1} steps)',
         f'Q-Learning — cliff edge  ({len(ql_path)-1} steps)'],
        [SARSA_COLOR, QL_COLOR]
    ):
        draw_grid(ax)
        ax.set_title(title, fontsize=11, fontweight='bold')

        if len(path) > 1:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, color=color, linewidth=2.5,
                    marker='o', markersize=5, zorder=5)
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, zorder=6)

        legend_items = [
            mpatches.Patch(color='#323238', label='Cliff (−100)'),
            mpatches.Patch(color='#F0997B', label='Start'),
            mpatches.Patch(color='#1D9E75', label='Goal'),
            plt.Line2D([0],[0], color=color, linewidth=2,
                       marker='o', markersize=5,
                       label=f'Path ({len(path)-1} steps)')
        ]
        ax.legend(handles=legend_items, fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig("paths_sidebyside.png", dpi=150, bbox_inches='tight')
    print("Saved → paths_sidebyside.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 3 — Overlay: both paths on one grid
# ─────────────────────────────────────────────

def plot_paths_overlay(sarsa_agent, ql_agent):
    env = gym.make("CliffWalking-v1")
    sarsa_path = get_greedy_path(sarsa_agent.Q, env)
    env.reset()
    ql_path = get_greedy_path(ql_agent.Q, env)
    env.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Path Overlay — SARSA vs Q-Learning on CliffWalking",
                 fontsize=13, fontweight='bold')

    draw_grid(ax)

    for path, color, label, offset in [
        (sarsa_path, SARSA_COLOR, 'SARSA (safe)',     -0.12),
        (ql_path,    QL_COLOR,    'Q-Learning (risky)', +0.12),
    ]:
        xs = [p[1] + offset for p in path]
        ys = [p[0] + offset for p in path]
        ax.plot(xs, ys, color=color, linewidth=2.5,
                marker='o', markersize=5, zorder=5, label=label)
        ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, zorder=6)

    legend_items = [
        mpatches.Patch(color='#323238', label='Cliff (−100)'),
        mpatches.Patch(color='#F0997B', label='Start'),
        mpatches.Patch(color='#1D9E75', label='Goal'),
        plt.Line2D([0],[0], color=SARSA_COLOR, linewidth=2.5,
                   marker='o', markersize=5, label=f'SARSA ({len(sarsa_path)-1} steps)'),
        plt.Line2D([0],[0], color=QL_COLOR, linewidth=2.5,
                   marker='o', markersize=5, label=f'Q-Learning ({len(ql_path)-1} steps)'),
    ]
    ax.legend(handles=legend_items, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig("paths_overlay.png", dpi=150, bbox_inches='tight')
    print("Saved → paths_overlay.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 4 — Policy arrow grids
# ─────────────────────────────────────────────

def plot_policy_arrows(sarsa_agent, ql_agent):
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Greedy Policy — SARSA vs Q-Learning", fontsize=13, fontweight='bold')

    for ax, agent, title, color in zip(
        axes,
        [sarsa_agent, ql_agent],
        ['SARSA', 'Q-Learning'],
        [SARSA_COLOR, QL_COLOR]
    ):
        draw_grid(ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        policy = agent.get_policy().reshape(ROWS, COLS)

        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in CLIFF_CELLS:
                    continue
                if (r, c) == START or (r, c) == GOAL:
                    continue
                ax.text(c, r, arrow_map[policy[r, c]],
                        ha='center', va='center',
                        fontsize=14, color=color, zorder=4)

    plt.tight_layout()
    plt.savefig("policy_arrows.png", dpi=150, bbox_inches='tight')
    print("Saved → policy_arrows.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sarsa_agent, ql_agent, sarsa_rewards, ql_rewards = train(n_episodes=500)
    plot_reward_curves(sarsa_rewards, ql_rewards)
    plot_paths_sidebyside(sarsa_agent, ql_agent)
    plot_paths_overlay(sarsa_agent, ql_agent)
    plot_policy_arrows(sarsa_agent, ql_agent)
    print("\nDone! Check reward_curves.png, paths_sidebyside.png, paths_overlay.png, policy_arrows.png")