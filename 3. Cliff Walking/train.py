import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
from agents import TDAgent, MCAgent


# ─────────────────────────────────────────────
# Episode runners
# ─────────────────────────────────────────────

def run_td_episode(env, agent):
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


def run_mc_episode(env, agent):
    obs, _ = env.reset()
    episode = []
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((obs, action, reward))
        obs = next_obs
    agent.update(episode)
    agent.decay_epsilon()
    return sum(r for _, _, r in episode)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(n_episodes=500):
    env_td = gym.make("CliffWalking-v1")
    env_mc = gym.make("CliffWalking-v1")

    n_states  = env_td.observation_space.n   # 48 (4x12 grid)
    n_actions = env_td.action_space.n        # 4

    td_agent = TDAgent(n_states, n_actions,
                       alpha=0.1, gamma=0.99,
                       epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

    mc_agent = MCAgent(n_states, n_actions,
                       gamma=0.99,
                       epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

    td_rewards = []
    mc_rewards = []

    print("Training TD and MC agents on CliffWalking...")
    print(f"{'Episode':>8} {'TD Reward':>12} {'MC Reward':>12}")
    print("-" * 38)

    for ep in range(1, n_episodes + 1):
        td_r = run_td_episode(env_td, td_agent)
        mc_r = run_mc_episode(env_mc, mc_agent)
        td_rewards.append(td_r)
        mc_rewards.append(mc_r)

        if ep % 50 == 0:
            print(f"{ep:>8} {np.mean(td_rewards[-50:]):>12.2f} {np.mean(mc_rewards[-50:]):>12.2f}")

    env_td.close()
    env_mc.close()

    np.save("td_q.npy", td_agent.Q)
    np.save("mc_q.npy", mc_agent.Q)
    print("\nQ-tables saved.")

    return td_agent, mc_agent, td_rewards, mc_rewards


# ─────────────────────────────────────────────
# Plot 1 — Reward curves head-to-head
# ─────────────────────────────────────────────

def plot_reward_curves(td_rewards, mc_rewards, window=20):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("CliffWalking — TD(0) vs Monte Carlo", fontsize=13, fontweight='bold')

    episodes = np.arange(1, len(td_rewards) + 1)

    # Raw rewards
    axes[0].plot(episodes, td_rewards, alpha=0.25, color='#534AB7', linewidth=0.8)
    axes[0].plot(episodes, mc_rewards, alpha=0.25, color='#D85A30', linewidth=0.8)

    # Smoothed
    td_smooth = np.convolve(td_rewards, np.ones(window)/window, mode='valid')
    mc_smooth = np.convolve(mc_rewards, np.ones(window)/window, mode='valid')
    x_smooth  = np.arange(window, len(td_rewards) + 1)

    axes[0].plot(x_smooth, td_smooth, color='#534AB7', linewidth=2, label='TD(0)')
    axes[0].plot(x_smooth, mc_smooth, color='#D85A30', linewidth=2, label='Monte Carlo')
    axes[0].axhline(-13, linestyle='--', color='#1D9E75', alpha=0.7, linewidth=1.2, label='Optimal (-13)')
    axes[0].set_title("Episode reward over training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-200, 5)

    # Variance comparison — last 100 episodes
    last_td = td_rewards[-100:]
    last_mc = mc_rewards[-100:]
    labels  = ['TD(0)', 'Monte Carlo']
    means   = [np.mean(last_td), np.mean(last_mc)]
    stds    = [np.std(last_td),  np.std(last_mc)]
    colors  = ['#534AB7', '#D85A30']

    bars = axes[1].bar(labels, means, color=colors, edgecolor='white',
                       width=0.4, yerr=stds, capsize=8,
                       error_kw=dict(ecolor='#555', elinewidth=1.5))
    axes[1].set_title("Avg reward ± std (last 100 episodes)")
    axes[1].set_ylabel("Reward")

    for bar, mean, std in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     mean + std + 2,
                     f'{mean:.1f} ± {std:.1f}',
                     ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("reward_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → reward_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 2 — Policy paths on the grid
# ─────────────────────────────────────────────

def get_greedy_path(Q, env, max_steps=100):
    """Follow greedy policy, return list of (row, col) visited."""
    obs, _ = env.reset()
    path = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        row, col = divmod(obs, 12)
        path.append((row, col))
        action = int(np.argmax(Q[obs]))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    row, col = divmod(obs, 12)
    path.append((row, col))
    return path


def plot_paths(td_agent, mc_agent):
    env = gym.make("CliffWalking-v1")
    td_path = get_greedy_path(td_agent.Q, env)
    env.reset()
    mc_path = get_greedy_path(mc_agent.Q, env)
    env.close()

    rows, cols = 4, 12

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Learned Paths — TD(0) vs Monte Carlo", fontsize=13, fontweight='bold')

    colors_map = {
        'empty': '#F5F5F2',
        'cliff': '#323238',
        'start': '#F0997B',
        'goal':  '#1D9E75',
    }

    cliff_cells = [(3, c) for c in range(1, 11)]

    for ax, path, title, color in zip(
        axes,
        [td_path, mc_path],
        ['TD(0) — hugs the cliff edge', 'Monte Carlo — takes the safe route'],
        ['#534AB7', '#D85A30']
    ):
        ax.set_facecolor(colors_map['empty'])
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True, color='#CCCCCC', linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                if (r, c) in cliff_cells:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                               color=colors_map['cliff'], zorder=2))
                    ax.text(c, r, '✕', ha='center', va='center',
                            fontsize=7, color='white', zorder=3)

        # Start & goal
        ax.add_patch(plt.Rectangle((-0.5, 2.5), 1, 1, color=colors_map['start'], zorder=2))
        ax.text(0, 3, 'S', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=3)
        ax.add_patch(plt.Rectangle((10.5, 2.5), 1, 1, color=colors_map['goal'], zorder=2))
        ax.text(11, 3, 'G', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=3)

        # Draw path
        if len(path) > 1:
            path_rows = [p[0] for p in path]
            path_cols = [p[1] for p in path]
            ax.plot(path_cols, path_rows, color=color, linewidth=2.5,
                    marker='o', markersize=5, zorder=5, label=f'{len(path)-1} steps')
            # Mark start of path
            ax.plot(path_cols[0], path_rows[0], 'o', color=color,
                    markersize=10, zorder=6)

        ax.legend(loc='upper right', fontsize=9)

        legend = [
            mpatches.Patch(color=colors_map['cliff'], label='Cliff (-100)'),
            mpatches.Patch(color=colors_map['start'], label='Start'),
            mpatches.Patch(color=colors_map['goal'],  label='Goal'),
        ]
        ax.legend(handles=legend + [
            plt.Line2D([0],[0], color=color, linewidth=2, label=f'Path ({len(path)-1} steps)')
        ], fontsize=8, loc='upper center')

    plt.tight_layout()
    plt.savefig("paths.png", dpi=150, bbox_inches='tight')
    print("Saved → paths.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 3 — Value function heatmaps
# ─────────────────────────────────────────────

def plot_value_heatmaps(td_agent, mc_agent):
    rows, cols = 4, 12
    cliff_cells = set((3, c) for c in range(1, 11))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Value Function V(s) — TD(0) vs Monte Carlo", fontsize=13, fontweight='bold')

    for ax, agent, title in zip(
        axes,
        [td_agent, mc_agent],
        ['TD(0)', 'Monte Carlo']
    ):
        V = np.max(agent.Q, axis=1).reshape(rows, cols).astype(float)

        # Mask cliff cells
        mask = np.zeros((rows, cols), dtype=bool)
        for (r, c) in cliff_cells:
            mask[r, c] = True
        V_masked = np.ma.array(V, mask=mask)

        im = ax.imshow(V_masked, cmap='RdYlGn', interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax, label='V(s)')

        for r in range(rows):
            for c in range(cols):
                if (r, c) in cliff_cells:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#323238'))
                    ax.text(c, r, '✕', ha='center', va='center',
                            fontsize=8, color='white')
                elif (r, c) == (3, 0):
                    ax.text(c, r, 'S', ha='center', va='center',
                            fontsize=10, fontweight='bold')
                elif (r, c) == (3, 11):
                    ax.text(c, r, 'G', ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white')
                else:
                    ax.text(c, r, f'{V[r,c]:.0f}', ha='center', va='center',
                            fontsize=7, color='#222')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

    plt.tight_layout()
    plt.savefig("value_heatmaps.png", dpi=150, bbox_inches='tight')
    print("Saved → value_heatmaps.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    td_agent, mc_agent, td_rewards, mc_rewards = train(n_episodes=500)
    plot_reward_curves(td_rewards, mc_rewards)
    plot_paths(td_agent, mc_agent)
    plot_value_heatmaps(td_agent, mc_agent)
    print("\nDone! Check reward_curves.png, paths.png, value_heatmaps.png")
