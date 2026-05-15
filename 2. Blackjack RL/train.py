import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import gymnasium as gym
from agent import MonteCarloAgent


def run_episode(env, agent):
    """Run one episode, return trajectory as list of (state, action, reward)."""
    obs, _ = env.reset()
    episode = []
    done = False

    while not done:
        state  = obs  # (player_sum, dealer_card, usable_ace)
        action = agent.select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))

    return episode


def train(n_episodes=500_000):
    env   = gym.make("Blackjack-v1", natural=False, sab=False)
    agent = MonteCarloAgent(
        gamma=1.0,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99997,
    )

    rewards_log  = []
    win_log      = []

    print("Training Monte Carlo agent on Blackjack...")
    print(f"{'Episode':>10} {'Avg Reward':>12} {'Win Rate':>10}")
    print("-" * 38)

    for ep in range(1, n_episodes + 1):
        episode = run_episode(env, agent)
        agent.update(episode)
        agent.decay_epsilon()

        final_reward = episode[-1][2]
        rewards_log.append(final_reward)
        win_log.append(1 if final_reward > 0 else 0)

        if ep % 50_000 == 0:
            avg_r = np.mean(rewards_log[-50_000:])
            wr    = np.mean(win_log[-50_000:]) * 100
            print(f"{ep:>10,} {avg_r:>12.4f} {wr:>9.1f}%")

    env.close()
    agent.save("mc_q.npy")
    print(f"\nFinal epsilon: {agent.epsilon:.4f}")
    return agent, rewards_log, win_log


def plot_training(rewards_log, win_log):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Blackjack — Monte Carlo Training", fontsize=13, fontweight='bold')

    window = 5000

    # Win rate over time
    wr = np.convolve(win_log, np.ones(window)/window, mode='valid') * 100
    axes[0].plot(range(window-1, len(win_log)), wr, color='#534AB7', linewidth=1.5)
    axes[0].axhline(43, linestyle='--', color='#D85A30', alpha=0.6, label='~43% human optimal')
    axes[0].set_title("Win rate over training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Win rate (%)")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(20, 60)

    # Reward distribution (last 50k episodes)
    last = rewards_log[-50_000:]
    vals, counts = np.unique(last, return_counts=True)
    colors_bar = ['#D85A30' if v < 0 else '#1D9E75' if v > 0 else '#888780' for v in vals]
    axes[1].bar([str(int(v)) for v in vals], counts/len(last)*100,
                color=colors_bar, edgecolor='white', width=0.6)
    axes[1].set_title("Reward distribution (last 50k episodes)")
    axes[1].set_xlabel("Reward")
    axes[1].set_ylabel("% of episodes")

    legend = [
        mpatches.Patch(color='#1D9E75', label='Win (+1)'),
        mpatches.Patch(color='#888780', label='Draw (0)'),
        mpatches.Patch(color='#D85A30', label='Loss (-1)'),
    ]
    axes[1].legend(handles=legend, fontsize=9)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → training_curves.png")
    plt.show()


def build_strategy_table(agent):
    """
    Build the classic Blackjack strategy chart from the learned Q-table.
    Returns two dicts: policy_no_ace, policy_ace
    keys: (player_sum, dealer_card)  values: 'H' or 'S'
    """
    actions = {0: 'S', 1: 'H'}  # 0=stick, 1=hit

    policy_no_ace = {}
    policy_ace    = {}

    for player_sum in range(4, 22):
        for dealer_card in range(1, 11):
            state_no_ace = (player_sum, dealer_card, False)
            state_ace    = (player_sum, dealer_card, True)

            policy_no_ace[(player_sum, dealer_card)] = actions[agent.get_policy(state_no_ace)]
            policy_ace[(player_sum, dealer_card)]    = actions[agent.get_policy(state_ace)]

    return policy_no_ace, policy_ace


def plot_strategy_chart(agent):
    """Plot the learned strategy chart — the iconic Blackjack visualization."""
    policy_no_ace, policy_ace = build_strategy_table(agent)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Blackjack Strategy — Learned via Monte Carlo (First-Visit)",
                 fontsize=13, fontweight='bold', y=1.01)

    dealer_cards  = list(range(1, 11))
    player_sums   = list(range(21, 3, -1))   # 21 down to 4
    dealer_labels = ['A','2','3','4','5','6','7','8','9','10']

    hit_color   = '#F0997B'   # coral — hit
    stick_color = '#9FE1CB'   # teal  — stick

    def draw_chart(ax, policy, title):
        data   = np.zeros((len(player_sums), len(dealer_cards)))
        labels = []

        for ri, ps in enumerate(player_sums):
            row_labels = []
            for ci, dc in enumerate(dealer_cards):
                action = policy.get((ps, dc), 'S')
                data[ri, ci] = 1 if action == 'H' else 0
                row_labels.append(action)
            labels.append(row_labels)

        cmap = mcolors.ListedColormap([stick_color, hit_color])
        ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Cell text
        for ri in range(len(player_sums)):
            for ci in range(len(dealer_cards)):
                action = labels[ri][ci]
                color  = '#712B13' if action == 'H' else '#085041'
                ax.text(ci, ri, action, ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

        ax.set_xticks(range(len(dealer_cards)))
        ax.set_xticklabels(dealer_labels, fontsize=10)
        ax.set_yticks(range(len(player_sums)))
        ax.set_yticklabels(player_sums, fontsize=9)
        ax.set_xlabel("Dealer's showing card", fontsize=11)
        ax.set_ylabel("Player sum", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

        # Grid lines
        for x in np.arange(-0.5, len(dealer_cards), 1):
            ax.axvline(x, color='white', linewidth=0.8)
        for y in np.arange(-0.5, len(player_sums), 1):
            ax.axhline(y, color='white', linewidth=0.8)

        legend = [
            mpatches.Patch(color=hit_color,   label='H — Hit'),
            mpatches.Patch(color=stick_color, label='S — Stick'),
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=9,
                  bbox_to_anchor=(1.0, -0.08), ncol=2)

    draw_chart(axes[0], policy_no_ace, "No usable ace")
    draw_chart(axes[1], policy_ace,    "Usable ace")

    plt.tight_layout()
    plt.savefig("strategy_chart.png", dpi=150, bbox_inches='tight')
    print("Saved → strategy_chart.png")
    plt.show()


def plot_value_function(agent):
    """3D surface plot of V(s) = max_a Q(s,a)."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Value Function V(s) — Blackjack", fontsize=13, fontweight='bold')

    dealer_cards = np.arange(1, 11)
    player_sums  = np.arange(12, 22)
    X, Y = np.meshgrid(dealer_cards, player_sums)

    for idx, (usable_ace, title) in enumerate([(False, "No usable ace"), (True, "Usable ace")]):
        Z = np.zeros_like(X, dtype=float)
        for ri, ps in enumerate(player_sums):
            for ci, dc in enumerate(dealer_cards):
                state = (ps, dc, usable_ace)
                Z[ri, ci] = np.max(agent.Q[state])

        ax = fig.add_subplot(1, 2, idx+1, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.9, linewidth=0)
        ax.set_xlabel("Dealer card", fontsize=9)
        ax.set_ylabel("Player sum",  fontsize=9)
        ax.set_zlabel("V(s)",        fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

    plt.tight_layout()
    plt.savefig("value_function.png", dpi=150, bbox_inches='tight')
    print("Saved → value_function.png")
    plt.show()


if __name__ == "__main__":
    agent, rewards_log, win_log = train(n_episodes=500_000)
    plot_training(rewards_log, win_log)
    plot_strategy_chart(agent)
    plot_value_function(agent)
    print("\nDone! Check strategy_chart.png, value_function.png, training_curves.png")
