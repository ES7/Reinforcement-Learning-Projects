import numpy as np
import gymnasium as gym
from agent import MonteCarloAgent


def evaluate(agent, n_episodes=100_000, greedy=True):
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    old_eps = agent.epsilon
    if greedy:
        agent.epsilon = 0.0

    wins = draws = losses = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:   wins   += 1
        elif reward == 0: draws  += 1
        else:             losses += 1

    agent.epsilon = old_eps
    env.close()
    return wins, draws, losses


def compare_random_vs_trained(agent, n_episodes=100_000):
    print(f"\nEvaluating over {n_episodes:,} episodes each...\n")

    # Random player
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    rw = rd = rl = 0
    for _ in range(n_episodes):
        env.reset()
        done = False
        reward = 0
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
        if reward > 0:    rw += 1
        elif reward == 0: rd += 1
        else:             rl += 1
    env.close()

    # Trained MC agent
    tw, td, tl = evaluate(agent, n_episodes=n_episodes, greedy=True)

    print("=" * 48)
    print(f"{'':18} {'Random':>12} {'MC Trained':>12}")
    print("=" * 48)
    print(f"{'Win rate':18} {rw/n_episodes*100:>11.1f}% {tw/n_episodes*100:>11.1f}%")
    print(f"{'Draw rate':18} {rd/n_episodes*100:>11.1f}% {td/n_episodes*100:>11.1f}%")
    print(f"{'Loss rate':18} {rl/n_episodes*100:>11.1f}% {tl/n_episodes*100:>11.1f}%")
    print("=" * 48)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Random Player vs Monte Carlo Agent", fontsize=13, fontweight='bold')

    categories  = ['Win', 'Draw', 'Loss']
    random_vals = [rw/n_episodes*100, rd/n_episodes*100, rl/n_episodes*100]
    mc_vals     = [tw/n_episodes*100, td/n_episodes*100, tl/n_episodes*100]
    bar_colors  = ['#1D9E75', '#888780', '#D85A30']

    x = np.arange(3)
    w = 0.35
    axes[0].bar(x - w/2, random_vals, w, label='Random', color=bar_colors, alpha=0.45, edgecolor='white')
    axes[0].bar(x + w/2, mc_vals,     w, label='MC Agent', color=bar_colors, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("% of episodes")
    axes[0].set_title("Outcome breakdown")
    axes[0].legend()

    # Win rate only bar
    axes[1].bar(['Random', 'MC Agent'],
                [rw/n_episodes*100, tw/n_episodes*100],
                color=['#D3D1C7', '#534AB7'], edgecolor='white', width=0.45)
    axes[1].set_ylabel("Win rate (%)")
    axes[1].set_title("Win rate comparison")
    axes[1].set_ylim(0, 55)
    for i, v in enumerate([rw/n_episodes*100, tw/n_episodes*100]):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
    print("Saved → comparison.png")
    plt.show()


def query_strategy(agent):
    """Print what the agent would do for any state."""
    print("\nStrategy lookup (type 'q' to quit)")
    print("Format: player_sum dealer_card usable_ace(y/n)")
    print("Example: 16 7 n\n")

    while True:
        inp = input("State: ").strip().lower()
        if inp == 'q':
            break
        try:
            parts = inp.split()
            ps = int(parts[0])
            dc = int(parts[1])
            ua = parts[2] == 'y'
            state  = (ps, dc, ua)
            action = agent.get_policy(state)
            label  = "HIT" if action == 1 else "STICK"
            q_vals = agent.Q[state]
            print(f"  → {label}  (Q_stick={q_vals[0]:.3f}, Q_hit={q_vals[1]:.3f})\n")
        except Exception:
            print("  Invalid input, try again.\n")


if __name__ == "__main__":
    agent = MonteCarloAgent()
    agent.load("mc_q.npy")
    agent.epsilon = 0.0

    compare_random_vs_trained(agent, n_episodes=100_000)
    query_strategy(agent)
