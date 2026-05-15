import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import torch
from agent import DQNAgent


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(n_episodes=600):
    env = gym.make("LunarLander-v3")

    state_dim  = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.n               # 4

    agent = DQNAgent(
        state_dim     = state_dim,
        action_dim    = action_dim,
        lr            = 1e-3,
        gamma         = 0.99,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
        batch_size    = 64,
        buffer_cap    = 50_000,
        target_update = 10,
    )

    rewards_log  = []
    steps_log    = []
    loss_log     = []
    epsilon_log  = []
    solved_ep    = None   # first episode where avg >= 200

    print(f"Training DQN on LunarLander-v3  |  device: {agent.device}")
    print(f"State dim: {state_dim}  |  Action dim: {action_dim}")
    print(f"{'Episode':>8} {'Reward':>10} {'Steps':>7} {'Epsilon':>9} {'Avg(50)':>9}")
    print("-" * 52)

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        ep_losses = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        # Sync target network every target_update episodes
        if ep % agent.target_update == 0:
            agent.sync_target()

        rewards_log.append(total_reward)
        steps_log.append(env._elapsed_steps if hasattr(env, '_elapsed_steps') else 0)
        epsilon_log.append(agent.epsilon)
        if ep_losses:
            loss_log.append(np.mean(ep_losses))

        avg50 = np.mean(rewards_log[-50:]) if len(rewards_log) >= 50 else np.mean(rewards_log)

        if avg50 >= 200 and solved_ep is None:
            solved_ep = ep
            print(f"  *** SOLVED at episode {ep} (avg50 = {avg50:.1f}) ***")

        if ep % 50 == 0:
            print(f"{ep:>8} {total_reward:>10.1f} {len(rewards_log):>7} "
                  f"{agent.epsilon:>9.3f} {avg50:>9.1f}")

    env.close()
    agent.save("dqn.pth")

    return agent, rewards_log, loss_log, epsilon_log, solved_ep


# ─────────────────────────────────────────────
# Plot 1 — Training curves
# ─────────────────────────────────────────────

def plot_training_curves(rewards_log, loss_log, epsilon_log, solved_ep=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("LunarLander — DQN Training", fontsize=13, fontweight='bold')

    window  = 50
    episodes = np.arange(1, len(rewards_log) + 1)

    # Reward
    axes[0].plot(episodes, rewards_log, alpha=0.2, color='#534AB7', linewidth=0.8)
    smoothed = np.convolve(rewards_log, np.ones(window)/window, mode='valid')
    axes[0].plot(np.arange(window, len(rewards_log)+1), smoothed,
                 color='#534AB7', linewidth=2, label='Avg(50)')
    axes[0].axhline(200, linestyle='--', color='#1D9E75', linewidth=1.2, alpha=0.8, label='Solved (200)')
    if solved_ep:
        axes[0].axvline(solved_ep, linestyle=':', color='#D85A30', linewidth=1.5,
                        label=f'First solved ep {solved_ep}')
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-400, 300)

    # Loss
    if loss_log:
        loss_smooth = np.convolve(loss_log, np.ones(20)/20, mode='valid')
        axes[1].plot(loss_log, alpha=0.2, color='#D85A30', linewidth=0.8)
        axes[1].plot(range(20, len(loss_log)+1), loss_smooth,
                     color='#D85A30', linewidth=2)
    axes[1].set_title("Training loss (MSE)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")

    # Epsilon decay
    axes[2].plot(episodes, epsilon_log, color='#1D9E75', linewidth=2)
    axes[2].set_title("Epsilon decay")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_ylim(0, 1.05)
    axes[2].fill_between(episodes, epsilon_log, alpha=0.15, color='#1D9E75')

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → training_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 2 — Q-value estimates vs actual returns
# ─────────────────────────────────────────────

def plot_q_vs_actual(agent, n_eval=50):
    """
    Run greedy episodes, collect Q estimates and actual returns.
    Shows whether the network is overestimating or underestimating.
    """
    env = gym.make("LunarLander-v3")
    agent.epsilon = 0.0   # pure greedy

    q_estimates_ep  = []
    actual_returns  = []

    for _ in range(n_eval):
        state, _ = env.reset()
        done = False
        ep_return = 0
        ep_q_vals = []

        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q = agent.online_net(s)
                ep_q_vals.append(q.max().item())
            action = q.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        q_estimates_ep.append(np.mean(ep_q_vals))
        actual_returns.append(ep_return)

    env.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Q-Value Estimates vs Actual Returns", fontsize=13, fontweight='bold')

    ep_range = np.arange(1, n_eval + 1)

    axes[0].plot(ep_range, q_estimates_ep, color='#534AB7', linewidth=2,
                 marker='o', markersize=4, label='Avg Q estimate')
    axes[0].plot(ep_range, actual_returns,  color='#D85A30', linewidth=2,
                 marker='s', markersize=4, label='Actual return')
    axes[0].set_title("Q estimate vs actual return per episode")
    axes[0].set_xlabel("Eval episode")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, linestyle='--', color='#AAA', linewidth=0.8)

    # Scatter: Q estimate vs actual
    axes[1].scatter(actual_returns, q_estimates_ep,
                    color='#534AB7', alpha=0.6, edgecolors='white', linewidth=0.5, s=50)
    mn = min(min(actual_returns), min(q_estimates_ep)) - 20
    mx = max(max(actual_returns), max(q_estimates_ep)) + 20
    axes[1].plot([mn, mx], [mn, mx], '--', color='#D85A30',
                 linewidth=1.5, label='Perfect estimate (y=x)')
    axes[1].set_title("Q estimate vs actual (scatter)")
    axes[1].set_xlabel("Actual return")
    axes[1].set_ylabel("Avg Q estimate")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("q_vs_actual.png", dpi=150, bbox_inches='tight')
    print("Saved → q_vs_actual.png")
    plt.show()

    print(f"\nAvg Q estimate : {np.mean(q_estimates_ep):.2f}")
    print(f"Avg actual return : {np.mean(actual_returns):.2f}")
    print(f"Overestimation bias: {np.mean(q_estimates_ep) - np.mean(actual_returns):.2f}")


# ─────────────────────────────────────────────
# Plot 3 — Network architecture diagram
# ─────────────────────────────────────────────

def plot_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#F5F5F2')
    ax.set_facecolor('#F5F5F2')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("DQN Architecture — LunarLander", fontsize=13, fontweight='bold', pad=15)

    layers = [
        (1.0,  "State\n(8 inputs)",       ['x pos', 'y pos', 'x vel', 'y vel',
                                            'angle', 'ang vel', 'leg L', 'leg R'], '#F0997B'),
        (4.0,  "Hidden 1\n(128 neurons)",  None, '#534AB7'),
        (7.0,  "Hidden 2\n(128 neurons)",  None, '#534AB7'),
        (10.0, "Q-Values\n(4 outputs)",    ['Nothing', 'Left\nengine', 'Main\nengine',
                                            'Right\nengine'], '#1D9E75'),
    ]

    node_positions = {}

    for x, label, nodes, color in layers:
        if nodes:
            n = len(nodes)
            ys = np.linspace(1, 5, n)
            node_positions[x] = ys
            for y, node_label in zip(ys, nodes):
                circ = plt.Circle((x, y), 0.28, color=color, zorder=3, alpha=0.85)
                ax.add_patch(circ)
                ax.text(x, y, node_label, ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold', zorder=4)
        else:
            # Show a subset of neurons for hidden layers
            ys = np.linspace(0.8, 5.2, 7)
            node_positions[x] = ys
            for i, y in enumerate(ys):
                if i == 3:
                    ax.text(x, y, '···', ha='center', va='center',
                            fontsize=14, color=color, zorder=4)
                else:
                    circ = plt.Circle((x, y), 0.22, color=color, zorder=3, alpha=0.85)
                    ax.add_patch(circ)

        # Layer label
        ax.text(x, 5.7, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#323238')

    # Draw connections (sample, not all)
    for i in range(len(layers) - 1):
        x1 = layers[i][0]
        x2 = layers[i+1][0]
        ys1 = node_positions[x1]
        ys2 = node_positions[x2]
        for y1 in ys1[::2]:
            for y2 in ys2[::2]:
                ax.plot([x1+0.28, x2-0.28], [y1, y2],
                        color='#CCCCCC', linewidth=0.4, zorder=1, alpha=0.6)

    # Activation labels
    for x, label in [(2.5, 'ReLU'), (5.5, 'ReLU')]:
        ax.text(x, 0.2, label, ha='center', va='center',
                fontsize=8, color='#534AB7', style='italic')

    plt.tight_layout()
    plt.savefig("architecture.png", dpi=150, bbox_inches='tight')
    print("Saved → architecture.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 4 — Target vs Online network divergence
# ─────────────────────────────────────────────

def plot_target_network_effect(rewards_log):
    """Show that target network updates cause step-changes in learning."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Effect of Target Network Sync (every 10 episodes)",
                 fontsize=12, fontweight='bold')

    window   = 10
    episodes = np.arange(1, len(rewards_log) + 1)
    smoothed = np.convolve(rewards_log, np.ones(window)/window, mode='valid')

    ax.plot(rewards_log, alpha=0.15, color='#534AB7', linewidth=0.8)
    ax.plot(range(window, len(rewards_log)+1), smoothed,
            color='#534AB7', linewidth=2, label='Smoothed reward')

    # Mark every target network sync
    sync_eps = list(range(10, len(rewards_log)+1, 10))
    for sep in sync_eps[:30]:   # only first 30 to avoid clutter
        ax.axvline(sep, color='#D85A30', linewidth=0.5, alpha=0.3)

    ax.axvline(sync_eps[0], color='#D85A30', linewidth=0.5,
               alpha=0.5, label='Target net sync')
    ax.axhline(200, linestyle='--', color='#1D9E75', linewidth=1.2,
               alpha=0.8, label='Solved threshold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend(fontsize=9)
    ax.set_ylim(-400, 300)

    plt.tight_layout()
    plt.savefig("target_network_effect.png", dpi=150, bbox_inches='tight')
    print("Saved → target_network_effect.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent, rewards_log, loss_log, epsilon_log, solved_ep = train(n_episodes=600)

    plot_training_curves(rewards_log, loss_log, epsilon_log, solved_ep)
    plot_q_vs_actual(agent, n_eval=50)
    plot_architecture()
    plot_target_network_effect(rewards_log)

    print("\nDone! Check training_curves.png, q_vs_actual.png, architecture.png, target_network_effect.png")
