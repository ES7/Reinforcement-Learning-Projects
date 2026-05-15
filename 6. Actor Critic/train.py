import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import torch
from agent import A2CAgent


ROLLOUT_LEN = 64   # steps per update


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(n_episodes=800):
    env = gym.make("MountainCarContinuous-v0")

    state_dim  = env.observation_space.shape[0]   # 2: position, velocity
    action_dim = env.action_space.shape[0]         # 1: force [-1, 1]

    agent = A2CAgent(
        state_dim    = state_dim,
        action_dim   = action_dim,
        actor_lr     = 3e-4,
        critic_lr    = 1e-3,
        gamma        = 0.99,
        entropy_coef = 0.01,
    )

    rewards_log      = []
    actor_loss_log   = []
    critic_loss_log  = []
    entropy_log      = []
    solved_ep        = None

    print(f"Training A2C on MountainCarContinuous-v0  |  device: {agent.device}")
    print(f"State dim: {state_dim}  |  Action dim: {action_dim}  |  Rollout: {ROLLOUT_LEN} steps")
    print(f"{'Episode':>8} {'Reward':>10} {'Actor L':>10} {'Critic L':>10} {'Entropy':>9}")
    print("-" * 56)

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done  = False
        total_reward = 0

        # Trajectory buffers
        states, actions, rewards, dones = [], [], [], []

        while not done:
            action, _ = agent.select_action(state)
            action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))

            state = next_state
            total_reward += reward

            # Update every ROLLOUT_LEN steps or at episode end
            if len(states) >= ROLLOUT_LEN or done:
                al, cl, ent = agent.update(states, actions, rewards, dones, next_state)
                actor_loss_log.append(al)
                critic_loss_log.append(cl)
                entropy_log.append(ent)
                states, actions, rewards, dones = [], [], [], []

        rewards_log.append(total_reward)
        avg50 = np.mean(rewards_log[-50:]) if len(rewards_log) >= 50 else np.mean(rewards_log)

        if avg50 >= 90 and solved_ep is None:
            solved_ep = ep
            print(f"  *** SOLVED at episode {ep} (avg50 = {avg50:.1f}) ***")

        if ep % 100 == 0:
            al_avg = np.mean(actor_loss_log[-50:]) if actor_loss_log else 0
            cl_avg = np.mean(critic_loss_log[-50:]) if critic_loss_log else 0
            ent_avg = np.mean(entropy_log[-50:]) if entropy_log else 0
            print(f"{ep:>8} {total_reward:>10.1f} {al_avg:>10.4f} {cl_avg:>10.4f} {ent_avg:>9.4f}")

    env.close()
    agent.save("a2c.pth")
    return agent, rewards_log, actor_loss_log, critic_loss_log, entropy_log, solved_ep


# ─────────────────────────────────────────────
# Plot 1 — Training curves
# ─────────────────────────────────────────────

def plot_training_curves(rewards_log, actor_loss_log, critic_loss_log, entropy_log, solved_ep):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("MountainCar — A2C Training", fontsize=13, fontweight='bold')
    axes = axes.flatten()

    window   = 30
    episodes = np.arange(1, len(rewards_log) + 1)

    # Reward
    axes[0].plot(episodes, rewards_log, alpha=0.2, color='#534AB7', linewidth=0.8)
    if len(rewards_log) >= window:
        smooth = np.convolve(rewards_log, np.ones(window)/window, mode='valid')
        axes[0].plot(np.arange(window, len(rewards_log)+1), smooth,
                     color='#534AB7', linewidth=2, label=f'Avg({window})')
    axes[0].axhline(90, linestyle='--', color='#1D9E75', linewidth=1.2,
                    alpha=0.8, label='Solved (90)')
    if solved_ep:
        axes[0].axvline(solved_ep, linestyle=':', color='#D85A30',
                        linewidth=1.5, label=f'Solved ep {solved_ep}')
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend(fontsize=8)

    # Actor loss
    updates = np.arange(1, len(actor_loss_log)+1)
    axes[1].plot(updates, actor_loss_log, alpha=0.2, color='#D85A30', linewidth=0.6)
    if len(actor_loss_log) >= 20:
        smooth = np.convolve(actor_loss_log, np.ones(20)/20, mode='valid')
        axes[1].plot(np.arange(20, len(actor_loss_log)+1), smooth,
                     color='#D85A30', linewidth=2)
    axes[1].set_title("Actor loss (policy gradient)")
    axes[1].set_xlabel("Update step")
    axes[1].set_ylabel("Loss")

    # Critic loss
    axes[2].plot(updates, critic_loss_log, alpha=0.2, color='#1D9E75', linewidth=0.6)
    if len(critic_loss_log) >= 20:
        smooth = np.convolve(critic_loss_log, np.ones(20)/20, mode='valid')
        axes[2].plot(np.arange(20, len(critic_loss_log)+1), smooth,
                     color='#1D9E75', linewidth=2)
    axes[2].set_title("Critic loss (value MSE)")
    axes[2].set_xlabel("Update step")
    axes[2].set_ylabel("Loss")

    # Entropy
    axes[3].plot(updates, entropy_log, alpha=0.2, color='#888780', linewidth=0.6)
    if len(entropy_log) >= 20:
        smooth = np.convolve(entropy_log, np.ones(20)/20, mode='valid')
        axes[3].plot(np.arange(20, len(entropy_log)+1), smooth,
                     color='#323238', linewidth=2)
    axes[3].set_title("Policy entropy (exploration)")
    axes[3].set_xlabel("Update step")
    axes[3].set_ylabel("Entropy")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → training_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 2 — Actor vs Critic loss dual axis
# ─────────────────────────────────────────────

def plot_dual_loss(actor_loss_log, critic_loss_log):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    fig.suptitle("Actor Loss vs Critic Loss — Dual Axis", fontsize=13, fontweight='bold')

    window  = 20
    updates = np.arange(1, len(actor_loss_log)+1)

    # Actor loss on left axis
    color1 = '#534AB7'
    ax1.set_xlabel("Update step")
    ax1.set_ylabel("Actor loss", color=color1)
    ax1.plot(updates, actor_loss_log, alpha=0.15, color=color1, linewidth=0.6)
    if len(actor_loss_log) >= window:
        smooth = np.convolve(actor_loss_log, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window, len(actor_loss_log)+1), smooth,
                 color=color1, linewidth=2.5, label='Actor loss')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Critic loss on right axis
    ax2 = ax1.twinx()
    color2 = '#D85A30'
    ax2.set_ylabel("Critic loss", color=color2)
    ax2.plot(updates, critic_loss_log, alpha=0.15, color=color2, linewidth=0.6)
    if len(critic_loss_log) >= window:
        smooth = np.convolve(critic_loss_log, np.ones(window)/window, mode='valid')
        ax2.plot(np.arange(window, len(critic_loss_log)+1), smooth,
                 color=color2, linewidth=2.5, label='Critic loss', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig("dual_loss.png", dpi=150, bbox_inches='tight')
    print("Saved → dual_loss.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 3 — Value function V(s) over state space
# ─────────────────────────────────────────────

def plot_value_surface(agent):
    """Show what the critic thinks each (position, velocity) state is worth."""
    positions  = np.linspace(-1.2, 0.6, 60)
    velocities = np.linspace(-0.07, 0.07, 60)
    P, V_grid  = np.meshgrid(positions, velocities)

    states = np.column_stack([P.ravel(), V_grid.ravel()])
    states_t = torch.FloatTensor(states).to(agent.device)

    with torch.no_grad():
        values = agent.critic(states_t).cpu().numpy().reshape(P.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Critic Value Function V(s) — MountainCar State Space",
                 fontsize=13, fontweight='bold')

    # Heatmap
    im = axes[0].contourf(P, V_grid, values, levels=30, cmap='RdYlGn')
    plt.colorbar(im, ax=axes[0], label='V(s)')
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Velocity")
    axes[0].set_title("Value heatmap")
    axes[0].axvline(0.45, linestyle='--', color='white', linewidth=1.5,
                    alpha=0.8, label='Goal (pos=0.45)')
    axes[0].legend(fontsize=8)

    # 3D surface
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = fig.add_subplot(1, 2, 2, projection='3d', label='3d')
    axes[1].set_visible(False)
    surf = ax3d.plot_surface(P, V_grid, values, cmap='RdYlGn', alpha=0.9)
    ax3d.set_xlabel("Position", fontsize=8)
    ax3d.set_ylabel("Velocity", fontsize=8)
    ax3d.set_zlabel("V(s)", fontsize=8)
    ax3d.set_title("Value surface (3D)", fontsize=11)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)

    plt.tight_layout()
    plt.savefig("value_surface.png", dpi=150, bbox_inches='tight')
    print("Saved → value_surface.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 4 — Action distribution evolution
# ─────────────────────────────────────────────

def plot_action_distribution(agent):
    """Show how the actor's action distribution changes across states."""
    positions  = np.linspace(-1.2, 0.6, 100)
    velocities = [0.0, 0.03, -0.03]
    vel_labels = ['zero vel', 'positive vel (+)', 'negative vel (-)']
    colors     = ['#534AB7', '#1D9E75', '#D85A30']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Actor Policy — Mean Action vs Position", fontsize=13, fontweight='bold')

    for vel, label, color in zip(velocities, vel_labels, colors):
        states = np.column_stack([positions, np.full_like(positions, vel)])
        states_t = torch.FloatTensor(states).to(agent.device)

        with torch.no_grad():
            means, stds = agent.actor(states_t)
            means = means.cpu().numpy().squeeze()
            stds  = stds.cpu().numpy().squeeze()

        axes[0].plot(positions, means, color=color, linewidth=2, label=label)
        axes[0].fill_between(positions, means - stds, means + stds,
                             alpha=0.15, color=color)

    axes[0].axhline(0, linestyle='--', color='#AAA', linewidth=0.8)
    axes[0].axvline(0.45, linestyle=':', color='#D85A30', linewidth=1.2,
                    alpha=0.7, label='Goal position')
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Mean action (force)")
    axes[0].set_title("Mean ± std of action distribution")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-1.5, 1.5)

    # Sample actions at a specific state
    test_state = np.array([-0.5, 0.0])
    states_t   = torch.FloatTensor(test_state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        dist = agent.actor.get_distribution(states_t)
        samples = dist.sample((1000,)).squeeze().cpu().numpy()

    axes[1].hist(np.clip(samples, -1, 1), bins=40, color='#534AB7',
                 edgecolor='white', alpha=0.85)
    axes[1].axvline(0, linestyle='--', color='#D85A30', linewidth=1.5)
    axes[1].set_xlabel("Action (force)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Action distribution at state (-0.5, 0.0)")

    plt.tight_layout()
    plt.savefig("action_distribution.png", dpi=150, bbox_inches='tight')
    print("Saved → action_distribution.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent, rewards_log, actor_loss_log, critic_loss_log, entropy_log, solved_ep = train(n_episodes=800)
    plot_training_curves(rewards_log, actor_loss_log, critic_loss_log, entropy_log, solved_ep)
    plot_dual_loss(actor_loss_log, critic_loss_log)
    plot_value_surface(agent)
    plot_action_distribution(agent)
    print("\nDone! Check training_curves.png, dual_loss.png, value_surface.png, action_distribution.png")
