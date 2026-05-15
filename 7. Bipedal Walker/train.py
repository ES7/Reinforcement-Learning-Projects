import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import torch
from agent import PPOAgent


ROLLOUT_LEN = 2048   # steps per PPO update — standard for BipedalWalker


# ─────────────────────────────────────────────
# Collect one rollout
# ─────────────────────────────────────────────

def collect_rollout(env, agent, rollout_len):
    states, actions, rewards, dones = [], [], [], []
    log_probs, values = [], []

    state, _ = env.reset()
    ep_rewards = []
    ep_total   = 0
    episodes_done = 0

    for _ in range(rollout_len):
        action, log_prob, value = agent.select_action(state)
        action_c = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(action_c)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value)

        ep_total += reward
        state = next_state

        if done:
            ep_rewards.append(ep_total)
            ep_total = 0
            episodes_done += 1
            state, _ = env.reset()

    # Bootstrap last value
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        _, last_value = agent.net.get_dist(s)
        last_value = last_value.item()

    advantages, returns = agent.compute_gae(rewards, values, dones, last_value)

    rollout = {
        "states":     states,
        "actions":    actions,
        "log_probs":  log_probs,
        "values":     values,
        "advantages": advantages,
        "returns":    returns,
    }
    return rollout, ep_rewards


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(n_updates=200):
    env = gym.make("BipedalWalker-v3")

    state_dim  = env.observation_space.shape[0]   # 24
    action_dim = env.action_space.shape[0]         # 4 (hip/knee joints x2)

    agent = PPOAgent(
        state_dim     = state_dim,
        action_dim    = action_dim,
        lr            = 3e-4,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_eps      = 0.2,
        epochs        = 10,
        batch_size    = 64,
        value_coef    = 0.5,
        entropy_coef  = 0.01,
        max_grad_norm = 0.5,
    )

    all_ep_rewards = []
    update_rewards = []
    solved_update  = None

    print(f"Training PPO on BipedalWalker-v3  |  device: {agent.device}")
    print(f"State: {state_dim}  |  Actions: {action_dim}  |  Rollout: {ROLLOUT_LEN} steps")
    print(f"{'Update':>8} {'Avg Reward':>12} {'Clip Frac':>11} {'Approx KL':>11}")
    print("-" * 50)

    for update in range(1, n_updates + 1):
        rollout, ep_rewards = collect_rollout(env, agent, ROLLOUT_LEN)
        agent.update(rollout)

        if ep_rewards:
            all_ep_rewards.extend(ep_rewards)
            avg = np.mean(ep_rewards)
            update_rewards.append(avg)

            avg50 = np.mean(all_ep_rewards[-50:]) if len(all_ep_rewards) >= 50 else np.mean(all_ep_rewards)
            cf    = np.mean(agent.clip_fractions[-100:]) if agent.clip_fractions else 0
            kl    = np.mean(agent.approx_kl_divs[-100:]) if agent.approx_kl_divs else 0

            if avg50 >= 300 and solved_update is None:
                solved_update = update
                print(f"  *** SOLVED at update {update} (avg50={avg50:.1f}) ***")

            if update % 20 == 0:
                print(f"{update:>8} {avg:>12.1f} {cf:>11.3f} {kl:>11.4f}")

    env.close()
    agent.save("ppo.pth")
    return agent, all_ep_rewards, update_rewards, solved_update


# ─────────────────────────────────────────────
# Plot 1 — Training curves
# ─────────────────────────────────────────────

def plot_training_curves(all_ep_rewards, solved_update=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("BipedalWalker — PPO Training", fontsize=13, fontweight='bold')

    window   = 20
    episodes = np.arange(1, len(all_ep_rewards) + 1)

    axes[0].plot(episodes, all_ep_rewards, alpha=0.2, color='#534AB7', linewidth=0.8)
    if len(all_ep_rewards) >= window:
        smooth = np.convolve(all_ep_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(np.arange(window, len(all_ep_rewards)+1), smooth,
                     color='#534AB7', linewidth=2, label=f'Avg({window})')
    axes[0].axhline(300, linestyle='--', color='#1D9E75', linewidth=1.2, alpha=0.8, label='Solved (300)')
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend(fontsize=8)

    # Reward distribution — early vs late
    if len(all_ep_rewards) >= 100:
        early = all_ep_rewards[:len(all_ep_rewards)//4]
        late  = all_ep_rewards[-len(all_ep_rewards)//4:]
        axes[1].hist(early, bins=25, color='#D85A30', alpha=0.6, label='Early training', density=True)
        axes[1].hist(late,  bins=25, color='#534AB7', alpha=0.6, label='Late training',  density=True)
        axes[1].set_title("Reward distribution — early vs late")
        axes[1].set_xlabel("Episode reward")
        axes[1].set_ylabel("Density")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → training_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 2 — THE clip ratio visualization
# ─────────────────────────────────────────────

def plot_clip_ratio(agent):
    """
    This is the defining PPO visualization.
    Shows clipped vs unclipped surrogate objective and the clip fraction over training.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("PPO Clip Ratio — The Core Mechanism", fontsize=13, fontweight='bold')

    # ── Left: what clipping looks like mathematically ─────────────────
    advantages = np.linspace(-2, 2, 200)
    ratios_pos = np.linspace(0.5, 1.8, 200)   # varying ratio, positive advantage
    clip_eps   = 0.2

    unclipped = ratios_pos * 1.0               # advantage = 1 (positive)
    clipped   = np.clip(ratios_pos, 1 - clip_eps, 1 + clip_eps) * 1.0
    ppo_obj   = np.minimum(unclipped, clipped)

    axes[0].plot(ratios_pos, unclipped, color='#D85A30', linewidth=2,
                 linestyle='--', label='Unclipped r·A')
    axes[0].plot(ratios_pos, clipped,   color='#534AB7', linewidth=2,
                 linestyle='--', label='Clipped r·A')
    axes[0].plot(ratios_pos, ppo_obj,   color='#1D9E75', linewidth=2.5,
                 label='min(r·A, clip(r)·A) — PPO objective')
    axes[0].axvspan(1 - clip_eps, 1 + clip_eps, alpha=0.08, color='#534AB7', label='Safe zone [0.8, 1.2]')
    axes[0].axvline(1.0, linestyle=':', color='#AAA', linewidth=1)
    axes[0].set_xlabel("Probability ratio r = π_new / π_old")
    axes[0].set_ylabel("Objective value")
    axes[0].set_title("Clip mechanism (positive advantage)")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, 2.2)

    # ── Middle: clip fraction over training updates ────────────────────
    cf = agent.clip_fractions
    if cf:
        x  = np.arange(1, len(cf) + 1)
        w  = max(1, len(cf) // 50)
        axes[1].plot(x, cf, alpha=0.2, color='#534AB7', linewidth=0.6)
        if len(cf) >= w:
            smooth = np.convolve(cf, np.ones(w)/w, mode='valid')
            axes[1].plot(np.arange(w, len(cf)+1), smooth, color='#534AB7', linewidth=2)
        axes[1].axhline(0.1, linestyle='--', color='#D85A30', linewidth=1.2,
                        alpha=0.7, label='10% clip target')
        axes[1].set_title("Clip fraction over training")
        axes[1].set_xlabel("Update step")
        axes[1].set_ylabel("Fraction of ratios clipped")
        axes[1].legend(fontsize=8)
        axes[1].set_ylim(0, 0.6)

    # ── Right: approx KL divergence ───────────────────────────────────
    kl = agent.approx_kl_divs
    if kl:
        x  = np.arange(1, len(kl) + 1)
        w  = max(1, len(kl) // 50)
        axes[2].plot(x, kl, alpha=0.2, color='#1D9E75', linewidth=0.6)
        if len(kl) >= w:
            smooth = np.convolve(kl, np.ones(w)/w, mode='valid')
            axes[2].plot(np.arange(w, len(kl)+1), smooth, color='#1D9E75', linewidth=2)
        axes[2].axhline(0.01, linestyle='--', color='#D85A30', linewidth=1.2,
                        alpha=0.7, label='KL target ~0.01')
        axes[2].set_title("Approx KL divergence")
        axes[2].set_xlabel("Update step")
        axes[2].set_ylabel("KL(π_old || π_new)")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("clip_ratio.png", dpi=150, bbox_inches='tight')
    print("Saved → clip_ratio.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 3 — Actor / Critic / Entropy losses
# ─────────────────────────────────────────────

def plot_losses(agent):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PPO Training Losses", fontsize=13, fontweight='bold')

    window = 50

    for ax, data, title, color in zip(
        axes,
        [agent.actor_losses, agent.critic_losses, agent.entropies],
        ['Actor loss (policy gradient)', 'Critic loss (value MSE)', 'Policy entropy'],
        ['#534AB7', '#D85A30', '#1D9E75']
    ):
        x = np.arange(1, len(data)+1)
        ax.plot(x, data, alpha=0.15, color=color, linewidth=0.6)
        if len(data) >= window:
            smooth = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window, len(data)+1), smooth, color=color, linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Update step")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig("losses.png", dpi=150, bbox_inches='tight')
    print("Saved → losses.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 4 — GAE vs raw returns comparison
# ─────────────────────────────────────────────

def plot_gae_illustration():
    """
    Show what GAE does — smooth advantage estimate vs noisy raw returns.
    Illustrative, not from a real episode (too noisy to plot cleanly).
    """
    np.random.seed(42)
    T = 80
    t = np.arange(T)

    # Simulated raw returns (high variance)
    raw_returns   = np.cumsum(np.random.randn(T) * 8)[::-1] + np.random.randn(T) * 15

    # GAE smoothed (λ=0.95 gives gentle decay)
    gae = np.zeros(T)
    gae[-1] = raw_returns[-1] * 0.1
    for i in range(T-2, -1, -1):
        gae[i] = raw_returns[i] * 0.3 + 0.95 * gae[i+1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("GAE vs Raw Returns — Variance Reduction", fontsize=13, fontweight='bold')

    axes[0].plot(t, raw_returns, color='#D85A30', linewidth=1.5, alpha=0.8, label='Raw returns')
    axes[0].plot(t, gae,         color='#534AB7', linewidth=2,   label='GAE (λ=0.95)')
    axes[0].set_title("Advantage estimates over a rollout")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Advantage")
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, linestyle='--', color='#AAA', linewidth=0.8)

    # Variance comparison
    axes[1].bar(['Raw returns', 'GAE (λ=0.95)'],
                [np.std(raw_returns), np.std(gae)],
                color=['#D85A30', '#534AB7'], edgecolor='white', width=0.4)
    axes[1].set_title("Std dev comparison (lower = less variance)")
    axes[1].set_ylabel("Standard deviation")
    for i, v in enumerate([np.std(raw_returns), np.std(gae)]):
        axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("gae_illustration.png", dpi=150, bbox_inches='tight')
    print("Saved → gae_illustration.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent, all_ep_rewards, update_rewards, solved_update = train(n_updates=200)
    plot_training_curves(all_ep_rewards, solved_update)
    plot_clip_ratio(agent)
    plot_losses(agent)
    plot_gae_illustration()
    print("\nDone! Check training_curves.png, clip_ratio.png, losses.png, gae_illustration.png")
