import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import torch
from tool_env import ToolUseEnv, TASKS, OBS_DIM
from ppo_agent import ToolPPOAgent
from tools import TOOL_NAMES, N_TOOLS, call_tool


ROLLOUT_LEN = 256


# ─────────────────────────────────────────────
# Rollout collection
# ─────────────────────────────────────────────

def collect_rollout(env, agent, rollout_len):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    ep_rewards, ep_infos = [], []
    ep_total = 0
    obs, _ = env.reset()

    for _ in range(rollout_len):
        action, lp, val = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(lp)
        values.append(val)
        ep_total += reward
        obs = next_obs

        if done:
            ep_rewards.append(ep_total)
            ep_infos.append(info)
            ep_total = 0
            obs, _ = env.reset()

    with torch.no_grad():
        s = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        _, last_val = agent.net(s)
        last_val = last_val.item()

    advantages, returns = agent.compute_gae(rewards, values, dones, last_val)

    return (
        {"states": states, "actions": actions, "log_probs": log_probs,
         "values": values, "advantages": advantages, "returns": returns},
        ep_rewards, ep_infos
    )


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(n_updates=150):
    env   = ToolUseEnv(max_steps=5)
    agent = ToolPPOAgent(
        obs_dim      = OBS_DIM,
        n_actions    = N_TOOLS,
        lr           = 3e-4,
        gamma        = 0.99,
        gae_lambda   = 0.95,
        clip_eps     = 0.2,
        epochs       = 4,
        batch_size   = 32,
        entropy_coef = 0.05,
    )

    all_rewards   = []
    all_successes = []
    tool_correct_log = []

    print(f"Training Tool-Use PPO Agent  |  device: {agent.device}")
    print(f"Tasks: {len(TASKS)}  |  Tools: {TOOL_NAMES}  |  Rollout: {ROLLOUT_LEN}")
    print(f"{'Update':>8} {'Avg Reward':>12} {'Success%':>10} {'ToolAcc%':>10}")
    print("-" * 48)

    for update in range(1, n_updates + 1):
        rollout, ep_rewards, ep_infos = collect_rollout(env, agent, ROLLOUT_LEN)
        agent.update(rollout)

        if ep_rewards:
            all_rewards.extend(ep_rewards)

        correct = [info["correct"] for info in ep_infos if info]
        if correct:
            all_successes.extend([1 if c else 0 for c in correct])
            tool_correct_log.append(np.mean(correct))

        if update % 15 == 0:
            avg_r  = np.mean(all_rewards[-50:]) if all_rewards else 0
            suc    = np.mean(all_successes[-100:]) * 100 if all_successes else 0
            tc     = np.mean(tool_correct_log[-10:]) * 100 if tool_correct_log else 0
            print(f"{update:>8} {avg_r:>12.3f} {suc:>9.1f}% {tc:>9.1f}%")

    env.close()
    agent.save("tool_ppo.pth")
    return agent, all_rewards, all_successes, tool_correct_log


# ─────────────────────────────────────────────
# Plot 1 — Training curves
# ─────────────────────────────────────────────

def plot_training(all_rewards, all_successes, tool_correct_log):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("LLM Tool-Use Agent — PPO Training", fontsize=13, fontweight='bold')

    window = 30

    # Reward
    ep = np.arange(1, len(all_rewards)+1)
    axes[0].plot(ep, all_rewards, alpha=0.2, color='#534AB7', linewidth=0.8)
    if len(all_rewards) >= window:
        s = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(np.arange(window, len(all_rewards)+1), s, color='#534AB7', linewidth=2)
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")

    # Success rate
    if all_successes:
        sr = np.convolve(all_successes, np.ones(window)/window, mode='valid') * 100
        axes[1].plot(np.arange(window, len(all_successes)+1), sr, color='#1D9E75', linewidth=2)
        axes[1].axhline(80, linestyle='--', color='#D85A30', alpha=0.7, linewidth=1.2, label='80% target')
        axes[1].set_title("Task success rate (%)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Success %")
        axes[1].set_ylim(0, 105)
        axes[1].legend(fontsize=8)

    # Tool accuracy
    if tool_correct_log:
        x = np.arange(1, len(tool_correct_log)+1)
        axes[2].plot(x, np.array(tool_correct_log)*100, color='#D85A30', linewidth=2)
        axes[2].set_title("Correct tool selection (%)")
        axes[2].set_xlabel("Update")
        axes[2].set_ylabel("Accuracy %")
        axes[2].set_ylim(0, 105)
        axes[2].axhline(33, linestyle='--', color='#AAA', linewidth=1, label='Random baseline (33%)')
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("Saved → training_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 2 — Tool selection heatmap
# ─────────────────────────────────────────────

def plot_tool_heatmap(agent):
    """For each task, show what tool the agent selects — correct tool highlighted."""
    env = ToolUseEnv()
    selections = []

    for i, (task_text, correct_tool, tool_input, answer_kw) in enumerate(TASKS):
        env.task_idx     = i
        env.tool_history = []
        env.last_result  = ""
        env.steps        = 0
        env.current_task = {
            "text": task_text, "correct_tool": correct_tool,
            "tool_input": tool_input, "answer_kw": answer_kw
        }
        obs = env._get_obs()

        s = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            logits, _ = agent.net(s)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        selections.append({
            "task":    task_text[:35] + "..." if len(task_text) > 35 else task_text,
            "correct": correct_tool,
            "probs":   probs,
            "chosen":  TOOL_NAMES[np.argmax(probs)],
        })

    n     = len(selections)
    data  = np.array([s["probs"] for s in selections])
    tasks_short = [s["task"] for s in selections]

    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.35)))
    fig.suptitle("Agent Tool Selection — Probability Heatmap", fontsize=12, fontweight='bold')

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Probability')

    ax.set_xticks(range(N_TOOLS))
    ax.set_xticklabels([t.capitalize() for t in TOOL_NAMES], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tasks_short, fontsize=7)

    # Mark correct tool and chosen tool
    for i, s in enumerate(selections):
        correct_idx = TOOL_NAMES.index(s["correct"])
        chosen_idx  = TOOL_NAMES.index(s["chosen"])

        # Correct tool: green border
        rect = plt.Rectangle((correct_idx - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor='#1D9E75', linewidth=2.5)
        ax.add_patch(rect)

        # Cell text
        for j in range(N_TOOLS):
            ax.text(j, i, f'{data[i,j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if data[i,j] > 0.6 else '#333')

    legend = [
        mpatches.Patch(edgecolor='#1D9E75', facecolor='none', linewidth=2, label='Correct tool'),
    ]
    ax.legend(handles=legend, fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig("tool_heatmap.png", dpi=150, bbox_inches='tight')
    print("Saved → tool_heatmap.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 3 — Per-tool accuracy breakdown
# ─────────────────────────────────────────────

def plot_tool_accuracy(agent):
    env = ToolUseEnv()
    tool_results = defaultdict(list)

    for i, (task_text, correct_tool, tool_input, answer_kw) in enumerate(TASKS):
        env.task_idx     = i
        env.tool_history = []
        env.last_result  = ""
        env.steps        = 0
        env.current_task = {
            "text": task_text, "correct_tool": correct_tool,
            "tool_input": tool_input, "answer_kw": answer_kw
        }
        obs = env._get_obs()

        s = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            logits, _ = agent.net(s)
        chosen = TOOL_NAMES[logits.argmax().item()]
        tool_results[correct_tool].append(1 if chosen == correct_tool else 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Tool Selection Accuracy Breakdown", fontsize=12, fontweight='bold')

    tools   = list(tool_results.keys())
    accs    = [np.mean(tool_results[t]) * 100 for t in tools]
    counts  = [len(tool_results[t]) for t in tools]
    colors  = ['#534AB7', '#D85A30', '#1D9E75']

    bars = axes[0].bar(tools, accs, color=colors[:len(tools)], edgecolor='white', width=0.45)
    axes[0].axhline(33, linestyle='--', color='#AAA', linewidth=1, label='Random (33%)')
    axes[0].set_title("Accuracy per tool type")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 110)
    axes[0].legend(fontsize=8)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, acc + 1,
                     f'{acc:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # Task count per tool
    axes[1].bar(tools, counts, color=colors[:len(tools)], edgecolor='white', width=0.45)
    axes[1].set_title("Number of tasks per tool")
    axes[1].set_ylabel("Count")
    for i, (t, c) in enumerate(zip(tools, counts)):
        axes[1].text(i, c + 0.1, str(c), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("tool_accuracy.png", dpi=150, bbox_inches='tight')
    print("Saved → tool_accuracy.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 4 — Agent demo: step through a task
# ─────────────────────────────────────────────

def demo_agent(agent, n_tasks=6):
    env = ToolUseEnv(max_steps=5)
    print("\n" + "="*65)
    print("AGENT DEMO — Watching the trained agent solve tasks")
    print("="*65)

    results = []
    for _ in range(n_tasks):
        obs, info = env.reset()
        done = False
        print(f"\nTask: {info['task']}")
        print(f"Correct tool: {env.current_task['correct_tool']}")

        while not done:
            s = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                logits, _ = agent.net(s)
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            action = np.argmax(probs)
            tool   = TOOL_NAMES[action]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            print(f"  → Chose: {tool:12s} | Correct: {'✓' if info['correct'] else '✗'} | "
                  f"Result: {info['result'][:50]}")

        results.append(info['correct'])
        print(f"  Episode: {'SOLVED ✓' if info['correct'] else 'FAILED ✗'}")

    print(f"\nDemo accuracy: {np.mean(results)*100:.0f}% ({sum(results)}/{n_tasks})")
    print("="*65)

    # Bar chart of demo
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ['#1D9E75' if r else '#D85A30' for r in results]
    ax.bar(range(1, n_tasks+1), [1]*n_tasks, color=colors, edgecolor='white', width=0.6)
    ax.set_title(f"Agent Demo Results ({sum(results)}/{n_tasks} solved)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Task")
    ax.set_yticks([])
    legend = [
        mpatches.Patch(color='#1D9E75', label='Solved'),
        mpatches.Patch(color='#D85A30', label='Failed'),
    ]
    ax.legend(handles=legend, fontsize=9)
    plt.tight_layout()
    plt.savefig("demo_results.png", dpi=150, bbox_inches='tight')
    print("Saved → demo_results.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent, all_rewards, all_successes, tool_correct_log = train(n_updates=150)
    plot_training(all_rewards, all_successes, tool_correct_log)
    plot_tool_heatmap(agent)
    plot_tool_accuracy(agent)
    demo_agent(agent, n_tasks=8)
    print("\nDone! Check training_curves.png, tool_heatmap.png, tool_accuracy.png, demo_results.png")
