import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from grid_env import GridWorldEnv
from agent import QLearningAgent


def run_episode(agent, env, max_steps=200):
    """Run one full greedy episode, collect frames."""
    obs, _ = env.reset()
    frames = []
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Store snapshot
        frames.append({
            "pos":    tuple(env.agent_pos),
            "path":   list(env.path),
            "reward": total_reward,
            "steps":  env.steps,
        })

    return frames


def animate(frames, env, save_gif=True):
    grid = env.grid_size
    walls = env.walls
    goal  = env.goal
    start = env.start

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#F5F5F2')
    ax.set_facecolor('#F5F5F2')
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(grid - 0.5, -0.5)
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.grid(True, color='#CCCCCC', linewidth=0.5)
    ax.set_title("Grid World — Trained Agent", fontsize=12, fontweight='bold', pad=10)

    # Static elements (walls, goal, start)
    for (r, c) in walls:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#323238', zorder=2))
    ax.add_patch(plt.Rectangle((goal[1]-0.5, goal[0]-0.5), 1, 1, color='#1D9E75', zorder=2))
    ax.text(goal[1], goal[0], 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=3)
    ax.add_patch(plt.Rectangle((start[1]-0.5, start[0]-0.5), 1, 1, color='#F0997B', zorder=2))
    ax.text(start[1], start[0], 'S', ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=3)

    # Dynamic: path trail + agent circle
    path_patches = []
    agent_circle = plt.Circle((0, 0), 0.3, color='#534AB7', zorder=5)
    ax.add_patch(agent_circle)

    info_text = ax.text(0.01, 0.99, '', transform=ax.transAxes,
                        va='top', fontsize=9, color='#323238',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    def update(frame_idx):
        f = frames[frame_idx]

        # Clear old path patches
        for p in path_patches:
            p.remove()
        path_patches.clear()

        # Draw path trail
        for (pr, pc) in f["path"][:-1]:
            if (pr, pc) not in walls and (pr, pc) != goal and (pr, pc) != start:
                patch = plt.Rectangle((pc-0.25, pr-0.25), 0.5, 0.5,
                                      color='#AFA9EC', alpha=0.5, zorder=1)
                ax.add_patch(patch)
                path_patches.append(patch)

        # Move agent
        ar, ac = f["pos"]
        agent_circle.center = (ac, ar)

        info_text.set_text(f'Step {f["steps"]}  |  Reward: {f["reward"]:.3f}')
        return [agent_circle, info_text] + path_patches

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=200, blit=False, repeat=False
    )

    if save_gif:
        ani.save("agent_run.gif", writer="pillow", fps=6)
        print("Saved → agent_run.gif")

    plt.tight_layout()
    plt.show()
    return ani


def compare_random_vs_trained(agent, n_runs=100):
    """Compare random policy vs trained agent over n_runs episodes."""
    env_r = GridWorldEnv()
    env_t = GridWorldEnv()

    random_steps, random_success = [], []
    trained_steps, trained_success = [], []

    # Random agent
    for _ in range(n_runs):
        env_r.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env_r.step(env_r.action_space.sample())
            done = term or trunc
        random_steps.append(env_r.steps)
        random_success.append(1 if tuple(env_r.agent_pos) == env_r.goal else 0)

    # Trained agent
    agent.epsilon = 0  # fully greedy
    for _ in range(n_runs):
        obs, _ = env_t.reset()
        done = False
        while not done:
            a = agent.select_action(obs)
            obs, _, term, trunc, _ = env_t.step(a)
            done = term or trunc
        trained_steps.append(env_t.steps)
        trained_success.append(1 if tuple(env_t.agent_pos) == env_t.goal else 0)

    print("\n" + "="*45)
    print(f"{'':20} {'Random':>10} {'Trained':>10}")
    print("="*45)
    print(f"{'Avg steps':20} {np.mean(random_steps):>10.1f} {np.mean(trained_steps):>10.1f}")
    print(f"{'Success rate':20} {np.mean(random_success)*100:>9.1f}% {np.mean(trained_success)*100:>9.1f}%")
    print(f"{'Best steps':20} {min(random_steps):>10} {min(trained_steps):>10}")
    print("="*45)

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Random Policy vs Trained Agent", fontsize=13, fontweight='bold')

    colors = ['#D3D1C7', '#534AB7']
    axes[0].bar(['Random', 'Trained'], [np.mean(random_steps), np.mean(trained_steps)],
                color=colors, edgecolor='white', width=0.5)
    axes[0].set_title("Avg steps to reach goal")
    axes[0].set_ylabel("Steps (lower = better)")

    axes[1].bar(['Random', 'Trained'], [np.mean(random_success)*100, np.mean(trained_success)*100],
                color=colors, edgecolor='white', width=0.5)
    axes[1].set_title("Success rate (%)")
    axes[1].set_ylabel("% episodes")
    axes[1].set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
    print("Saved → comparison.png")
    plt.show()


if __name__ == "__main__":
    # Load trained agent
    agent = QLearningAgent(n_states=64, n_actions=4)
    agent.load("q_table.npy")
    agent.epsilon = 0.0   # pure greedy

    env = GridWorldEnv()

    print("Running trained agent...")
    frames = run_episode(agent, env)
    print(f"Episode done in {len(frames)} steps | "
          f"{'Reached goal!' if tuple(env.agent_pos) == env.goal else 'Did not reach goal'}")

    animate(frames, env, save_gif=True)
    compare_random_vs_trained(agent, n_runs=200)
