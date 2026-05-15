import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from reward_model import (
    RewardModel, VOCAB_SIZE, PREFERENCE_PAIRS,
    tokenize, train_reward_model, bradley_terry_loss,
    PreferenceDataset, VOCAB
)
from trainer import (
    TextGenerator, GEN_VOCAB_SIZE, rlhf_train, idx2gen
)


# ─────────────────────────────────────────────
# Plot 1 — Reward model training
# ─────────────────────────────────────────────

def plot_reward_model_training(losses, accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Step 1: Reward Model Training on Human Preferences",
                 fontsize=13, fontweight='bold')

    epochs = np.arange(1, len(losses)+1)
    axes[0].plot(epochs, losses, color='#D85A30', linewidth=2)
    axes[0].set_title("Bradley-Terry loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(0)

    axes[1].plot(epochs, accs, color='#1D9E75', linewidth=2)
    axes[1].axhline(50, linestyle='--', color='#AAA', linewidth=1, label='Random (50%)')
    axes[1].axhline(100, linestyle='--', color='#534AB7', alpha=0.4, linewidth=1, label='Perfect (100%)')
    axes[1].set_title("Preference prediction accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 110)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("reward_model_training.png", dpi=150, bbox_inches='tight')
    print("Saved → reward_model_training.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 2 — Reward model scores on example texts
# ─────────────────────────────────────────────

def plot_reward_scores(reward_model, device="cpu"):
    positive_texts = [
        "this is excellent and i really loved it",
        "amazing quality wonderful experience overall",
        "i enjoyed this very helpful and clear",
        "outstanding brilliant and impressive work",
        "perfect product i highly recommend it",
    ]
    negative_texts = [
        "this is terrible and i hated it",
        "awful quality horrible experience overall",
        "i hated this very confusing and useless",
        "disappointing boring and mediocre work",
        "poor product waste of money do not buy",
    ]

    def score(texts):
        scores = []
        for t in texts:
            ids = torch.LongTensor(tokenize(t, 20)).unsqueeze(0).to(device)
            with torch.no_grad():
                s = reward_model(ids).item()
            scores.append(s)
        return scores

    pos_scores = score(positive_texts)
    neg_scores = score(negative_texts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step 1: Reward Model Scores on Example Texts",
                 fontsize=13, fontweight='bold')

    # Bar chart
    all_texts  = [t[:30]+'…' for t in positive_texts] + [t[:30]+'…' for t in negative_texts]
    all_scores = pos_scores + neg_scores
    colors     = ['#1D9E75'] * len(pos_scores) + ['#D85A30'] * len(neg_scores)

    bars = axes[0].barh(range(len(all_texts)), all_scores, color=colors, edgecolor='white')
    axes[0].set_yticks(range(len(all_texts)))
    axes[0].set_yticklabels(all_texts, fontsize=8)
    axes[0].axvline(0, color='#333', linewidth=0.8)
    axes[0].set_xlabel("Reward score")
    axes[0].set_title("Positive vs negative texts")
    legend = [
        mpatches.Patch(color='#1D9E75', label='Positive sentiment'),
        mpatches.Patch(color='#D85A30', label='Negative sentiment'),
    ]
    axes[0].legend(handles=legend, fontsize=8)

    # Distribution
    axes[1].hist(pos_scores, bins=8, color='#1D9E75', alpha=0.7,
                 label=f'Positive (avg={np.mean(pos_scores):.2f})', density=False)
    axes[1].hist(neg_scores, bins=8, color='#D85A30', alpha=0.7,
                 label=f'Negative (avg={np.mean(neg_scores):.2f})', density=False)
    axes[1].axvline(0, color='#333', linewidth=1, linestyle='--')
    axes[1].set_xlabel("Reward score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Score distributions")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("reward_scores.png", dpi=150, bbox_inches='tight')
    print("Saved → reward_scores.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 3 — RLHF fine-tuning curves
# ─────────────────────────────────────────────

def plot_rlhf_training(rewards_log, kl_log):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Step 2: PPO Fine-Tuning with Reward Model Signal",
                 fontsize=13, fontweight='bold')

    window   = 40
    episodes = np.arange(1, len(rewards_log)+1)

    # Reward score over training
    axes[0].plot(episodes, rewards_log, alpha=0.2, color='#534AB7', linewidth=0.8)
    if len(rewards_log) >= window:
        smooth = np.convolve(rewards_log, np.ones(window)/window, mode='valid')
        axes[0].plot(np.arange(window, len(rewards_log)+1), smooth,
                     color='#534AB7', linewidth=2)
    axes[0].axhline(0, linestyle='--', color='#AAA', linewidth=0.8)
    axes[0].set_title("Reward model score over training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward score")

    # KL penalty
    axes[1].plot(episodes, kl_log, alpha=0.2, color='#D85A30', linewidth=0.8)
    if len(kl_log) >= window:
        smooth = np.convolve(kl_log, np.ones(window)/window, mode='valid')
        axes[1].plot(np.arange(window, len(kl_log)+1), smooth,
                     color='#D85A30', linewidth=2)
    axes[1].set_title("KL penalty (divergence from reference)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("KL penalty")

    # Reward distribution — early vs late
    n = len(rewards_log)
    early = rewards_log[:n//4]
    late  = rewards_log[-n//4:]
    axes[2].hist(early, bins=20, color='#D85A30', alpha=0.65,
                 label=f'Early (avg={np.mean(early):.2f})', density=True)
    axes[2].hist(late,  bins=20, color='#1D9E75', alpha=0.65,
                 label=f'Late  (avg={np.mean(late):.2f})', density=True)
    axes[2].set_title("Reward distribution — early vs late")
    axes[2].set_xlabel("Reward score")
    axes[2].set_ylabel("Density")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("rlhf_training.png", dpi=150, bbox_inches='tight')
    print("Saved → rlhf_training.png")
    plt.show()


# ─────────────────────────────────────────────
# Plot 4 — Before vs after: generated text quality
# ─────────────────────────────────────────────

def plot_before_after(generator, reward_model, device="cpu", n_samples=12):
    """Compare reward scores of text generated before and after fine-tuning."""

    # Before: random (untrained) generator
    random_gen = TextGenerator(GEN_VOCAB_SIZE).to(device)
    random_gen.eval()

    def sample_scores(gen, n, temperature=1.2):
        scores = []
        texts  = []
        for _ in range(n):
            with torch.no_grad():
                tokens, _ = gen.generate_with_grad(seq_len=10, temperature=temperature, device=device)
            text = gen.tokens_to_text(tokens)
            ids  = torch.LongTensor(tokenize(text, 20)).unsqueeze(0).to(device)
            with torch.no_grad():
                s = reward_model(ids).item()
            scores.append(s)
            texts.append(text)
        return scores, texts

    generator.eval()
    before_scores, before_texts = sample_scores(random_gen, n_samples)
    after_scores,  after_texts  = sample_scores(generator,   n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step 2: Before vs After RLHF Fine-Tuning",
                 fontsize=13, fontweight='bold')

    # Score comparison
    x = np.arange(n_samples)
    w = 0.35
    axes[0].bar(x - w/2, before_scores, w, label='Before RLHF', color='#D85A30',
                alpha=0.8, edgecolor='white')
    axes[0].bar(x + w/2, after_scores,  w, label='After RLHF',  color='#1D9E75',
                alpha=0.8, edgecolor='white')
    axes[0].axhline(0, color='#333', linewidth=0.8, linestyle='--')
    axes[0].set_title(f"Reward scores — before vs after\n"
                      f"Before avg: {np.mean(before_scores):.2f} | After avg: {np.mean(after_scores):.2f}")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Reward score")
    axes[0].legend(fontsize=9)

    # Distribution shift
    axes[1].hist(before_scores, bins=12, color='#D85A30', alpha=0.65,
                 label=f'Before (avg={np.mean(before_scores):.2f})', density=True)
    axes[1].hist(after_scores,  bins=12, color='#1D9E75', alpha=0.65,
                 label=f'After  (avg={np.mean(after_scores):.2f})', density=True)
    axes[1].axvline(0, linestyle='--', color='#333', linewidth=0.8)
    axes[1].set_title("Score distribution shift")
    axes[1].set_xlabel("Reward score")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("before_after.png", dpi=150, bbox_inches='tight')
    print("Saved → before_after.png")
    plt.show()

    print(f"\nBefore RLHF sample texts:")
    for t in before_texts[:4]: print(f"  {t}")
    print(f"\nAfter RLHF sample texts:")
    for t in after_texts[:4]:  print(f"  {t}")
    print(f"\nReward improvement: {np.mean(before_scores):.3f} → {np.mean(after_scores):.3f} "
          f"(+{np.mean(after_scores)-np.mean(before_scores):.3f})")


# ─────────────────────────────────────────────
# Plot 5 — Full RLHF pipeline diagram
# ─────────────────────────────────────────────

def plot_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#F5F5F2')
    ax.set_facecolor('#F5F5F2')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("RLHF Pipeline — From Human Preferences to Fine-Tuned Policy",
                 fontsize=12, fontweight='bold', pad=12)

    boxes = [
        (1.2,  2.5, "Human\nPreference\nPairs",    "#F0997B", "30 labeled\npairs"),
        (4.0,  2.5, "Reward\nModel",                "#534AB7", "Bradley-Terry\nloss"),
        (7.0,  2.5, "PPO\nFine-Tuning",             "#D85A30", "REINFORCE +\nKL penalty"),
        (10.2, 2.5, "Fine-Tuned\nGenerator",        "#1D9E75", "Higher reward\nscores"),
    ]

    for x, y, title, color, subtitle in boxes:
        rect = mpatches.FancyBboxPatch((x-1.1, y-1.0), 2.2, 2.0,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, alpha=0.15,
                                   edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.3, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#222')
        ax.text(x, y-0.5, subtitle, ha='center', va='center',
                fontsize=8, color='#555')

    # Arrows
    arrow_props = dict(arrowstyle='->', color='#555', lw=1.8)
    for x1, x2 in [(2.3, 2.9), (5.1, 5.9), (8.1, 9.1)]:
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=arrow_props)

    # Step labels
    for x, label in [(2.5, 'Step 1:\nTrain'), (6.0, 'Step 2:\nFine-tune'), (9.1, 'Result')]:
        ax.text(x, 0.5, label, ha='center', va='center',
                fontsize=8, color='#777', style='italic')

    # KL feedback arrow
    ax.annotate('', xy=(7.0, 1.5), xytext=(10.2, 1.5),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2, linestyle='dashed'))
    ax.text(8.6, 1.1, 'KL penalty\n(stays close to ref)', ha='center',
            fontsize=7, color='#888')

    plt.tight_layout()
    plt.savefig("pipeline_diagram.png", dpi=150, bbox_inches='tight')
    print("Saved → pipeline_diagram.png")
    plt.show()


# ─────────────────────────────────────────────
# Main — full pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Train reward model ─────────────────────────────────────
    print("=" * 55)
    print("STEP 1 — Training Reward Model")
    print("=" * 55)
    reward_model, rm_losses, rm_accs = train_reward_model(n_epochs=80, device=device)
    plot_reward_model_training(rm_losses, rm_accs)
    plot_reward_scores(reward_model, device=device)

    # ── Step 2: PPO fine-tuning ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2 — PPO Fine-Tuning with Reward Signal")
    print("=" * 55)
    generator, reward_model, rewards_log, kl_log, gen_texts, reward_scores = rlhf_train(
        n_episodes  = 600,
        seq_len     = 10,
        lr          = 1e-3,
        kl_coef     = 0.05,
        temperature = 1.2,
        device      = device,
    )

    # ── Visualizations ─────────────────────────────────────────────────
    plot_rlhf_training(rewards_log, kl_log)
    plot_before_after(generator, reward_model, device=device, n_samples=12)
    plot_pipeline_diagram()

    print("\n" + "=" * 55)
    print("RLHF pipeline complete!")
    print("Outputs: reward_model_training.png, reward_scores.png,")
    print("         rlhf_training.png, before_after.png, pipeline_diagram.png")
    print("=" * 55)
