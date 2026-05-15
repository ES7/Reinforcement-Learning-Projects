"""
Step 2 of RLHF:
  - Take a pre-trained (or randomly initialised) text generator
  - Fine-tune it with PPO using the reward model as the reward signal
  - The generator learns to produce text the reward model scores highly
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from reward_model import RewardModel, tokenize, VOCAB, VOCAB_SIZE, word2idx


# ─────────────────────────────────────────────
# Text Generator (the policy)
# ─────────────────────────────────────────────

# Generation vocabulary — words the generator can produce
GEN_VOCAB = [
    "this", "is", "was", "i", "the", "a", "it", "very",
    "really", "quite", "product", "book", "course", "movie",
    "excellent", "amazing", "wonderful", "great", "good", "nice",
    "helpful", "clear", "perfect", "impressive", "beautiful",
    "terrible", "awful", "horrible", "bad", "poor", "boring",
    "loved", "enjoyed", "hated", "recommend", "experience",
    "quality", "content", "writing", "explanation", "overall",
]
GEN_VOCAB_SIZE = len(GEN_VOCAB)
idx2gen = {i: w for i, w in enumerate(GEN_VOCAB)}


class TextGenerator(nn.Module):
    """
    Autoregressive text generator.
    At each step, given the tokens generated so far, outputs logits over GEN_VOCAB.
    This is the POLICY in RLHF — it generates text token by token.
    """

    def __init__(self, vocab_size, emb_dim=32, hidden=64, max_len=12):
        super().__init__()
        self.max_len   = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn       = nn.GRU(emb_dim, hidden, batch_first=True)
        self.head      = nn.Linear(hidden, vocab_size)

    def forward(self, token_ids):
        emb     = self.embedding(token_ids)
        out, _  = self.rnn(emb)
        logits  = self.head(out)
        return logits

    def generate(self, seq_len=10, temperature=1.0, device="cpu"):
        """
        Generate a sequence of tokens autoregressively.
        Returns tokens, log_probs.
        """
        tokens    = []
        log_probs = []
        input_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
        hidden    = None

        with torch.no_grad():
            pass  # will use grad during PPO

        for _ in range(seq_len):
            emb       = self.embedding(input_ids)
            out, hidden = self.rnn(emb, hidden)
            logits    = self.head(out[:, -1, :]) / max(temperature, 1e-5)
            dist      = torch.distributions.Categorical(logits=logits)
            token     = dist.sample()
            log_prob  = dist.log_prob(token)
            tokens.append(token.item())
            log_probs.append(log_prob)
            input_ids = token.unsqueeze(0)

        return tokens, log_probs

    def generate_with_grad(self, seq_len=10, temperature=1.0, device="cpu"):
        """Generate with gradient tracking for PPO update."""
        tokens    = []
        log_probs = []
        input_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
        hidden    = None

        for _ in range(seq_len):
            emb         = self.embedding(input_ids)
            out, hidden = self.rnn(emb, hidden)
            logits      = self.head(out[:, -1, :]) / max(temperature, 1e-5)
            dist        = torch.distributions.Categorical(logits=logits)
            token       = dist.sample()
            log_prob    = dist.log_prob(token)
            tokens.append(token.item())
            log_probs.append(log_prob)
            input_ids   = token.unsqueeze(0).detach()

        return tokens, log_probs

    def tokens_to_text(self, tokens):
        return " ".join(idx2gen.get(t, "<unk>") for t in tokens)


# ─────────────────────────────────────────────
# KL penalty (reference policy divergence)
# ─────────────────────────────────────────────

def compute_kl_penalty(log_probs_new, log_probs_ref, kl_coef=0.1):
    """
    Penalise deviation from reference policy.
    Prevents the generator from collapsing to reward-hacking gibberish.
    KL(π || π_ref) ≈ log π - log π_ref
    """
    kl = torch.stack(log_probs_new) - torch.stack(log_probs_ref)
    return kl_coef * kl.mean()


# ─────────────────────────────────────────────
# RLHF PPO Training
# ─────────────────────────────────────────────

def rlhf_train(
    n_episodes    = 600,
    seq_len       = 10,
    lr            = 1e-3,
    kl_coef       = 0.05,
    temperature   = 1.2,
    device        = "cpu",
):
    # Load trained reward model
    reward_model = RewardModel(VOCAB_SIZE).to(device)
    reward_model.load_state_dict(torch.load("reward_model.pth", map_location=device))
    reward_model.eval()
    print("Reward model loaded.")

    # Generator — the policy to fine-tune
    generator = TextGenerator(GEN_VOCAB_SIZE).to(device)

    # Reference policy — frozen copy of initial generator
    ref_generator = TextGenerator(GEN_VOCAB_SIZE).to(device)
    ref_generator.load_state_dict(generator.state_dict())
    ref_generator.eval()
    print(f"Generator initialised. Fine-tuning with PPO + KL penalty...")

    optimizer = optim.Adam(generator.parameters(), lr=lr)

    rewards_log      = []
    kl_log           = []
    gen_texts_log    = []
    reward_scores    = []

    print(f"\n{'Episode':>8} {'Reward':>10} {'KL pen':>10} {'Avg(50)':>10}")
    print("-" * 45)

    for ep in range(1, n_episodes + 1):
        # ── Generate with current policy ──────────────────────────────
        tokens, log_probs_new = generator.generate_with_grad(
            seq_len=seq_len, temperature=temperature, device=device
        )

        # Reference log probs (frozen)
        with torch.no_grad():
            _, log_probs_ref = ref_generator.generate_with_grad(
                seq_len=seq_len, temperature=temperature, device=device
            )

        # ── Reward from reward model ───────────────────────────────────
        text     = generator.tokens_to_text(tokens)
        token_ids = torch.LongTensor(tokenize(text, max_len=20)).unsqueeze(0).to(device)

        with torch.no_grad():
            reward_score = reward_model(token_ids).item()

        reward_scores.append(reward_score)

        # ── KL penalty ────────────────────────────────────────────────
        kl_pen = compute_kl_penalty(log_probs_new, log_probs_ref, kl_coef)

        # ── PPO-style loss ────────────────────────────────────────────
        # Simple REINFORCE with KL: maximise reward, penalise KL
        log_prob_sum = torch.stack(log_probs_new).sum()
        loss = -(reward_score * log_prob_sum) + kl_pen

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
        optimizer.step()

        rewards_log.append(reward_score)
        kl_log.append(kl_pen.item())
        gen_texts_log.append(text)

        avg50 = np.mean(rewards_log[-50:])
        if ep % 100 == 0:
            print(f"{ep:>8} {reward_score:>10.4f} {kl_pen.item():>10.4f} {avg50:>10.4f}")
            print(f"         Sample: \"{text}\"")

    torch.save(generator.state_dict(), "generator.pth")
    print(f"\nGenerator saved → generator.pth")
    return generator, reward_model, rewards_log, kl_log, gen_texts_log, reward_scores
