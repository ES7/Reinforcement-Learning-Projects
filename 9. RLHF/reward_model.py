"""
Reward Model for miniature RLHF pipeline.

Step 1 of RLHF:
  - Collect human preference pairs (text_a, text_b, preferred)
  - Train a reward model to predict which text humans prefer
  - The reward model becomes the reward signal for PPO fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# Vocabulary & Tokenizer
# ─────────────────────────────────────────────

VOCAB = [
    "<pad>", "<unk>",
    # positive sentiment words
    "excellent", "amazing", "wonderful", "fantastic", "brilliant", "outstanding",
    "superb", "great", "good", "nice", "love", "loved", "enjoy", "enjoyed",
    "perfect", "beautiful", "incredible", "delightful", "pleasant", "impressive",
    "helpful", "useful", "clear", "well", "best", "recommend", "happy", "positive",
    "quality", "awesome", "inspiring", "engaging", "fascinating", "creative",
    # negative sentiment words
    "terrible", "awful", "horrible", "bad", "poor", "worst", "hate", "hated",
    "boring", "useless", "confusing", "disappointing", "dreadful", "mediocre",
    "frustrating", "annoying", "waste", "difficult", "unclear", "negative",
    "broken", "failed", "slow", "ugly", "wrong", "errors", "problems", "issues",
    # neutral/filler words
    "the", "a", "an", "is", "it", "this", "was", "very", "quite", "really",
    "i", "me", "my", "we", "they", "book", "movie", "product", "course",
    "content", "writing", "explanation", "overall", "experience", "topic",
    "found", "think", "feel", "seems", "much", "more", "than", "but", "and",
    "not", "no", "so", "with", "of", "to", "in", "for", "on", "at", "has",
]

VOCAB_SIZE = len(VOCAB)
word2idx   = {w: i for i, w in enumerate(VOCAB)}

def tokenize(text: str, max_len: int = 20) -> list:
    tokens = text.lower().split()[:max_len]
    ids    = [word2idx.get(t, word2idx["<unk>"]) for t in tokens]
    # pad
    ids += [word2idx["<pad>"]] * (max_len - len(ids))
    return ids


# ─────────────────────────────────────────────
# Human preference dataset
# ─────────────────────────────────────────────

PREFERENCE_PAIRS = [
    # (text_a, text_b, label)  — label=1 means text_a preferred, 0 means text_b preferred
    ("This book is excellent and i really loved it",
     "This book was terrible and i hated reading it", 1),
    ("The course content is clear well explained and helpful",
     "The course content is confusing and useless", 1),
    ("Amazing product works perfectly i highly recommend",
     "Poor quality product broke after one day worst purchase", 1),
    ("I enjoyed this movie it was wonderful and inspiring",
     "I hated this movie it was boring and disappointing", 1),
    ("Outstanding explanation very impressive and engaging",
     "Awful explanation frustrating and unclear", 1),
    ("Great experience very pleasant and delightful overall",
     "Horrible experience very annoying and negative overall", 1),
    ("The writing is beautiful creative and fascinating",
     "The writing is mediocre dull and not engaging", 1),
    ("Brilliant work this is the best i have seen",
     "Poor work this is the worst i have seen", 1),
    ("Very useful and helpful content i feel happy",
     "Very useless and not helpful i feel disappointed", 1),
    ("Incredible quality superb performance i love it",
     "Terrible quality awful performance i hate it", 1),
    # Reversed pairs for balance
    ("This book was terrible and i hated reading it",
     "This book is excellent and i really loved it", 0),
    ("The course content is confusing and useless",
     "The course content is clear well explained and helpful", 0),
    ("Poor quality product broke after one day worst purchase",
     "Amazing product works perfectly i highly recommend", 0),
    ("I hated this movie it was boring and disappointing",
     "I enjoyed this movie it was wonderful and inspiring", 0),
    ("Awful explanation frustrating and unclear",
     "Outstanding explanation very impressive and engaging", 0),
    ("Horrible experience very annoying and negative overall",
     "Great experience very pleasant and delightful overall", 0),
    ("The writing is mediocre dull and not engaging",
     "The writing is beautiful creative and fascinating", 0),
    ("Poor work this is the worst i have seen",
     "Brilliant work this is the best i have seen", 0),
    ("Very useless and not helpful i feel disappointed",
     "Very useful and helpful content i feel happy", 0),
    ("Terrible quality awful performance i hate it",
     "Incredible quality superb performance i love it", 0),
    # Harder pairs — closer in sentiment
    ("The product is good and works well for the price",
     "The product has some issues and problems but is okay", 1),
    ("I think this is quite nice and pleasant to use",
     "I think this is not great and has errors", 1),
    ("The explanation seems clear and i found it helpful",
     "The explanation seems wrong and i found it confusing", 1),
    ("Overall a positive experience i recommend this",
     "Overall a negative experience i would not recommend", 1),
    ("Really enjoyed this very engaging and well written",
     "Did not enjoy this very boring and poorly written", 1),
    # More reversed harder pairs
    ("The product has some issues and problems but is okay",
     "The product is good and works well for the price", 0),
    ("I think this is not great and has errors",
     "I think this is quite nice and pleasant to use", 0),
    ("The explanation seems wrong and i found it confusing",
     "The explanation seems clear and i found it helpful", 0),
    ("Overall a negative experience i would not recommend",
     "Overall a positive experience i recommend this", 0),
    ("Did not enjoy this very boring and poorly written",
     "Really enjoyed this very engaging and well written", 0),
]


class PreferenceDataset(Dataset):
    def __init__(self, pairs, max_len=20):
        self.pairs   = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.pairs[idx]
        tokens_a = torch.LongTensor(tokenize(text_a, self.max_len))
        tokens_b = torch.LongTensor(tokenize(text_b, self.max_len))
        return tokens_a, tokens_b, torch.FloatTensor([label])


# ─────────────────────────────────────────────
# Reward Model
# ─────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Predicts a scalar reward score for a piece of text.
    Trained on human preference pairs using Bradley-Terry loss.

    Architecture:
      token ids → embedding → mean pool → MLP → scalar score
    """

    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        emb  = self.embedding(token_ids)       # (batch, seq_len, emb_dim)
        mask = (token_ids != 0).float().unsqueeze(-1)
        emb  = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pool
        return self.mlp(emb).squeeze(-1)       # (batch,) scalar score


def bradley_terry_loss(score_a, score_b, label):
    """
    Bradley-Terry preference loss.
    P(a preferred) = sigmoid(score_a - score_b)
    Loss = -label * log P(a) - (1-label) * log P(b)
    """
    logits = score_a - score_b
    return F.binary_cross_entropy_with_logits(logits, label.squeeze(-1))


def train_reward_model(n_epochs=80, lr=1e-3, device="cpu"):
    dataset    = PreferenceDataset(PREFERENCE_PAIRS)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model     = RewardModel(VOCAB_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses    = []
    accs      = []

    print("Training Reward Model on human preference pairs...")
    print(f"{'Epoch':>7} {'Loss':>10} {'Accuracy':>10}")
    print("-" * 32)

    for epoch in range(1, n_epochs + 1):
        epoch_loss, correct, total = 0.0, 0, 0

        for tokens_a, tokens_b, label in dataloader:
            tokens_a = tokens_a.to(device)
            tokens_b = tokens_b.to(device)
            label    = label.to(device)

            score_a = model(tokens_a)
            score_b = model(tokens_b)

            loss = bradley_terry_loss(score_a, score_b, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # Accuracy: did we predict the right preference?
            pred    = (score_a > score_b).float()
            correct += (pred == label.squeeze(-1)).sum().item()
            total   += len(label)

        avg_loss = epoch_loss / len(dataloader)
        acc      = correct / total * 100
        losses.append(avg_loss)
        accs.append(acc)

        if epoch % 10 == 0:
            print(f"{epoch:>7} {avg_loss:>10.4f} {acc:>9.1f}%")

    torch.save(model.state_dict(), "reward_model.pth")
    print(f"\nReward model saved → reward_model.pth")
    return model, losses, accs
