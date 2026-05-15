import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# Shared Actor-Critic Network
# ─────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared trunk for actor and critic.
    Actor head  → mean + log_std of Gaussian (continuous actions)
    Critic head → scalar V(s)
    """

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor heads
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

        # Critic head
        self.value_head   = nn.Linear(hidden, 1)

        # Orthogonal init — standard for PPO
        self._init_weights()

    def _init_weights(self):
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        features = self.trunk(x)
        mean     = self.mean_head(features)
        log_std  = self.log_std_head(features).clamp(-2, 1)
        std      = log_std.exp()
        value    = self.value_head(features).squeeze(-1)
        return mean, std, value

    def get_dist(self, x):
        mean, std, value = self.forward(x)
        return torch.distributions.Normal(mean, std), value

    def evaluate(self, states, actions):
        """Compute log_probs, entropy, values for a batch — used in PPO update."""
        mean, std, values = self.forward(states)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy, values


# ─────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimization.

    Key idea: clip the probability ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
    to [1-ε, 1+ε] so policy updates can't be too large.

    Loss = -min(r * A,  clip(r, 1-ε, 1+ε) * A)
         + c1 * value_loss
         - c2 * entropy
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr            = 3e-4,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_eps      = 0.2,
        epochs        = 10,
        batch_size    = 64,
        value_coef    = 0.5,
        entropy_coef  = 0.01,
        max_grad_norm = 0.5,
        device        = None,
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net       = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # Logging — clip ratio tracking is the unique PPO visualization
        self.actor_losses    = []
        self.critic_losses   = []
        self.entropies       = []
        self.clip_fractions  = []   # fraction of ratios that got clipped
        self.approx_kl_divs  = []   # approximate KL divergence per update

    # ------------------------------------------------------------------
    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.net.get_dist(s)
            action      = dist.sample()
            log_prob    = dist.log_prob(action).sum(dim=-1)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    # ------------------------------------------------------------------
    def compute_gae(self, rewards, values, dones, last_value):
        """
        Generalized Advantage Estimation (GAE).
        Balances bias vs variance with λ parameter.
        δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        """
        advantages = []
        gae = 0.0
        values_ext = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t+1] * (1 - dones[t]) - values_ext[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns    = advantages + torch.FloatTensor(values).to(self.device)
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, rollout):
        """
        PPO update: run `epochs` passes over the collected rollout.
        """
        states     = torch.FloatTensor(np.array(rollout["states"])).to(self.device)
        actions    = torch.FloatTensor(np.array(rollout["actions"])).to(self.device)
        old_lp     = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages = rollout["advantages"]
        returns    = rollout["returns"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        indices = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]

                b_states     = states[idx]
                b_actions    = actions[idx]
                b_old_lp     = old_lp[idx]
                b_advantages = advantages[idx]
                b_returns    = returns[idx]

                new_lp, entropy, values = self.net.evaluate(b_states, b_actions)

                # ── PPO clip loss ─────────────────────────────────────
                ratio          = (new_lp - b_old_lp).exp()
                clipped_ratio  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)

                unclipped_obj  = ratio * b_advantages
                clipped_obj    = clipped_ratio * b_advantages
                actor_loss     = -torch.min(unclipped_obj, clipped_obj).mean()

                # Track clip fraction for visualization
                clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                self.clip_fractions.append(clip_frac)

                # Approx KL for early stopping / monitoring
                approx_kl = (b_old_lp - new_lp).mean().item()
                self.approx_kl_divs.append(approx_kl)

                # ── Value loss ────────────────────────────────────────
                critic_loss = F.mse_loss(values, b_returns)

                # ── Combined loss ─────────────────────────────────────
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.entropies.append(entropy.mean().item())

    def save(self, path="ppo.pth"):
        torch.save(self.net.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path="ppo.pth"):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded ← {path}")
