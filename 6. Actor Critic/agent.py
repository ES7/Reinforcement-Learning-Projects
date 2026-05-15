import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# Actor Network — outputs action mean + std
# ─────────────────────────────────────────────

class Actor(nn.Module):
    """
    Policy network for continuous action space.
    Outputs mean and log_std of a Gaussian distribution.
    Action = sample from N(mean, std)
    """

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, x):
        features  = self.shared(x)
        mean      = self.mean_head(features)
        log_std   = self.log_std_head(features).clamp(-2, 2)
        std       = log_std.exp()
        return mean, std

    def get_distribution(self, x):
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)

    def select_action(self, state, device):
        s    = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self.get_distribution(s)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).cpu().numpy(), log_prob.item()


# ─────────────────────────────────────────────
# Critic Network — outputs state value V(s)
# ─────────────────────────────────────────────

class Critic(nn.Module):
    """
    Value network.
    Outputs V(s) — scalar estimate of expected return from state s.
    """

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
# A2C Agent
# ─────────────────────────────────────────────

class A2CAgent:
    """
    Advantage Actor-Critic (A2C).

    Actor  loss: -log_prob * advantage   (policy gradient)
    Critic loss: MSE(V(s), actual return)
    Entropy bonus: encourages exploration

    Advantage = actual_return - V(s)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr    = 3e-4,
        critic_lr   = 1e-3,
        gamma       = 0.99,
        entropy_coef = 0.01,
        device      = None,
    ):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.actor  = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Logging
        self.actor_losses  = []
        self.critic_losses = []
        self.entropies     = []

    # ------------------------------------------------------------------
    def select_action(self, state):
        return self.actor.select_action(state, self.device)

    def compute_returns(self, rewards, dones, last_value, normalize=True):
        """
        Compute discounted returns G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        Bootstraps with critic value at end of trajectory.
        """
        returns = []
        G = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        if normalize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, states, actions, rewards, dones, next_state):
        """
        One A2C update on a collected trajectory.
        """
        states_t  = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)

        # Bootstrap value from last state
        with torch.no_grad():
            last_value = self.critic(
                torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            ).item()

        returns = self.compute_returns(rewards, dones, last_value)

        # ── Critic update ──────────────────────────────────────────────
        values      = self.critic(states_t)
        critic_loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ── Actor update ───────────────────────────────────────────────
        with torch.no_grad():
            advantages = returns - self.critic(states_t)

        dist     = self.actor.get_distribution(states_t)
        log_prob = dist.log_prob(actions_t).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1).mean()

        actor_loss = -(log_prob * advantages).mean() - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropies.append(entropy.item())

        return actor_loss.item(), critic_loss.item(), entropy.item()

    def save(self, path="a2c.pth"):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
        print(f"Model saved → {path}")

    def load(self, path="a2c.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        print(f"Model loaded ← {path}")
