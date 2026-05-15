import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tools import N_TOOLS


class PolicyNetwork(nn.Module):
    """
    Policy network for discrete tool selection.
    Input:  observation vector (task + history + result signal)
    Output: logits over N_TOOLS actions
    """

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

        # Init
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x):
        features = self.net(x)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action(self, obs, device):
        x      = torch.FloatTensor(obs).unsqueeze(0).to(device)
        logits, value = self.forward(x)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate(self, obs, actions):
        logits, values = self.forward(obs)
        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy, values


class ToolPPOAgent:
    """PPO agent for discrete tool-selection environment."""

    def __init__(
        self,
        obs_dim,
        n_actions,
        lr            = 3e-4,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_eps      = 0.2,
        epochs        = 4,
        batch_size    = 32,
        value_coef    = 0.5,
        entropy_coef  = 0.05,
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

        self.net       = PolicyNetwork(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.actor_losses   = []
        self.critic_losses  = []
        self.entropies      = []
        self.clip_fractions = []

    def select_action(self, obs):
        return self.net.get_action(obs, self.device)

    def compute_gae(self, rewards, values, dones, last_value):
        gae = 0.0
        advantages = []
        values_ext = values + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t+1] * (1-dones[t]) - values_ext[t]
            gae   = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns    = advantages + torch.FloatTensor(values).to(self.device)
        return advantages, returns

    def update(self, rollout):
        states     = torch.FloatTensor(np.array(rollout["states"])).to(self.device)
        actions    = torch.LongTensor(rollout["actions"]).to(self.device)
        old_lp     = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages = rollout["advantages"]
        returns    = rollout["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                b = idx[start:start+self.batch_size]
                new_lp, entropy, values = self.net.evaluate(states[b], actions[b])

                ratio         = (new_lp - old_lp[b]).exp()
                clipped       = ratio.clamp(1-self.clip_eps, 1+self.clip_eps)
                actor_loss    = -torch.min(ratio * advantages[b], clipped * advantages[b]).mean()
                critic_loss   = F.mse_loss(values, returns[b])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                cf = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.entropies.append(entropy.mean().item())
                self.clip_fractions.append(cf)

    def save(self, path="tool_ppo.pth"):
        torch.save(self.net.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path="tool_ppo.pth"):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded ← {path}")
