import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer


# ─────────────────────────────────────────────
# Neural Network — the Q-function approximator
# ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Maps state → Q-values for all actions.
    Input:  state vector (8 floats for LunarLander)
    Output: Q(s, a) for each of the 4 actions
    """

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent with:
    - Online network  : trained every step
    - Target network  : frozen copy, updated every C steps
    - Experience replay buffer
    - Epsilon-greedy exploration
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr            = 1e-3,
        gamma         = 0.99,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
        batch_size    = 64,
        buffer_cap    = 50_000,
        target_update = 10,       # sync target net every N episodes
        device        = None,
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Online network — trained every step
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)

        # Target network — frozen, copied from online every target_update episodes
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_cap)

        # Logging
        self.losses      = []
        self.q_estimates = []

    # ------------------------------------------------------------------
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.online_net(s)
            self.q_estimates.append(q.max().item())   # log Q estimate
            return q.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def train_step(self):
        """One gradient update on a random mini-batch."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values from online network
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values from frozen target network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            td_target  = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — stabilises training
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        return loss.item()

    def sync_target(self):
        """Copy online network weights → target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="dqn.pth"):
        torch.save({
            "online":  self.online_net.state_dict(),
            "target":  self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path="dqn.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon = self.epsilon_min
        print(f"Model loaded ← {path}")
