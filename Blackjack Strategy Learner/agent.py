import numpy as np
import gymnasium as gym
from collections import defaultdict


class MonteCarloAgent:
    """
    First-Visit Monte Carlo Control with Epsilon-Greedy policy.
    No model of the game needed — learns purely from experience.

    State: (player_sum, dealer_card, usable_ace)  → 32 * 11 * 2 = 704 states
    Action: 0 = stick, 1 = hit
    """

    def __init__(self, gamma=1.0, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9999):
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q(s, a) and count N(s, a) for incremental mean
        self.Q = defaultdict(lambda: np.zeros(2))
        self.N = defaultdict(lambda: np.zeros(2))

    # ------------------------------------------------------------------
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.Q[state]))

    def update(self, episode):
        """
        First-visit MC update.
        episode: list of (state, action, reward) tuples
        """
        G = 0.0
        visited = set()

        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                self.N[state][action] += 1
                # Incremental mean: Q = Q + (G - Q) / N
                self.N[state][action] += 0  # already incremented
                n = self.N[state][action]
                self.Q[state][action] += (G - self.Q[state][action]) / n

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, state):
        return int(np.argmax(self.Q[state]))

    def save(self, path="mc_q.npy"):
        # Convert defaultdict to regular dict for saving
        np.save(path, dict(self.Q), allow_pickle=True)
        print(f"Q-table saved → {path}")

    def load(self, path="mc_q.npy"):
        data = np.load(path, allow_pickle=True).item()
        for k, v in data.items():
            self.Q[k] = v
        self.epsilon = self.epsilon_min
        print(f"Q-table loaded ← {path}")
