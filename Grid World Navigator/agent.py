import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent.
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha        # learning rate
        self.gamma     = gamma        # discount factor
        self.epsilon   = epsilon      # exploration rate
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape (64, 4) for 8x8 grid, 4 actions
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        return np.argmax(self.Q[state])                # exploit

    def update(self, state, action, reward, next_state, done):
        """Single Q-Learning update step."""
        best_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Return greedy policy: best action for every state."""
        return np.argmax(self.Q, axis=1)

    def save(self, path="q_table.npy"):
        np.save(path, self.Q)
        print(f"Q-table saved → {path}")

    def load(self, path="q_table.npy"):
        self.Q = np.load(path)
        self.epsilon = self.epsilon_min   # no exploration after loading
        print(f"Q-table loaded ← {path}")
