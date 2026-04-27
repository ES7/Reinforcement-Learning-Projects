import numpy as np
from collections import defaultdict


class TDAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


class MCAgent:
    def __init__(self, n_states, n_actions, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions))
        self.N = np.zeros((n_states, n_actions))   # visit counts

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, episode):
        """episode: list of (state, action, reward)"""
        G = 0.0
        visited = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                self.N[state, action] += 1
                self.Q[state, action] += (G - self.Q[state, action]) / self.N[state, action]

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)
