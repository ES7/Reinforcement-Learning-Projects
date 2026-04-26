import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    8x8 Grid World Environment
    - Agent starts at top-left (0,0)
    - Goal is at bottom-right (7,7)
    - Walls block movement
    - Episode ends on reaching goal or max 200 steps
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode=None):
        self.grid_size = 8
        self.max_steps = 200
        self.render_mode = render_mode

        # 4 actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Observation: agent (row, col) — flattened to single int
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)

        # Fixed walls (row, col)
        self.walls = {
            (1,1),(1,2),(1,3),
            (2,5),(3,5),(4,5),
            (5,2),(5,3),(5,4),
            (6,6),(6,7) if False else (6,6),  # keep it solvable
        }
        self.walls = {(1,1),(1,2),(1,3),(2,5),(3,5),(4,5),(5,2),(5,3),(5,4),(6,6)}

        self.start = (0, 0)
        self.goal  = (7, 7)

        # For rendering
        self._cell = 64
        self.window = None
        self.clock  = None

    # ------------------------------------------------------------------
    def _pos_to_obs(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start)
        self.steps = 0
        self.path  = [tuple(self.agent_pos)]
        return self._pos_to_obs(self.agent_pos), {}

    def step(self, action):
        self.steps += 1
        r, c = self.agent_pos

        # Compute next position
        moves = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        # Boundary & wall check — stay in place if blocked
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and (nr, nc) not in self.walls:
            self.agent_pos = [nr, nc]

        self.path.append(tuple(self.agent_pos))

        pos = tuple(self.agent_pos)
        terminated = (pos == self.goal)
        truncated  = (self.steps >= self.max_steps)

        # Reward shaping
        if terminated:
            reward = 1.0
        else:
            # Small negative per step + distance-based shaping
            dist = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
            reward = -0.01 - 0.001 * dist

        obs = self._pos_to_obs(self.agent_pos)
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()

    def _render_frame(self):
        """Return RGB array of current grid state."""
        import pygame
        cell = self._cell
        size = self.grid_size * cell

        surf = pygame.Surface((size, size))
        surf.fill((245, 245, 240))  # background

        colors = {
            "empty": (245, 245, 240),
            "wall":  (50,  50,  60),
            "goal":  (29,  158, 117),
            "agent": (83,  74,  183),
            "path":  (206, 203, 246),
            "start": (240, 153, 123),
        }

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x, y = col * cell, row * cell
                rect = pygame.Rect(x, y, cell, cell)

                if (row, col) in self.walls:
                    pygame.draw.rect(surf, colors["wall"], rect)
                elif (row, col) == self.goal:
                    pygame.draw.rect(surf, colors["goal"], rect)
                elif (row, col) == self.start:
                    pygame.draw.rect(surf, colors["start"], rect)
                else:
                    pygame.draw.rect(surf, colors["empty"], rect)

                # Draw path trail
                if (row, col) in self.path[:-1]:
                    pygame.draw.rect(surf, colors["path"], rect.inflate(-20, -20))

                # Grid lines
                pygame.draw.rect(surf, (200, 200, 195), rect, 1)

        # Draw agent
        ar, ac = self.agent_pos
        cx = ac * cell + cell // 2
        cy = ar * cell + cell // 2
        pygame.draw.circle(surf, colors["agent"], (cx, cy), cell // 3)

        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def _render_human(self):
        import pygame
        pygame.init()
        cell = self._cell
        size = self.grid_size * cell

        if self.window is None:
            self.window = pygame.display.set_mode((size, size))
            pygame.display.set_caption("Grid World — RL Agent")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        frame = self._render_frame()
        surf  = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self.window.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
