"""
ToolUseEnv — Custom Gymnasium environment.

State:  encoded representation of (task, tool_history, last_result)
Action: discrete — which tool to call (0=calculator, 1=dictionary, 2=search)
        + a continuous arg index selecting the tool input

For simplicity, tool inputs are pre-defined per task — the agent learns
WHICH tool to use, not how to construct the argument from scratch.
This keeps the environment tractable while demonstrating the core RL loop.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tools import call_tool, TOOL_NAMES, N_TOOLS


# ─────────────────────────────────────────────
# Task bank
# ─────────────────────────────────────────────

TASKS = [
    # (task_text, correct_tool, tool_input, answer_keywords)
    ("What is 144 * 37?",                   "calculator",  "144 * 37",         ["5328"]),
    ("What is sqrt(256)?",                   "calculator",  "sqrt(256)",        ["16"]),
    ("What is 1000 / 8?",                    "calculator",  "1000 / 8",         ["125"]),
    ("What is 2 ** 10?",                     "calculator",  "2 ** 10",          ["1024"]),
    ("What is 99 * 99?",                     "calculator",  "99 * 99",          ["9801"]),
    ("What is 17 + 83 * 2?",                 "calculator",  "17 + 83 * 2",      ["183"]),
    ("Define the word photosynthesis.",      "dictionary",  "photosynthesis",   ["sunlight", "plants", "oxygen"]),
    ("What does entropy mean?",              "dictionary",  "entropy",          ["randomness", "disorder"]),
    ("Define algorithm.",                    "dictionary",  "algorithm",        ["step", "procedure", "problem"]),
    ("What is the meaning of osmosis?",      "dictionary",  "osmosis",          ["membrane", "water"]),
    ("Define catalyst.",                     "dictionary",  "catalyst",         ["reaction", "chemical"]),
    ("What does metabolism mean?",           "dictionary",  "metabolism",       ["chemical", "organism"]),
    ("Tell me about reinforcement learning.","search",      "reinforcement learning", ["learning", "agent", "environment"]),
    ("What is PyTorch?",                     "search",      "pytorch",          ["deep learning", "meta", "framework"]),
    ("Search for information on black holes.","search",     "black hole",       ["gravity", "spacetime"]),
    ("What do you know about DNA?",          "search",      "dna",              ["genetic", "molecule"]),
    ("Find info about the Eiffel Tower.",    "search",      "eiffel tower",     ["paris", "1889", "330"]),
    ("What is the speed of light?",          "search",      "speed of light",   ["299", "vacuum", "metres"]),
    ("What is 512 + 768?",                   "calculator",  "512 + 768",        ["1280"]),
    ("Define the word gradient.",            "dictionary",  "gradient",         ["rate", "change", "function"]),
    ("What is deep learning?",               "search",      "deep learning",    ["neural", "machine learning"]),
    ("What is 3.14159 * 100?",               "calculator",  "3.14159 * 100",    ["314"]),
    ("Define democracy.",                    "dictionary",  "democracy",        ["government", "vote"]),
    ("Search for information on climate change.", "search", "climate change",   ["temperature", "global", "human"]),
]


# ─────────────────────────────────────────────
# Observation encoder
# ─────────────────────────────────────────────

VOCAB_SIZE   = 50
TASK_EMB_DIM = 16
OBS_DIM      = TASK_EMB_DIM + N_TOOLS + 8   # task + tool history + result signal

def encode_task(task_text: str) -> np.ndarray:
    """Simple hash-based task embedding — deterministic, no LLM needed."""
    vec = np.zeros(TASK_EMB_DIM, dtype=np.float32)
    words = task_text.lower().split()
    for i, word in enumerate(words[:TASK_EMB_DIM]):
        vec[i] = (hash(word) % 1000) / 1000.0
    return vec

def encode_tool_history(history: list) -> np.ndarray:
    """One-hot count of tools used so far."""
    vec = np.zeros(N_TOOLS, dtype=np.float32)
    for tool in history:
        if tool in TOOL_NAMES:
            vec[TOOL_NAMES.index(tool)] += 1
    return vec / max(1, len(history))

def encode_result_signal(last_result: str, answer_keywords: list) -> np.ndarray:
    """8-dim signal: has result, is error, keyword hit count, length."""
    vec = np.zeros(8, dtype=np.float32)
    if last_result:
        vec[0] = 1.0
        vec[1] = 1.0 if "ERROR" in last_result else 0.0
        hits = sum(1 for kw in answer_keywords if kw.lower() in last_result.lower())
        vec[2] = hits / max(1, len(answer_keywords))
        vec[3] = min(1.0, len(last_result) / 200)
    return vec


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class ToolUseEnv(gym.Env):
    """
    Custom Gym environment for LLM tool-use via RL.

    Observation: encoded (task, tool_history, last_result)
    Action:      Discrete(N_TOOLS) — which tool to call
    Reward:      +1.0 correct tool + answer keywords found
                 -0.3 wrong tool
                 -0.1 error result
                 -0.05 per step (efficiency penalty)
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=5):
        super().__init__()
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_TOOLS)

        self.task_idx      = 0
        self.current_task  = None
        self.tool_history  = []
        self.last_result   = ""
        self.steps         = 0
        self.solved        = False

    def _get_obs(self):
        task_text, _, _, answer_kw = TASKS[self.task_idx]
        return np.concatenate([
            encode_task(task_text),
            encode_tool_history(self.tool_history),
            encode_result_signal(self.last_result, answer_kw),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Cycle through tasks
        self.task_idx     = np.random.randint(len(TASKS))
        self.tool_history = []
        self.last_result  = ""
        self.steps        = 0
        self.solved       = False

        task_text, correct_tool, tool_input, answer_kw = TASKS[self.task_idx]
        self.current_task = {
            "text":         task_text,
            "correct_tool": correct_tool,
            "tool_input":   tool_input,
            "answer_kw":    answer_kw,
        }
        return self._get_obs(), {"task": task_text}

    def step(self, action):
        self.steps += 1
        tool_name  = TOOL_NAMES[action]
        tool_input = self.current_task["tool_input"]

        # Call the tool
        result = call_tool(tool_name, tool_input)
        self.last_result = result
        self.tool_history.append(tool_name)

        # ── Reward ────────────────────────────────────────────────────
        reward = -0.05   # step penalty

        correct = (tool_name == self.current_task["correct_tool"])
        if correct:
            reward += 0.5

        # Keyword hits in result
        hits = sum(
            1 for kw in self.current_task["answer_kw"]
            if kw.lower() in result.lower()
        )
        keyword_bonus = hits / max(1, len(self.current_task["answer_kw"]))
        reward += keyword_bonus * 0.5

        if "ERROR" in result:
            reward -= 0.1

        # Solved: correct tool AND at least one keyword found
        terminated = correct and hits > 0
        truncated  = self.steps >= self.max_steps

        if terminated:
            reward += 1.0   # success bonus
            self.solved = True

        obs  = self._get_obs()
        info = {
            "task":         self.current_task["text"],
            "tool_chosen":  tool_name,
            "correct_tool": self.current_task["correct_tool"],
            "result":       result,
            "correct":      correct,
            "keyword_hits": hits,
        }
        return obs, reward, terminated, truncated, info
