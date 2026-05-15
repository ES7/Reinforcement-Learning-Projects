"""
Tool definitions for the LLM Tool-Use Agent.
Each tool takes a string argument and returns a string result.
"""

import math
import re


# ─────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────

TOOLS = {
    "calculator": "Evaluates a math expression. Input: expression string like '182 * 47'",
    "dictionary": "Defines a word. Input: a single English word",
    "search":     "Searches a knowledge base. Input: a keyword or short phrase",
}

TOOL_NAMES  = list(TOOLS.keys())
N_TOOLS     = len(TOOL_NAMES)


# ─────────────────────────────────────────────
# Tool 1 — Calculator
# ─────────────────────────────────────────────

def tool_calculator(expr: str) -> str:
    """
    Safely evaluate a math expression.
    Supports: +, -, *, /, **, sqrt, abs, round, sin, cos, log
    """
    try:
        expr = expr.strip().lower()
        # Replace common english forms
        expr = expr.replace("^", "**")
        expr = expr.replace("sqrt", "math.sqrt")
        expr = expr.replace("sin",  "math.sin")
        expr = expr.replace("cos",  "math.cos")
        expr = expr.replace("log",  "math.log")
        expr = expr.replace("abs",  "abs")
        expr = expr.replace("pi",   "math.pi")

        # Only allow safe characters
        if not re.match(r"^[0-9\s\+\-\*\/\(\)\.\,mathsqriclogbepa]+$", expr):
            return "ERROR: unsafe expression"

        result = eval(expr, {"__builtins__": {}, "math": math, "abs": abs, "round": round})
        return str(round(float(result), 6))
    except Exception as e:
        return f"ERROR: {str(e)}"


# ─────────────────────────────────────────────
# Tool 2 — Dictionary
# ─────────────────────────────────────────────

DICTIONARY = {
    "photosynthesis":  "The process by which plants use sunlight, water, and CO2 to produce oxygen and energy.",
    "algorithm":       "A step-by-step procedure for solving a problem or accomplishing a task.",
    "entropy":         "A measure of randomness or disorder in a system.",
    "gradient":        "The rate of change of a function with respect to its variables.",
    "neuron":          "A nerve cell that transmits electrical signals in the brain and nervous system.",
    "osmosis":         "The movement of water through a semi-permeable membrane from low to high solute concentration.",
    "catalyst":        "A substance that speeds up a chemical reaction without being consumed.",
    "hypothesis":      "A proposed explanation for an observation that can be tested.",
    "democracy":       "A system of government where citizens vote to elect representatives.",
    "metabolism":      "The chemical processes in a living organism that sustain life.",
    "gravity":         "The force that attracts objects with mass toward each other.",
    "inflation":       "The rate at which the general level of prices for goods and services rises.",
    "mitosis":         "A type of cell division resulting in two genetically identical daughter cells.",
    "quantum":         "The minimum discrete unit of any physical property, especially energy.",
    "renaissance":     "A period of European cultural and artistic revival from the 14th to 17th century.",
    "stoicism":        "A philosophy teaching that virtue and reason are sufficient for happiness.",
    "symbiosis":       "A close interaction between two different organisms that benefits at least one.",
    "transistor":      "A semiconductor device used to amplify or switch electronic signals.",
    "vaccination":     "The administration of a vaccine to stimulate immune response against a disease.",
    "wavelength":      "The distance between successive crests of a wave.",
}

def tool_dictionary(word: str) -> str:
    word = word.strip().lower()
    if word in DICTIONARY:
        return DICTIONARY[word]
    # Partial match
    for k, v in DICTIONARY.items():
        if word in k or k in word:
            return f"({k}): {v}"
    return f"ERROR: '{word}' not found in dictionary"


# ─────────────────────────────────────────────
# Tool 3 — Search (knowledge base)
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "python":         "Python is a high-level, interpreted programming language known for readability.",
    "reinforcement learning": "RL is a type of machine learning where agents learn by interacting with an environment.",
    "neural network": "A computational model inspired by the brain, used in deep learning.",
    "openai":         "OpenAI is an AI research company known for GPT and ChatGPT.",
    "gymnasium":      "Gymnasium is a toolkit for developing reinforcement learning environments.",
    "pytorch":        "PyTorch is an open-source deep learning framework developed by Meta.",
    "transformer":    "A neural network architecture based on self-attention, used in LLMs.",
    "deep learning":  "A subset of machine learning using multi-layered neural networks.",
    "cpu":            "Central Processing Unit — the primary processor in a computer.",
    "gpu":            "Graphics Processing Unit — used for parallel computation in deep learning.",
    "eiffel tower":   "A wrought-iron lattice tower in Paris, France, built in 1889. Height: 330m.",
    "mount everest":  "The highest mountain on Earth at 8,848.86m, located in the Himalayas.",
    "photosynthesis": "The process plants use to convert sunlight into glucose and oxygen.",
    "water":          "H2O — a compound of hydrogen and oxygen, essential for life.",
    "speed of light": "Approximately 299,792,458 metres per second in a vacuum.",
    "dna":            "Deoxyribonucleic acid — the molecule carrying genetic information in living organisms.",
    "black hole":     "A region of spacetime where gravity is so strong that nothing can escape it.",
    "climate change": "Long-term shifts in global temperatures and weather patterns, accelerated by human activity.",
    "bitcoin":        "A decentralised digital currency using blockchain technology, created in 2009.",
    "internet":       "A global network of computers that communicate via standardised protocols.",
}

def tool_search(query: str) -> str:
    query = query.strip().lower()
    # Exact match
    if query in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[query]
    # Partial match — find best hit
    best_key, best_score = None, 0
    for key in KNOWLEDGE_BASE:
        words_in_common = len(set(query.split()) & set(key.split()))
        if words_in_common > best_score:
            best_score = words_in_common
            best_key   = key
        if query in key or key in query:
            return KNOWLEDGE_BASE[key]
    if best_key and best_score > 0:
        return f"({best_key}): {KNOWLEDGE_BASE[best_key]}"
    return f"ERROR: no results for '{query}'"


# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────

def call_tool(tool_name: str, tool_input: str) -> str:
    if tool_name == "calculator":
        return tool_calculator(tool_input)
    elif tool_name == "dictionary":
        return tool_dictionary(tool_input)
    elif tool_name == "search":
        return tool_search(tool_input)
    else:
        return f"ERROR: unknown tool '{tool_name}'"
