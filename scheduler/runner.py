"""Single entry point so the UI, experiments and tests call algorithms the same way."""
from __future__ import annotations

from .annealing import hybrid, simulated_annealing
from .backtracking import BacktrackingSolver
from .min_conflicts import min_conflicts

ALGORITHMS = [
    "Backtracking (plain)",
    "Backtracking + heuristics",
    "Min-Conflicts",
    "Simulated Annealing",
    "Hill Climbing",
    "Hybrid (Backtracking + SA)",
]
CORE = ALGORITHMS[:3]

def run_algorithm(p, name: str, seed: int = 0, time_limit: float = 30.0, **params):
    if name == "Backtracking (plain)":
        return BacktrackingSolver(p, seed=seed, time_limit=time_limit).solve()
    if name == "Backtracking + heuristics":
        opts = {"mrv": True, "degree": True, "forward_checking": True, "lcv": False, "ac3": False}
        opts.update({k: v for k, v in params.items() if k in opts})
        return BacktrackingSolver(p, mrv=opts["mrv"], degree=opts["degree"], lcv=opts["lcv"],
                                  forward_checking=opts["forward_checking"], ac3=opts["ac3"],
                                  seed=seed, time_limit=time_limit).solve()
    if name == "Min-Conflicts":
        keys = ("max_iters", "p_walk", "restart_after", "init", "tie_break")
        return min_conflicts(p, seed=seed, time_limit=time_limit, **{k: v for k, v in params.items() if k in keys})
    if name == "Simulated Annealing":
        keys = ("iters", "t0", "swap_prob")
        return simulated_annealing(p, seed=seed, time_limit=time_limit, **{k: v for k, v in params.items() if k in keys})
    if name == "Hill Climbing":
        keys = ("iters", "swap_prob")
        return simulated_annealing(p, seed=seed, time_limit=time_limit, hill_climbing=True,
                                   **{k: v for k, v in params.items() if k in keys})
    if name == "Hybrid (Backtracking + SA)":
        keys = ("sa_iters", "t0")
        return hybrid(p, seed=seed, time_limit=time_limit, **{k: v for k, v in params.items() if k in keys})
    raise ValueError(f"Unknown algorithm: {name}")
