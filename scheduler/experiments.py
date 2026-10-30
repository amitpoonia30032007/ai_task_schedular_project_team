"""Experiment harness: run algorithms x datasets x seeds and summarise."""
from __future__ import annotations

import pandas as pd

from .runner import run_algorithm

ALIASES = {
    "Backtracking (plain)": "Backtracking (baseline)",
    "Backtracking + heuristics": "Backtracking (MRV+Degree+FC)",
}

def run_experiments(problems: dict, algorithms: list, runs: int = 10, time_limit: float = 20.0,
                    base_seed: int = 42, params: dict | None = None, progress=None):
    """problems: {dataset name: Problem}.  Returns (summary_df, raw_df, traces)."""
    rows, traces = [], {}
    total = len(problems) * len(algorithms) * runs
    done = 0
    for dname, p in problems.items():
        for alg in algorithms:
            for k in range(runs):
                r = run_algorithm(p, alg, seed=base_seed + k, time_limit=time_limit, **(params or {}))
                rows.append({"dataset": dname, "algorithm": ALIASES.get(alg, alg), "run": k + 1,
                             "seed": base_seed + k, "seconds": r.seconds, "steps": r.steps,
                             "hard": r.hard, "soft": r.soft, "success": int(r.success)})
                if r.trace and (dname, alg) not in traces:
                    traces[(dname, alg)] = r.trace
                done += 1
                if progress:
                    progress(done / total, f"{dname} - {alg} - run {k + 1}/{runs}")
    raw = pd.DataFrame(rows)
    g = raw.groupby(["dataset", "algorithm"], sort=False)
    summary = g.agg(time_s=("seconds", "mean"), nodes_or_iterations=("steps", "mean"),
                    hard_violations=("hard", "mean"), soft_penalty=("soft", "mean"),
                    successes=("success", "sum"), runs=("run", "count")).reset_index()
    summary["success_rate"] = summary.apply(lambda r: f"{int(r.successes)}/{int(r.runs)}", axis=1)
    summary = summary.drop(columns=["successes", "runs"])
    for c in ("time_s", "nodes_or_iterations", "hard_violations", "soft_penalty"):
        summary[c] = summary[c].round(3 if c == "time_s" else 1)
    return summary, raw, traces
