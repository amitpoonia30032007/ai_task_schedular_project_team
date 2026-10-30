"""Optional algorithms: Simulated Annealing, Hill Climbing and the Hybrid"""
from __future__ import annotations

import math
import random
import time

from .analysis import finalize
from .backtracking import BacktrackingSolver
from .min_conflicts import greedy_initial
from .state import State

def simulated_annealing(p, seed=0, iters=60000, t0=10.0, t_min=0.05, time_limit=30.0,
                        swap_prob=0.3, start=None, hill_climbing=False, name=None):
    rng = random.Random(seed)
    clock = time.perf_counter()
    st = State(p, track_soft=True)
    if start:
        for sid, v in start.items():
            st.add(sid, v)
    greedy_initial(p, st, rng)

    n = len(p.sessions)
    dom_set = [set(d) for d in p.domains]
    groups = {}
    for s in p.sessions:
        groups.setdefault((s.kind, s.duration), []).append(s.sid)
    swap_groups = [g for g in groups.values() if len(g) > 1]

    alpha = (t_min / t0) ** (1.0 / max(1, iters))
    T = 0.0 if hill_climbing else t0
    best_cost, best_assign = st.cost, dict(st.assign)
    start_soft = st.soft
    every = max(1, iters // 1500)
    trace = [st.cost]
    accepted = it = 0

    while it < iters and best_cost > 0:
        if (it & 127) == 0 and time.perf_counter() - clock > time_limit:
            break
        it += 1
        before = st.cost
        undo = []
        if swap_groups and rng.random() < swap_prob:
            a, b = rng.sample(rng.choice(swap_groups), 2)
            va, vb = st.assign[a], st.assign[b]
            if vb in dom_set[a] and va in dom_set[b] and va != vb:
                st.remove(a); st.remove(b)
                st.add(a, vb); st.add(b, va)
                undo = [(a, va), (b, vb)]
        if not undo:
            sid = rng.randrange(n)
            old = st.assign[sid]
            new = rng.choice(p.domains[sid])
            if new == old:
                continue
            st.remove(sid)
            st.add(sid, new)
            undo = [(sid, old)]

        delta = st.cost - before
        if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            accepted += 1
            if st.cost < best_cost:
                best_cost, best_assign = st.cost, dict(st.assign)
        else:
            for sid, _ in undo:
                st.remove(sid)
            for sid, v in undo:
                st.add(sid, v)
        if not hill_climbing:
            T *= alpha
        if it % every == 0:
            trace.append(st.cost)

    label = name or ("Hill Climbing" if hill_climbing else "Simulated Annealing")
    return finalize(p, label, best_assign, time.perf_counter() - clock, it, trace,
                    {"accepted_moves": accepted, "final_temperature": round(T, 4),
                     "soft_before": start_soft, "t0": t0, "status": "finished"})

def hybrid(p, seed=0, time_limit=60.0, sa_iters=60000, t0=5.0, **_):
    """Backtracking (MRV + degree + forward checking) -> Simulated Annealing."""
    clock = time.perf_counter()
    bt = BacktrackingSolver(p, mrv=True, degree=True, forward_checking=True, seed=seed,
                            time_limit=time_limit * 0.5).solve()
    left = max(1.0, time_limit - (time.perf_counter() - clock))
    sa = simulated_annealing(p, seed=seed, iters=sa_iters, t0=t0, time_limit=left,
                             start=bt.assignment, name="Hybrid (BT + SA)")
    sa.steps += bt.steps
    sa.seconds = time.perf_counter() - clock
    sa.info.update({"bt_nodes": bt.steps, "bt_status": bt.info.get("status"),
                    "soft_after_backtracking": bt.soft, "soft_after_sa": sa.soft})
    return sa
