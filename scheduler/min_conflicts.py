"""Algorithm 2 - Min-Conflicts local search."""
from __future__ import annotations

import random
import time

from .analysis import finalize
from .state import State

def greedy_initial(p, state: State, rng: random.Random, tie_break="soft", randomize=False, max_tie_eval=12):
    """Place every unplaced session. Works on a partially filled state too."""
    todo = [s.sid for s in p.sessions if s.sid not in state.assign]
    rng.shuffle(todo)
    todo.sort(key=lambda sid: len(p.domains[sid]))
    for sid in todo:
        dom = p.domains[sid]
        if randomize:
            state.add(sid, rng.choice(dom))
            continue
        state.add(sid, _best_value(p, state, sid, dom, rng, tie_break, max_tie_eval, exclude=None, placed=False))
    return state

def _best_value(p, state, sid, dom, rng, tie_break, max_tie_eval, exclude, placed=False):
    best, ties = None, []
    vc = state.value_conflicts
    for v in dom:
        if v == exclude:
            continue
        c = vc(sid, v)
        if best is None or c < best:
            best, ties = c, [v]
        elif c == best:
            ties.append(v)
    if not ties:
        return exclude
    if len(ties) == 1 or tie_break != "soft":
        return rng.choice(ties)
    sample = rng.sample(ties, min(max_tie_eval, len(ties)))
    pick, pick_soft = None, None
    for v in sample:
        state.add(sid, v)
        sc = state.soft
        state.remove(sid)
        if pick_soft is None or sc < pick_soft:
            pick, pick_soft = v, sc
    return pick

def min_conflicts(p, seed=0, max_iters=20000, time_limit=30.0, p_walk=0.05, restart_after=1500,
                  restart_fraction=0.2, init="random", tie_break="soft", max_tie_eval=12, start=None):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    st = State(p, track_soft=True)
    if start:
        for sid, v in start.items():
            st.add(sid, v)
    greedy_initial(p, st, rng, tie_break, randomize=(init == "random"), max_tie_eval=max_tie_eval)

    conflicted = st.conflicted_set()
    trace = [st.cost]
    best_assign, best_key = dict(st.assign), (st.hard, st.soft)
    best_hard_seen, stagnation, restarts, walks, it = st.hard, 0, 0, 0, 0

    while st.hard > 0 and it < max_iters:
        if (it & 31) == 0 and time.perf_counter() - t0 > time_limit:
            break
        it += 1
        sid = rng.choice(list(conflicted))
        old = st.assign[sid]
        affected = st.cell_occupants(sid, old)
        st.remove(sid)
        dom = p.domains[sid]
        if rng.random() < p_walk:
            new, walks = rng.choice(dom), walks + 1
        else:
            new = _best_value(p, st, sid, dom, rng, tie_break, max_tie_eval, exclude=old)
        st.add(sid, new)
        affected |= st.cell_occupants(sid, new)
        affected.add(sid)
        for x in affected:
            if st.conflicts_of(x) > 0:
                conflicted.add(x)
            else:
                conflicted.discard(x)
        trace.append(st.cost)

        key = (st.hard, st.soft)
        if key < best_key:
            best_key, best_assign = key, dict(st.assign)
        if st.hard < best_hard_seen:
            best_hard_seen, stagnation = st.hard, 0
        else:
            stagnation += 1
        if stagnation >= restart_after and st.hard > 0:
            restarts += 1
            stagnation = 0
            for x in rng.sample(range(len(p.sessions)), max(1, int(restart_fraction * len(p.sessions)))):
                st.remove(x)
                st.add(x, rng.choice(p.domains[x]))
            conflicted = st.conflicted_set()
            best_hard_seen = st.hard

    final = dict(st.assign) if st.hard == 0 else best_assign
    return finalize(p, "Min-Conflicts", final, time.perf_counter() - t0, it, trace,
                    {"restarts": restarts, "random_walks": walks, "init": init, "tie_break": tie_break,
                     "status": "solved" if st.hard == 0 else "iteration/time limit reached"})
