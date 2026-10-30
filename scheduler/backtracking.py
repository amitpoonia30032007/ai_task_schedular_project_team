"""Algorithm 1 - Backtracking CSP search."""
from __future__ import annotations

import random
import sys
import time
from collections import deque

from .analysis import finalize
from .state import State

class _Abort(Exception):
    pass

class BacktrackingSolver:
    def __init__(self, problem, mrv=False, degree=False, lcv=False, forward_checking=False,
                 ac3=False, seed=0, time_limit=30.0, node_limit=None, shuffle_values=True):
        self.p = problem
        self.mrv, self.degree, self.lcv = mrv, degree, lcv
        self.fc, self.use_ac3 = forward_checking, ac3
        self.seed, self.time_limit, self.node_limit = seed, time_limit, node_limit
        self.shuffle = shuffle_values

    def solve(self):
        p = self.p
        rng = random.Random(self.seed)
        self.order = [list(d) for d in p.domains]
        if self.shuffle:
            for o in self.order:
                rng.shuffle(o)
        self.dom = [set(d) for d in p.domains]
        self.state = State(p, track_soft=False)
        self.nodes = 0
        self.backtracks = 0
        self.deg = [len(nb) for nb in p.neighbors]
        self.best = {}
        self.t0 = time.perf_counter()
        self.pruned_by_ac3 = 0
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

        solved, status = False, "unsolved"
        try:
            if self.use_ac3 and not self._ac3():
                status = "AC-3 emptied a domain: problem is unsatisfiable"
            else:
                solved = self._search()
                status = "solved" if solved else "search space exhausted"
        except _Abort:
            status = "time/node limit reached"
        assignment = dict(self.state.assign) if solved else dict(self.best)
        return finalize(p, self.name, assignment, time.perf_counter() - self.t0, self.nodes,
                        info={"status": status, "backtracks": self.backtracks,
                              "ac3_pruned": self.pruned_by_ac3})

    @property
    def name(self) -> str:
        flags = [n for n, on in (("MRV", self.mrv), ("Degree", self.degree), ("LCV", self.lcv),
                                 ("FC", self.fc), ("AC-3", self.use_ac3)) if on]
        return "Backtracking (" + "+".join(flags) + ")" if flags else "Backtracking (plain)"

    def _search(self) -> bool:
        n = len(self.p.sessions)
        if len(self.state.assign) == n:
            return True
        if (self.nodes & 63) == 0 and time.perf_counter() - self.t0 > self.time_limit:
            raise _Abort
        if self.node_limit and self.nodes >= self.node_limit:
            raise _Abort
        sid = self._select_var()
        for val in self._ordered_values(sid):
            if not self.fc and self.state.value_conflicts(sid, val) > 0:
                continue
            self.nodes += 1
            self.state.add(sid, val)
            for nb in self.p.neighbors[sid]:
                self.deg[nb] -= 1
            trail = []
            ok = self._forward_check(sid, val, trail) if self.fc else True
            if ok:
                if len(self.state.assign) > len(self.best):
                    self.best = dict(self.state.assign)
                if self._search():
                    return True
            for nb, v in trail:
                self.dom[nb].add(v)
            for nb in self.p.neighbors[sid]:
                self.deg[nb] += 1
            self.state.remove(sid)
            self.backtracks += 1
        return False

    def _live_count(self, sid) -> int:
        """Remaining legal values of an unassigned session (when FC is off)."""
        vc = self.state.value_conflicts
        return sum(1 for v in self.dom[sid] if vc(sid, v) == 0)

    def _select_var(self) -> int:
        assign = self.state.assign
        free = [s for s in range(len(self.p.sessions)) if s not in assign]
        if not (self.mrv or self.degree):
            return free[0]

        def key(s):
            size = (len(self.dom[s]) if self.fc else self._live_count(s)) if self.mrv else 0
            deg = -self.deg[s] if self.degree else 0
            return (size, deg, -self.p.sessions[s].duration, s)
        return min(free, key=key)

    def _ordered_values(self, sid):
        vals = [v for v in self.order[sid] if v in self.dom[sid]]
        if not self.lcv:
            return vals
        cache = {}
        dur = self.p.sessions[sid].duration

        def eliminated(v):
            k = (v[0], v[1])
            if k not in cache:
                tot = 0
                for nb in self.p.neighbors[sid]:
                    if nb in self.state.assign:
                        continue
                    dn = self.p.sessions[nb].duration
                    for s2 in range(v[1] - dn + 1, v[1] + dur):
                        for r2 in self.p.rooms_of[nb]:
                            if (v[0], s2, r2) in self.dom[nb]:
                                tot += 1
                cache[k] = tot
            return cache[k]
        return sorted(vals, key=eliminated)

    def _forward_check(self, sid, val, trail) -> bool:
        p, assign = self.p, self.state.assign
        d, st, r = val
        dur = p.sessions[sid].duration
        for nb in p.neighbors[sid]:
            if nb in assign:
                continue
            dom, dn = self.dom[nb], p.sessions[nb].duration
            for s2 in range(st - dn + 1, st + dur):
                for r2 in p.rooms_of[nb]:
                    v2 = (d, s2, r2)
                    if v2 in dom:
                        dom.remove(v2)
                        trail.append((nb, v2))
            if not dom:
                return False
        for nb in p.room_users[r]:
            if nb in assign or nb == sid:
                continue
            dom, dn = self.dom[nb], p.sessions[nb].duration
            for s2 in range(st - dn + 1, st + dur):
                v2 = (d, s2, r)
                if v2 in dom:
                    dom.remove(v2)
                    trail.append((nb, v2))
            if not dom:
                return False
        return True

    def _revise(self, x, y) -> bool:
        """Drop values of x that no value of y can coexist with (time overlap only;"""
        sx, sy = self.p.sessions[x], self.p.sessions[y]
        ty = {(v[0], v[1]) for v in self.dom[y]}
        if len(ty) > sx.duration + sy.duration - 1:
            return False
        removed = False
        for v in list(self.dom[x]):
            d, s, _ = v
            if all(d2 == d and s2 < s + sx.duration and s < s2 + sy.duration for d2, s2 in ty):
                self.dom[x].discard(v)
                self.pruned_by_ac3 += 1
                removed = True
        return removed

    def _ac3(self) -> bool:
        queue = deque((x, y) for x in range(len(self.p.sessions)) for y in self.p.neighbors[x])
        while queue:
            x, y = queue.popleft()
            if self._revise(x, y):
                if not self.dom[x]:
                    return False
                queue.extend((z, x) for z in self.p.neighbors[x] if z != y)
        return True

def backtracking_plain(p, **kw):
    return BacktrackingSolver(p, **kw).solve()

def backtracking_heuristics(p, mrv=True, degree=True, lcv=False, forward_checking=True, ac3=False, **kw):
    return BacktrackingSolver(p, mrv=mrv, degree=degree, lcv=lcv, forward_checking=forward_checking,
                              ac3=ac3, **kw).solve()
