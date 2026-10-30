"""Timetable state with incremental cost bookkeeping."""
from __future__ import annotations

WEIGHTS = {"hard": 1000, "S1": 2, "S2": 3, "S4": 2, "S5": 2}

def unary_issues(p, sid: int, val) -> list:
    """Unary hard-constraint check for one session at (day, start, room)."""
    d, st, r = val
    s = p.sessions[sid]
    if not (0 <= r < len(p.room_names) and 0 <= d < p.n_days and st >= 0 and st + s.duration <= p.n_periods):
        return [("RANGE", "placed outside the timetable grid")]
    out = []
    need_room = "lab" if s.kind == "lab" else "classroom"
    if p.room_kinds[r] != need_room:
        out.append(("H5", f"{s.kind} session placed in {p.room_kinds[r]} '{p.room_names[r]}'"))
    if p.room_caps[r] < p.class_strength[s.cls]:
        out.append(("H4", f"room '{p.room_names[r]}' holds {p.room_caps[r]} but class has {p.class_strength[s.cls]}"))
    for q in range(st, st + s.duration):
        if not p.usable[s.cls][q]:
            what = "lunch" if q == p.lunch[s.cls] else "break"
            out.append(("H9", f"scheduled in {what} period P{q + 1}"))
        if (d, q) in p.t_unavail[s.teacher]:
            out.append(("H6", f"{p.teacher_names[s.teacher]} is unavailable {p.days[d]} P{q + 1}"))
    return out

class State:
    def __init__(self, problem, weights=None, track_soft: bool = True):
        self.p = problem
        self.w = weights or WEIGHTS
        self.track_soft = track_soft
        self.assign: dict = {}
        self.t_occ: dict = {}
        self.c_occ: dict = {}
        self.r_occ: dict = {}
        self.cd: dict = {}
        self.td: dict = {}
        self.hard = 0
        self.soft = 0
        self._soft_cache: dict = {}

    def add(self, sid: int, val) -> None:
        s = self.p.sessions[sid]
        d, st, r = val
        for q in range(st, st + s.duration):
            for occ, key in ((self.t_occ, (s.teacher, d, q)),
                             (self.c_occ, (s.cls, d, q)),
                             (self.r_occ, (r, d, q))):
                cell = occ.get(key)
                if cell is None:
                    occ[key] = {sid}
                else:
                    self.hard += len(cell)
                    cell.add(sid)
        self.assign[sid] = val
        if self.track_soft:
            self.cd.setdefault((s.cls, d), set()).add(sid)
            self.td.setdefault((s.teacher, d), set()).add(sid)
            self._refresh(s.cls, s.teacher, d)

    def remove(self, sid: int) -> None:
        s = self.p.sessions[sid]
        d, st, r = self.assign.pop(sid)
        for q in range(st, st + s.duration):
            for occ, key in ((self.t_occ, (s.teacher, d, q)),
                             (self.c_occ, (s.cls, d, q)),
                             (self.r_occ, (r, d, q))):
                cell = occ[key]
                cell.discard(sid)
                self.hard -= len(cell)
        if self.track_soft:
            self.cd[(s.cls, d)].discard(sid)
            self.td[(s.teacher, d)].discard(sid)
            self._refresh(s.cls, s.teacher, d)

    def value_conflicts(self, sid: int, val) -> int:
        """Hard overlaps `sid` would have at `val` (sid itself must NOT be placed)."""
        s = self.p.sessions[sid]
        d, st, r = val
        n = 0
        t_get, c_get, r_get = self.t_occ.get, self.c_occ.get, self.r_occ.get
        for q in range(st, st + s.duration):
            n += len(t_get((s.teacher, d, q), ())) + len(c_get((s.cls, d, q), ())) + len(r_get((r, d, q), ()))
        return n

    def conflicts_of(self, sid: int) -> int:
        s = self.p.sessions[sid]
        return self.value_conflicts(sid, self.assign[sid]) - 3 * s.duration

    def cell_occupants(self, sid: int, val) -> set:
        s = self.p.sessions[sid]
        d, st, r = val
        out = set()
        for q in range(st, st + s.duration):
            out |= self.t_occ.get((s.teacher, d, q), set())
            out |= self.c_occ.get((s.cls, d, q), set())
            out |= self.r_occ.get((r, d, q), set())
        return out

    def conflicted_set(self) -> set:
        return {sid for sid in self.assign if self.conflicts_of(sid) > 0}

    def class_day_parts(self, c: int, d: int):
        """(S1 gaps, S2 repeated theory subjects) for one class on one day."""
        sids = self.cd.get((c, d))
        if not sids:
            return 0, 0
        p = self.p
        occ, subj = set(), {}
        for sid in sids:
            s = p.sessions[sid]
            st = self.assign[sid][1]
            occ.update(range(st, st + s.duration))
            if s.kind == "theory":
                subj[s.subject] = subj.get(s.subject, 0) + 1
        use = p.usable[c]
        gaps = sum(1 for q in range(min(occ), max(occ) + 1) if q not in occ and use[q])
        rep = sum(k - 1 for k in subj.values() if k > 1)
        return gaps, rep

    def teacher_day_parts(self, t: int, d: int):
        """(S4 periods the teacher wanted to avoid, S5 periods beyond 2 in a row)."""
        sids = self.td.get((t, d))
        if not sids:
            return 0, 0
        p = self.p
        occ = set()
        for sid in sids:
            st = self.assign[sid][1]
            occ.update(range(st, st + p.sessions[sid].duration))
        pref = sum(1 for q in occ if q in p.t_avoid[t])
        long_, run, prev = 0, 0, None
        for q in sorted(occ):
            run = run + 1 if prev is not None and q == prev + 1 else 1
            if run > 2:
                long_ += 1
            prev = q
        return pref, long_

    def _refresh(self, c: int, t: int, d: int) -> None:
        w = self.w
        g, r = self.class_day_parts(c, d)
        new = w["S1"] * g + w["S2"] * r
        key = ("c", c, d)
        self.soft += new - self._soft_cache.get(key, 0)
        self._soft_cache[key] = new
        pf, lg = self.teacher_day_parts(t, d)
        new = w["S4"] * pf + w["S5"] * lg
        key = ("t", t, d)
        self.soft += new - self._soft_cache.get(key, 0)
        self._soft_cache[key] = new

    @property
    def cost(self) -> int:
        return self.w["hard"] * self.hard + self.soft
