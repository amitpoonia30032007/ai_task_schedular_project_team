"""Violation counting - the ONE function every algorithm, experiment and the"""
from __future__ import annotations

from dataclasses import dataclass, field

from .state import State, WEIGHTS, unary_issues

HARD_NAMES = {
    "H1": "Teacher clash", "H2": "Class clash", "H3": "Room clash",
    "H4": "Room too small", "H5": "Wrong room type", "H6": "Teacher unavailable",
    "H7": "Hours not scheduled", "H8": "Teacher overload", "H9": "Lunch/break violated",
    "RANGE": "Outside grid",
}

@dataclass
class Analysis:
    hard_total: int
    by_code: dict
    issues: list
    soft_total: int
    soft_parts: dict
    conflicted: set
    cost: int
    unscheduled: list = field(default_factory=list)

@dataclass
class SolveResult:
    algorithm: str
    assignment: dict
    success: bool
    seconds: float
    steps: int
    analysis: Analysis
    trace: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def hard(self) -> int:
        return self.analysis.hard_total

    @property
    def soft(self) -> int:
        return self.analysis.soft_total

    @property
    def cost(self) -> int:
        return self.analysis.cost

def _span(qs):
    qs = sorted(qs)
    return f"P{qs[0] + 1}" if len(qs) == 1 else f"P{qs[0] + 1}-P{qs[-1] + 1}"

def analyze(p, assignment: dict, weights=None) -> Analysis:
    w = weights or WEIGHTS
    st = State(p, w, track_soft=True)
    issues = []

    for sid, val in assignment.items():
        bad = unary_issues(p, sid, val)
        for code, msg in bad:
            issues.append({"code": code, "text": f"{p.label(sid)}: {msg}", "sids": [sid], "count": 1})
        if not (bad and bad[0][0] == "RANGE"):
            st.add(sid, val)

    groups = {}
    for occ, code, who in ((st.t_occ, "H1", lambda i: p.teacher_names[i]),
                           (st.c_occ, "H2", lambda i: p.class_names[i]),
                           (st.r_occ, "H3", lambda i: p.room_names[i])):
        for (res, d, q), cell in occ.items():
            if len(cell) > 1:
                g = groups.setdefault((code, res, d, frozenset(cell)), {"n": 0, "qs": [], "who": who(res)})
                g["n"] += len(cell) * (len(cell) - 1) // 2
                g["qs"].append(q)
    for (code, res, d, cell), g in groups.items():
        names = ", ".join(p.label(s) for s in sorted(cell))
        issues.append({
            "code": code, "sids": sorted(cell), "count": g["n"],
            "text": f"{g['who']} double-booked {p.days[d]} {_span(g['qs'])}: {names}"})

    got = {}
    unscheduled = []
    for s in p.sessions:
        if s.sid in assignment:
            got[(s.cls, s.subject)] = got.get((s.cls, s.subject), 0) + s.duration
        else:
            unscheduled.append(s.sid)
    for (c, subj), need in p.required.items():
        miss = need - got.get((c, subj), 0)
        if miss > 0:
            issues.append({"code": "H7", "sids": [], "count": miss,
                           "text": f"{p.class_names[c]} / {subj}: {miss} of {need} hours not scheduled"})

    load = {}
    for s in p.sessions:
        load[s.teacher] = load.get(s.teacher, 0) + s.duration
    for t, hrs in load.items():
        if hrs > p.teacher_max_load[t]:
            issues.append({"code": "H8", "sids": [], "count": hrs - p.teacher_max_load[t],
                           "text": f"{p.teacher_names[t]} teaches {hrs}h, maximum is {p.teacher_max_load[t]}h"})

    by_code = {}
    for it in issues:
        by_code[it["code"]] = by_code.get(it["code"], 0) + it["count"]
    hard_total = sum(by_code.values())
    conflicted = {s for it in issues for s in it["sids"]}

    parts = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0}
    for c in range(len(p.class_names)):
        for d in range(p.n_days):
            g, r = st.class_day_parts(c, d)
            parts["S1"] += g
            parts["S2"] += r
    for t in range(len(p.teacher_names)):
        for d in range(p.n_days):
            pf, lg = st.teacher_day_parts(t, d)
            parts["S4"] += pf
            parts["S5"] += lg

    return Analysis(hard_total, by_code, issues, st.soft, parts, conflicted,
                    w["hard"] * hard_total + st.soft, unscheduled)

def finalize(p, algorithm, assignment, seconds, steps, trace=None, info=None, weights=None) -> SolveResult:
    a = analyze(p, assignment, weights)
    ok = a.hard_total == 0 and len(assignment) == len(p.sessions)
    return SolveResult(algorithm, dict(assignment), ok, seconds, steps, a, trace or [], info or {})
