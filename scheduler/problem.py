"""CSP formulation of the timetable problem."""
from __future__ import annotations

from dataclasses import dataclass, field

class ProblemError(ValueError):
    """Raised when the input data cannot form a solvable CSP."""

@dataclass
class Session:
    sid: int
    cls: int
    subject: str
    subject_name: str
    teacher: int
    duration: int
    kind: str
    index: int

@dataclass
class Problem:
    name: str
    days: list
    n_days: int
    n_periods: int
    class_names: list
    class_strength: list
    teacher_names: list
    teacher_max_load: list
    room_names: list
    room_caps: list
    room_kinds: list
    lunch: list
    break_periods: set
    usable: list
    t_unavail: list
    t_avoid: list
    sessions: list = field(default_factory=list)
    domains: list = field(default_factory=list)
    neighbors: list = field(default_factory=list)
    rooms_of: list = field(default_factory=list)
    room_users: list = field(default_factory=list)
    required: dict = field(default_factory=dict)

    def label(self, sid: int, short: bool = False) -> str:
        s = self.sessions[sid]
        base = f"{s.subject} #{s.index + 1}"
        return base if short else f"{base} ({self.class_names[s.cls]})"

    def describe(self, sid: int, val) -> str:
        d, st, r = val
        end = st + self.sessions[sid].duration
        per = f"P{st + 1}" if end - st == 1 else f"P{st + 1}-P{end}"
        return f"{self.days[d]} {per} in {self.room_names[r]}"

    def stats(self) -> dict:
        return {
            "sessions": len(self.sessions),
            "classes": len(self.class_names),
            "teachers": len(self.teacher_names),
            "rooms": len(self.room_names),
            "slots": self.n_days * self.n_periods,
            "avg_domain": round(sum(map(len, self.domains)) / max(1, len(self.domains)), 1),
        }

def parse_slot(text: str, days: list, n_periods: int):
    """'Mon-1' -> (0, 0).  Periods are 1-based in files, 0-based internally."""
    try:
        day_txt, per_txt = str(text).strip().rsplit("-", 1)
        per = int(per_txt) - 1
    except ValueError:
        raise ProblemError(f"Bad slot '{text}' (use Day-Period, e.g. Mon-1)")
    key = day_txt.strip().lower()
    for i, d in enumerate(days):
        if d.lower() == key or d.lower()[:3] == key[:3]:
            if not 0 <= per < n_periods:
                raise ProblemError(f"Slot '{text}': period out of range 1..{n_periods}")
            return i, per
    raise ProblemError(f"Slot '{text}': unknown day '{day_txt}'")

def assign_teachers(raw: dict) -> dict:
    """Fix a teacher for each (class, subject).  Greedy, scarcest subject first,"""
    subj = {s["code"]: s for s in raw["subjects"]}
    teachers = raw["teachers"]
    tnames = [t["name"] for t in teachers]
    load = [0] * len(teachers)
    cap = [int(t.get("max_load", 10 ** 6)) for t in teachers]
    out, pending = {}, []
    for ci, c in enumerate(raw["classes"]):
        forced = c.get("teachers", {}) or {}
        for code in c["subjects"]:
            h = int(subj[code]["hours"])
            if code in forced:
                if forced[code] not in tnames:
                    raise ProblemError(f"Class {c['name']}: unknown teacher '{forced[code]}' for {code}")
                ti = tnames.index(forced[code])
                out[(ci, code)] = ti
                load[ti] += h
            else:
                pending.append((ci, code, h))
    elig = {code: [i for i, t in enumerate(teachers) if code in t["subjects"]] for code in subj}
    pending.sort(key=lambda x: (len(elig[x[1]]), -x[2], x[0]))
    for ci, code, h in pending:
        options = [i for i in elig[code] if load[i] + h <= cap[i]]
        if not options:
            if not elig[code]:
                raise ProblemError(f"No teacher can teach subject {code}")
            raise ProblemError(
                f"Teachers for {code} have no spare load for class {raw['classes'][ci]['name']} "
                f"(raise max_load or add a teacher)")
        best = max(options, key=lambda i: (cap[i] - load[i], -i))
        out[(ci, code)] = best
        load[best] += h
    return out

def build_problem(raw: dict) -> Problem:
    days = list(raw["days"])
    P = int(raw["periods_per_day"])
    breaks = {int(b) - 1 for b in raw.get("break_periods", []) if str(b).strip() != ""}
    subj = {s["code"]: s for s in raw["subjects"]}

    classes, teachers, rooms = raw["classes"], raw["teachers"], raw["rooms"]
    lunch = [int(c.get("lunch_period", 0)) - 1 for c in classes]
    usable = [[(q != lunch[ci]) and (q not in breaks) for q in range(P)] for ci in range(len(classes))]

    p = Problem(
        name=raw.get("name", "dataset"), days=days, n_days=len(days), n_periods=P,
        class_names=[c["name"] for c in classes],
        class_strength=[int(c["strength"]) for c in classes],
        teacher_names=[t["name"] for t in teachers],
        teacher_max_load=[int(t.get("max_load", 10 ** 6)) for t in teachers],
        room_names=[r["name"] for r in rooms],
        room_caps=[int(r["capacity"]) for r in rooms],
        room_kinds=[r["type"] for r in rooms],
        lunch=lunch, break_periods=breaks, usable=usable,
        t_unavail=[{parse_slot(s, days, P) for s in t.get("unavailable", [])} for t in teachers],
        t_avoid=[{int(x) - 1 for x in t.get("avoid_periods", [])} for t in teachers],
    )

    tmap = assign_teachers(raw)
    for ci, c in enumerate(classes):
        for code in c["subjects"]:
            sj = subj[code]
            hours = int(sj["hours"])
            p.required[(ci, code)] = hours
            kind = sj["type"]
            if kind == "lab":
                block = int(sj.get("lab_block", 2)) or 2
                durs = [block] * (hours // block) + ([hours % block] if hours % block else [])
            else:
                durs = [1] * hours
            for k, dur in enumerate(durs):
                p.sessions.append(Session(len(p.sessions), ci, code, sj["name"], tmap[(ci, code)], dur, kind, k))

    from .state import unary_issues
    for s in p.sessions:
        dom = []
        for d in range(p.n_days):
            for st in range(P - s.duration + 1):
                for r in range(len(rooms)):
                    if not unary_issues(p, s.sid, (d, st, r)):
                        dom.append((d, st, r))
        if not dom:
            raise ProblemError(
                f"No legal slot exists for {p.label(s.sid)}: check room capacity/type, "
                f"teacher availability and lunch/break periods")
        p.domains.append(dom)
        p.rooms_of.append(sorted({v[2] for v in dom}))

    by_t, by_c = {}, {}
    for s in p.sessions:
        by_t.setdefault(s.teacher, []).append(s.sid)
        by_c.setdefault(s.cls, []).append(s.sid)
    for s in p.sessions:
        nb = set(by_t[s.teacher]) | set(by_c[s.cls])
        nb.discard(s.sid)
        p.neighbors.append(sorted(nb))
    p.room_users = [[s.sid for s in p.sessions if r in p.rooms_of[s.sid]] for r in range(len(rooms))]
    return p
