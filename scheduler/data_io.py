"""Reading, writing and validating timetable input data (form tables, CSV, JSON)."""
from __future__ import annotations

import io
import json
import re
import zipfile

import pandas as pd

from .problem import ProblemError, build_problem, parse_slot

DEFAULT_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

COLUMNS = {
    "subjects": ["code", "name", "hours", "type"],
    "teachers": ["name", "subjects", "unavailable", "max_load", "avoid_periods"],
    "classes": ["name", "strength", "subjects", "lunch_period"],
    "rooms": ["name", "capacity", "type"],
}

def split_list(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [x.strip() for x in re.split(r"[;,|]", str(value)) if x.strip()]

def join_list(values) -> str:
    return ", ".join(str(v) for v in values)

def raw_to_frames(raw: dict) -> dict:
    f = {
        "subjects": pd.DataFrame([{k: s.get(k) for k in COLUMNS["subjects"]} for s in raw["subjects"]],
                                 columns=COLUMNS["subjects"]),
        "teachers": pd.DataFrame([{
            "name": t["name"], "subjects": join_list(t["subjects"]),
            "unavailable": join_list(t.get("unavailable", [])), "max_load": t.get("max_load", 16),
            "avoid_periods": join_list(t.get("avoid_periods", []))} for t in raw["teachers"]],
            columns=COLUMNS["teachers"]),
        "classes": pd.DataFrame([{
            "name": c["name"], "strength": c["strength"], "subjects": join_list(c["subjects"]),
            "lunch_period": c.get("lunch_period", 4)} for c in raw["classes"]], columns=COLUMNS["classes"]),
        "rooms": pd.DataFrame([{k: r.get(k) for k in COLUMNS["rooms"]} for r in raw["rooms"]],
                              columns=COLUMNS["rooms"]),
    }
    return f

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").fillna("")
    return df[df.iloc[:, 0].astype(str).str.strip() != ""] if len(df) else df

def frames_to_raw(frames: dict, name: str, days, periods_per_day: int, break_periods, extra_teacher_map=None) -> dict:
    sub, tea, cls, rms = (_clean(frames[k]) for k in ("subjects", "teachers", "classes", "rooms"))

    def num(v, default=0):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    return {
        "name": name, "days": list(days), "periods_per_day": int(periods_per_day),
        "break_periods": [int(b) for b in break_periods],
        "subjects": [{"code": str(r.code).strip(), "name": str(r.name).strip(), "hours": num(r.hours),
                      "type": str(r.type).strip().lower()} for r in sub.itertuples()],
        "teachers": [{"name": str(r.name).strip(), "subjects": split_list(r.subjects),
                      "unavailable": split_list(r.unavailable), "max_load": num(r.max_load, 16),
                      "avoid_periods": [int(x) for x in split_list(r.avoid_periods) if str(x).isdigit()]}
                     for r in tea.itertuples()],
        "classes": [{"name": str(r.name).strip(), "strength": num(r.strength), "subjects": split_list(r.subjects),
                     "lunch_period": num(r.lunch_period, 4)} for r in cls.itertuples()],
        "rooms": [{"name": str(r.name).strip(), "capacity": num(r.capacity),
                   "type": str(r.type).strip().lower()} for r in rms.itertuples()],
    }

def load_json(data: bytes | str) -> dict:
    raw = json.loads(data)
    for key in ("days", "periods_per_day", "subjects", "teachers", "classes", "rooms"):
        if key not in raw:
            raise ProblemError(f"JSON is missing the '{key}' section")
    raw.setdefault("name", "uploaded dataset")
    raw.setdefault("break_periods", [])
    return raw

def load_csvs(files: dict, name: str, days, periods_per_day: int, break_periods) -> dict:
    """files: {'subjects': bytes, 'teachers': bytes, 'classes': bytes, 'rooms': bytes}"""
    frames = {}
    for key, cols in COLUMNS.items():
        if key not in files:
            raise ProblemError(f"Missing {key}.csv")
        df = pd.read_csv(io.BytesIO(files[key]), dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        miss = [c for c in cols if c not in df.columns and c in ("code", "name", "hours", "type", "strength",
                                                                  "capacity", "subjects", "lunch_period")]
        if miss:
            raise ProblemError(f"{key}.csv is missing column(s): {', '.join(miss)}")
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        frames[key] = df[cols]
    return frames_to_raw(frames, name, days, periods_per_day, break_periods)

def raw_to_zip(raw: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key, df in raw_to_frames(raw).items():
            z.writestr(f"{key}.csv", df.to_csv(index=False))
        z.writestr("dataset.json", json.dumps(raw, indent=2))
    return buf.getvalue()

def validate_raw(raw: dict):
    """Return (errors, warnings, stats). Errors block generation; warnings do not."""
    errs, warns, stats = [], [], {}
    days, P = raw.get("days", []), int(raw.get("periods_per_day", 0) or 0)
    if not days:
        errs.append("Add at least one working day.")
    if P < 1:
        errs.append("Periods per day must be at least 1.")
    for sec in ("subjects", "teachers", "classes", "rooms"):
        if not raw.get(sec):
            errs.append(f"Add at least one row to {sec}.")
    if errs:
        return errs, warns, stats

    def dupes(names, what):
        seen = set()
        for n in names:
            if n in seen:
                errs.append(f"Duplicate {what} name '{n}'.")
            seen.add(n)

    codes = [s["code"] for s in raw["subjects"]]
    dupes(codes, "subject code")
    dupes([t["name"] for t in raw["teachers"]], "teacher")
    dupes([c["name"] for c in raw["classes"]], "class")
    dupes([r["name"] for r in raw["rooms"]], "room")

    for s in raw["subjects"]:
        if not s["code"] or not s["name"]:
            errs.append("Every subject needs a code and a name.")
        if s["type"] not in ("theory", "lab"):
            errs.append(f"Subject {s['code']}: type must be 'theory' or 'lab'.")
        if s["hours"] < 1:
            errs.append(f"Subject {s['code']}: weekly hours must be at least 1.")
    for r in raw["rooms"]:
        if r["type"] not in ("classroom", "lab"):
            errs.append(f"Room {r['name']}: type must be 'classroom' or 'lab'.")
        if r["capacity"] < 1:
            errs.append(f"Room {r['name']}: capacity must be at least 1.")
    for c in raw["classes"]:
        if c["strength"] < 1:
            errs.append(f"Class {c['name']}: strength must be at least 1.")
        if not 1 <= c["lunch_period"] <= P:
            errs.append(f"Class {c['name']}: lunch period must be between 1 and {P}.")
        for code in c["subjects"]:
            if code not in codes:
                errs.append(f"Class {c['name']}: unknown subject '{code}'.")
    for t in raw["teachers"]:
        for code in t["subjects"]:
            if code not in codes:
                errs.append(f"Teacher {t['name']}: unknown subject '{code}'.")
        if t["max_load"] < 1:
            errs.append(f"Teacher {t['name']}: maximum load must be at least 1.")
        for sl in t.get("unavailable", []):
            try:
                parse_slot(sl, days, P)
            except ProblemError as e:
                errs.append(f"Teacher {t['name']}: {e}")
        for a in t.get("avoid_periods", []):
            if not 1 <= a <= P:
                errs.append(f"Teacher {t['name']}: avoid period {a} is outside 1..{P}.")
    if errs:
        return errs, warns, stats

    subj = {s["code"]: s for s in raw["subjects"]}
    breaks = len(set(raw.get("break_periods", [])))
    per_day = P - breaks
    for kind, rtype in (("theory", "classroom"), ("lab", "lab")):
        need = sum(subj[code]["hours"] for c in raw["classes"] for code in c["subjects"]
                   if subj[code]["type"] == kind)
        rooms = [r for r in raw["rooms"] if r["type"] == rtype]
        have = len(rooms) * len(days) * per_day
        stats[f"{kind}_hours"], stats[f"{rtype}_slots"] = need, have
        if need > have:
            errs.append(f"Not solvable: {need} {kind} hours are required but only {have} {rtype}-slots exist.")
        elif have and need / have > 0.85:
            warns.append(f"{kind.title()} rooms are {need / have:.0%} full - expect a hard search.")
        for c in raw["classes"]:
            if any(subj[x]["type"] == kind for x in c["subjects"]) and not any(r["capacity"] >= c["strength"] for r in rooms):
                errs.append(f"Class {c['name']} ({c['strength']} students) fits in no {rtype}.")
    for c in raw["classes"]:
        need = sum(subj[x]["hours"] for x in c["subjects"])
        have = len(days) * (per_day - (1 if c["lunch_period"] not in raw.get("break_periods", []) else 0))
        if need > have:
            errs.append(f"Class {c['name']} needs {need}h but has only {have} usable periods per week.")
    for code in {x for c in raw["classes"] for x in c["subjects"]}:
        if not any(code in t["subjects"] for t in raw["teachers"]):
            errs.append(f"No teacher can teach {code}.")
    if errs:
        return errs, warns, stats

    try:
        p = build_problem(raw)
    except ProblemError as e:
        errs.append(str(e))
        return errs, warns, stats
    stats.update(p.stats())
    load = {}
    for s in p.sessions:
        load[s.teacher] = load.get(s.teacher, 0) + s.duration
    for t, hrs in load.items():
        free = len(days) * per_day - len(p.t_unavail[t])
        if hrs > free:
            errs.append(f"{p.teacher_names[t]} must teach {hrs}h but is free for only {free} periods.")
        elif hrs / max(1, free) > 0.7:
            warns.append(f"{p.teacher_names[t]} is busy {hrs}/{free} available periods.")
    return errs, warns, stats
