"""Timetable grids (by class / teacher / room) and CSV / Excel export."""
from __future__ import annotations

import io
import re

import pandas as pd

VIEWS = ("Class", "Teacher", "Room")

def entities(p, view: str) -> list:
    return {"Class": p.class_names, "Teacher": p.teacher_names, "Room": p.room_names}[view]

def grid_cells(p, assignment: dict, view: str, idx: int) -> dict:
    """{(day, period): [(sid, is_first_period), ...]} for one class / teacher / room."""
    cells = {}
    for sid, (d, st, r) in assignment.items():
        s = p.sessions[sid]
        owner = {"Class": s.cls, "Teacher": s.teacher, "Room": r}[view]
        if owner != idx:
            continue
        for q in range(st, st + s.duration):
            cells.setdefault((d, q), []).append((sid, q == st))
    return cells

def cell_text(p, assignment: dict, sid: int, view: str) -> str:
    s = p.sessions[sid]
    r = assignment[sid][2]
    other = {"Class": f"{p.teacher_names[s.teacher]} | {p.room_names[r]}",
             "Teacher": f"{p.class_names[s.cls]} | {p.room_names[r]}",
             "Room": f"{p.class_names[s.cls]} | {p.teacher_names[s.teacher]}"}[view]
    return f"{s.subject} | {other}"

def grid_frame(p, assignment: dict, view: str, idx: int) -> pd.DataFrame:
    cells = grid_cells(p, assignment, view, idx)
    rows = []
    for q in range(p.n_periods):
        row = {}
        for d in range(p.n_days):
            items = cells.get((d, q), [])
            if items:
                row[p.days[d]] = " / ".join(cell_text(p, assignment, sid, view) for sid, _ in items)
            elif view == "Class" and q == p.lunch[idx]:
                row[p.days[d]] = "LUNCH"
            elif q in p.break_periods:
                row[p.days[d]] = "BREAK"
            else:
                row[p.days[d]] = ""
        rows.append(row)
    return pd.DataFrame(rows, index=[f"P{q + 1}" for q in range(p.n_periods)])

def sessions_frame(p, assignment: dict) -> pd.DataFrame:
    rows = []
    for sid in sorted(assignment):
        s = p.sessions[sid]
        d, st, r = assignment[sid]
        rows.append({
            "Class": p.class_names[s.cls], "Subject code": s.subject, "Subject": s.subject_name,
            "Type": s.kind, "Teacher": p.teacher_names[s.teacher], "Room": p.room_names[r],
            "Day": p.days[d], "Start period": st + 1, "End period": st + s.duration, "Periods": s.duration})
    cols = ["Class", "Subject code", "Subject", "Type", "Teacher", "Room", "Day", "Start period", "End period", "Periods"]
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["Class", "Day", "Start period"], key=lambda c: c.map({d: i for i, d in enumerate(p.days)}) if c.name == "Day" else c
    ).reset_index(drop=True)

def to_csv_bytes(p, assignment: dict) -> bytes:
    return sessions_frame(p, assignment).to_csv(index=False).encode("utf-8")

def _sheet(name: str, used: set) -> str:
    base = re.sub(r"[\[\]\*\?/\\:]", "-", name)[:27]
    out, i = base, 2
    while out in used:
        out, i = f"{base}~{i}", i + 1
    used.add(out)
    return out

def to_excel_bytes(p, assignment: dict) -> bytes:
    from openpyxl.styles import Alignment, Font, PatternFill

    buf = io.BytesIO()
    used = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        sessions_frame(p, assignment).to_excel(xl, sheet_name="All sessions", index=False)
        used.add("All sessions")
        for view in VIEWS:
            for i, name in enumerate(entities(p, view)):
                sheet = _sheet(f"{view[0]}-{name}", used)
                grid_frame(p, assignment, view, i).to_excel(xl, sheet_name=sheet)
                ws = xl.sheets[sheet]
                ws.column_dimensions["A"].width = 7
                for col in "BCDEFGHIJ"[: p.n_days]:
                    ws.column_dimensions[col].width = 30
                for cell in ws[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", fgColor="14213D")
                for row in ws.iter_rows(min_row=2, min_col=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws = xl.sheets["All sessions"]
        for col, w in zip("ABCDEFGHIJ", (12, 12, 28, 9, 22, 12, 7, 12, 11, 9)):
            ws.column_dimensions[col].width = w
    return buf.getvalue()
