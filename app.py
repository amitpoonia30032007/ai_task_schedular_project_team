"""Timetable Studio - Streamlit front end for the AI timetable scheduler."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from scheduler import export
from scheduler.analysis import HARD_NAMES, analyze
from scheduler.data_io import (COLUMNS, frames_to_raw, load_csvs, load_json, raw_to_frames, raw_to_zip,
                               split_list, validate_raw)
from scheduler.experiments import run_experiments
from scheduler.problem import ProblemError, build_problem
from scheduler.runner import ALGORITHMS, CORE, run_algorithm
from scheduler.state import WEIGHTS, State

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SAMPLES = {"Small sample (3 classes)": "small_dataset.json", "Large sample (8 classes)": "large_dataset.json"}

st.set_page_config(page_title="Timetable Studio", page_icon="🗓️", layout="wide")

PALETTE = [("#DBEAFE", "#1E3A8A"), ("#DCFCE7", "#14532D"), ("#FEF3C7", "#78350F"), ("#FFE4E6", "#881337"),
           ("#EDE9FE", "#4C1D95"), ("#CCFBF1", "#134E4A"), ("#FFEDD5", "#7C2D12"), ("#E0F2FE", "#0C4A6E"),
           ("#ECFCCB", "#365314"), ("#FCE7F3", "#831843"), ("#E2E8F0", "#1E293B"), ("#E0E7FF", "#312E81")]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, label { font-family: 'DM Sans', system-ui, sans-serif; }
h1, h2, h3, .ts-hero h1 { font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif !important; letter-spacing: -0.01em; color:#14213D; }
.block-container { padding-top: 1.4rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: #14213D; }
section[data-testid="stSidebar"] * { color: #E6EBF5 !important; }
section[data-testid="stSidebar"] .ts-side-card { background:#1D2D52; border-radius:10px; padding:.7rem .85rem; margin-top:.6rem; font-size:.86rem; line-height:1.45; }
.ts-hero { display:flex; justify-content:space-between; align-items:center; gap:1.5rem; background:#14213D; color:#fff;
  border-radius:14px; padding:1.3rem 1.6rem; margin-bottom:1.1rem; }
.ts-hero h1 { color:#fff !important; margin:0 0 .25rem 0; font-size:2rem; }
.ts-hero p { margin:0; color:#B9C4DD; max-width:46ch; font-size:.98rem; }
.ts-mini { display:grid; grid-template-columns: repeat(6, 22px); gap:4px; }
.ts-mini i { display:block; width:22px; height:22px; border-radius:5px; }
.ts-card { background:#fff; border:1px solid #E2E8F0; border-radius:10px; padding:.8rem 1rem; height:100%; }
.ts-card .k { color:#64748B; font-size:.82rem; }
.ts-card .v { font-family:'Bricolage Grotesque',sans-serif; font-size:1.7rem; font-weight:700; color:#14213D; line-height:1.2; }
.ts-card .h { color:#64748B; font-size:.78rem; }
.ts-card.good .v { color:#0F766E; } .ts-card.bad .v { color:#DC2626; }
.ts-pill { display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:700; font-size:.85rem; }
.ts-pill.ok { background:#CCFBF1; color:#115E59; } .ts-pill.no { background:#FEE2E2; color:#991B1B; }
.tt-wrap { overflow-x:auto; border:1px solid #E2E8F0; border-radius:10px; background:#fff; }
table.tt { border-collapse:collapse; width:100%; table-layout:fixed; min-width:720px; }
table.tt th { background:#14213D; color:#fff; padding:.55rem; font-weight:600; font-size:.9rem; }
table.tt th.tt-p { background:#F1F5F9; color:#14213D; width:52px; }
.tt-cell { border:1px solid #E8EDF3; height:74px; padding:3px; vertical-align:top; }
.tt-cell.tt-lunch { background: repeating-linear-gradient(45deg,#F1F5F9,#F1F5F9 8px,#E8EDF3 8px,#E8EDF3 16px);
  color:#64748B; text-align:center; vertical-align:middle; font-size:.85rem; }
.tt-cell.clash { outline:2px solid #DC2626; outline-offset:-2px; background:#FEF2F2; }
.tt-chip { border-left:4px solid; border-radius:6px; padding:3px 7px; font-size:.82rem; line-height:1.25; margin-bottom:2px; }
.tt-chip span { display:block; font-size:.72rem; opacity:.85; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tt-chip.cont { opacity:.7; font-size:.75rem; }
.ts-legend span { display:inline-block; margin:2px 6px 2px 0; padding:2px 9px; border-radius:6px; font-size:.8rem; border-left:4px solid; }
div.stButton > button[kind="primary"] { width:100%; background:#0F766E; border:0; font-weight:700; padding:.65rem 1rem; }
div.stButton > button[kind="primary"]:hover { background:#115E59; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def card(label, value, hint="", tone=""):
    return (f'<div class="ts-card {tone}"><div class="k">{html.escape(str(label))}</div>'
            f'<div class="v">{html.escape(str(value))}</div><div class="h">{html.escape(str(hint))}</div></div>')

def cards(items):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(card(*it), unsafe_allow_html=True)

def hero():
    mini = ["#DBEAFE", "#DCFCE7", "#FEF3C7", "#FFE4E6", "#EDE9FE", "#CCFBF1", "#FFEDD5"]
    pattern = [0, 1, 2, None, 3, 4, 5, None, 0, 6, 1, 2, None, 2, 3, 4, 5, 0, 6, 1, 2, None, 4, 3]
    cells = "".join(f'<i style="background:{mini[c] if c is not None else "#22345E"}"></i>' for c in pattern)
    st.markdown(f'<div class="ts-hero"><div><h1>Timetable Studio</h1>'
                f'<p>Load your classes, teachers and rooms, pick a search algorithm, and get a clash-free timetable.</p></div>'
                f'<div class="ts-mini">{cells}</div></div>', unsafe_allow_html=True)

def set_dataset(raw: dict) -> None:
    ss = st.session_state
    ss.problem = build_problem(raw)
    ss.raw = raw
    ss.ver = ss.get("ver", 0) + 1
    ss.result = ss.assignment = ss.generated = None
    ss.history, ss.runlog, ss.edit_ver, ss.excel = [], [], 0, None

def load_sample(fname: str) -> None:
    set_dataset(json.loads((DATA / fname).read_text()))

def init_state():
    if "raw" not in st.session_state:
        load_sample("small_dataset.json")

def colors_for(p):
    codes = sorted({s.subject for s in p.sessions})
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(codes)}

def show_issues(errs=(), warns=()):
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)

def grid_html(p, assignment, view, idx, conflicted, colors):
    cells = export.grid_cells(p, assignment, view, idx)
    head = "".join(f"<th>{html.escape(d)}</th>" for d in p.days)
    rows = []
    for q in range(p.n_periods):
        tds = []
        for d in range(p.n_days):
            items = cells.get((d, q), [])
            if items:
                clash = len(items) > 1 or any(sid in conflicted for sid, _ in items)
                inner = ""
                for sid, first in items:
                    s = p.sessions[sid]
                    bg, fg = colors[s.subject]
                    style = f"background:{bg};color:{fg};border-left-color:{fg}"
                    r = assignment[sid][2]
                    a, b = {"Class": (p.teacher_names[s.teacher], p.room_names[r]),
                            "Teacher": (p.class_names[s.cls], p.room_names[r]),
                            "Room": (p.class_names[s.cls], p.teacher_names[s.teacher])}[view]
                    if first:
                        inner += (f'<div class="tt-chip" style="{style}"><b>{html.escape(s.subject)}</b>'
                                  f'{" (lab)" if s.kind == "lab" else ""}<span>{html.escape(a)}</span>'
                                  f'<span>{html.escape(b)}</span></div>')
                    else:
                        inner += f'<div class="tt-chip cont" style="{style}">↳ {html.escape(s.subject)} continues</div>'
                tds.append(f'<td class="tt-cell{" clash" if clash else ""}">{inner}</td>')
            elif view == "Class" and q == p.lunch[idx]:
                tds.append('<td class="tt-cell tt-lunch">Lunch</td>')
            elif q in p.break_periods:
                tds.append('<td class="tt-cell tt-lunch">Break</td>')
            else:
                tds.append('<td class="tt-cell"></td>')
        rows.append(f'<tr><th class="tt-p">P{q + 1}</th>{"".join(tds)}</tr>')
    return (f'<div class="tt-wrap"><table class="tt"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')

def legend_html(colors, p):
    names = {s.subject: s.subject_name for s in p.sessions}
    return '<div class="ts-legend">' + "".join(
        f'<span style="background:{bg};color:{fg};border-left-color:{fg}">{html.escape(c)} · {html.escape(names[c])}</span>'
        for c, (bg, fg) in colors.items()) + "</div>"

def page_data():
    ss = st.session_state
    st.subheader("Your data")
    st.caption("Start from a sample, upload a file, or edit the tables below. Nothing is used until you press "
               "**Check and use these tables**.")

    src = st.radio("Where does your data come from?",
                   ["Sample dataset", "Upload JSON", "Upload CSV files"], horizontal=True)
    if src == "Sample dataset":
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox("Sample", list(SAMPLES), label_visibility="collapsed")
        if c2.button("Load sample"):
            load_sample(SAMPLES[pick])
            st.toast("Sample loaded")
            st.rerun()
    elif src == "Upload JSON":
        f = st.file_uploader("JSON file", type=["json"])
        if f is not None and st.button("Use this JSON file"):
            try:
                raw = load_json(f.getvalue())
                errs, warns, _ = validate_raw(raw)
                if errs:
                    show_issues(errs, warns)
                else:
                    set_dataset(raw)
                    st.rerun()
            except (ValueError, KeyError, ProblemError) as e:
                st.error(f"Could not read the file: {e}")
    else:
        st.caption("Upload subjects.csv, teachers.csv, classes.csv and rooms.csv. Separate list values with ';' "
                   "(e.g. `MA;PH`). Slots look like `Mon-1`. Set the time structure in the form below first.")
        files = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True)
        if files and st.button("Use these CSV files"):
            try:
                by = {}
                for f in files:
                    for key in COLUMNS:
                        if key[:-1] in f.name.lower():
                            by[key] = f.getvalue()
                ver = ss.ver
                raw = load_csvs(by, ss.get(f"name_in_{ver}", "CSV dataset"),
                                [d.strip() for d in ss.get(f"days_in_{ver}", "Mon,Tue,Wed,Thu,Fri").split(",") if d.strip()],
                                int(ss.get(f"ppd_in_{ver}", 6)),
                                [int(x) for x in split_list(ss.get(f"brk_in_{ver}", "")) if x.isdigit()])
                errs, warns, _ = validate_raw(raw)
                if errs:
                    show_issues(errs, warns)
                else:
                    set_dataset(raw)
                    st.rerun()
            except (ValueError, KeyError, ProblemError) as e:
                st.error(f"Could not read the CSV files: {e}")

    raw, v = ss.raw, ss.ver
    st.divider()
    st.markdown("##### Edit as a form")
    t1, t2, t3, t4 = st.columns([2, 3, 1.2, 1.5])
    name = t1.text_input("Dataset name", raw["name"], key=f"name_in_{v}")
    days_txt = t2.text_input("Working days (comma separated)", ", ".join(raw["days"]), key=f"days_in_{v}")
    ppd = t3.number_input("Periods per day", 1, 12, int(raw["periods_per_day"]), key=f"ppd_in_{v}")
    brk = t4.text_input("Break periods (e.g. 3)", ", ".join(str(b) for b in raw.get("break_periods", [])), key=f"brk_in_{v}")

    fr = raw_to_frames(raw)
    tabs = st.tabs(["Subjects", "Teachers", "Classes", "Rooms"])
    edited = {}
    cfg = {
        "subjects": {"type": st.column_config.SelectboxColumn("type", options=["theory", "lab"], required=True),
                     "hours": st.column_config.NumberColumn("weekly hours", min_value=1, max_value=20, step=1)},
        "teachers": {"max_load": st.column_config.NumberColumn("max weekly load", min_value=1, step=1),
                     "subjects": st.column_config.TextColumn("subjects they teach", help="codes, e.g. MA, PH"),
                     "unavailable": st.column_config.TextColumn("unavailable slots", help="Day-Period, e.g. Mon-1, Tue-3"),
                     "avoid_periods": st.column_config.TextColumn("avoid periods", help="soft preference, e.g. 1")},
        "classes": {"strength": st.column_config.NumberColumn("strength", min_value=1, step=1),
                    "lunch_period": st.column_config.NumberColumn("lunch period", min_value=1, step=1),
                    "subjects": st.column_config.TextColumn("subjects", help="codes, e.g. MA, PH, CPL")},
        "rooms": {"type": st.column_config.SelectboxColumn("type", options=["classroom", "lab"], required=True),
                  "capacity": st.column_config.NumberColumn("capacity", min_value=1, step=1)},
    }
    for tab, key in zip(tabs, ["subjects", "teachers", "classes", "rooms"]):
        with tab:
            edited[key] = st.data_editor(fr[key], key=f"ed_{key}_{v}", num_rows="dynamic",
                                         column_config=cfg[key], hide_index=True)

    if st.button("Check and use these tables", type="primary"):
        try:
            new = frames_to_raw(edited, name, [d.strip() for d in days_txt.split(",") if d.strip()], ppd,
                                [int(x) for x in split_list(brk) if x.isdigit()])
            errs, warns, _ = validate_raw(new)
        except Exception as e:
            errs, warns, new = [f"Could not read the tables: {e}"], [], None
        if errs:
            show_issues(errs, warns)
        else:
            set_dataset(new)
            st.toast("Data applied")
            st.rerun()

    st.divider()
    st.markdown("##### Is this dataset solvable?")
    errs, warns, stats = validate_raw(raw)
    show_issues(errs, warns)
    if not errs:
        cards([("Sessions to place", stats["sessions"], "lectures + lab blocks"),
               ("Classes", stats["classes"], ""), ("Teachers", stats["teachers"], ""),
               ("Rooms", stats["rooms"], ""), ("Average choices", stats["avg_domain"], "slots x rooms per session")])
        c1, c2 = st.columns(2)
        for col, kind, rt in ((c1, "theory", "classroom"), (c2, "lab", "lab")):
            need, have = stats[f"{kind}_hours"], stats[f"{rt}_slots"]
            col.progress(min(1.0, need / max(1, have)), text=f"{kind.title()} hours {need} of {have} {rt}-slots "
                                                              f"({need / max(1, have):.0%})")
    d1, d2 = st.columns(2)
    d1.download_button("Download dataset (JSON)", json.dumps(raw, indent=2), f"{raw['name'][:20]}.json", "application/json")
    d2.download_button("Download dataset (CSV zip)", raw_to_zip(raw), "dataset_csv.zip", "application/zip")

BLURB = {
    "Backtracking (plain)": "Baseline systematic search. Tries values in order, backs up on a clash. Can be very slow on big inputs.",
    "Backtracking + heuristics": "Systematic search with MRV, degree heuristic and forward checking. Complete and usually fast.",
    "Min-Conflicts": "Local search. Starts from a full (clashing) timetable and repeatedly moves one clashing session to its best slot.",
    "Simulated Annealing": "Optimiser. Random moves and swaps, accepting some worse ones early on to escape local optima.",
    "Hill Climbing": "Like annealing but only accepts moves that do not make things worse.",
    "Hybrid (Backtracking + SA)": "Backtracking finds a feasible timetable, then annealing improves its soft-constraint quality.",
}

def algo_params(alg):
    prm = {}
    with st.expander("Advanced settings"):
        if alg == "Backtracking + heuristics":
            c = st.columns(5)
            prm["mrv"] = c[0].checkbox("MRV", True, key="h_mrv")
            prm["degree"] = c[1].checkbox("Degree", True, key="h_deg")
            prm["forward_checking"] = c[2].checkbox("Forward checking", True, key="h_fc")
            prm["lcv"] = c[3].checkbox("Least constraining value", False, key="h_lcv")
            prm["ac3"] = c[4].checkbox("AC-3 first", False, key="h_ac3")
        elif alg == "Min-Conflicts":
            c = st.columns(2)
            prm["max_iters"] = int(c[0].number_input("Iteration limit", 100, 500000, 20000, 1000, key="m_it"))
            prm["restart_after"] = int(c[1].number_input("Partial restart after (stalled iterations)", 50, 50000, 1500, key="m_rs"))
            prm["p_walk"] = c[0].slider("Random-walk probability", 0.0, 0.5, 0.05, 0.01, key="m_pw")
            prm["init"] = c[1].selectbox("Starting timetable", ["random", "greedy"], key="m_init")
            prm["tie_break"] = c[0].selectbox("Tie-breaking", ["soft", "random"], key="m_tb",
                                              help="soft = among equally good slots, pick the one with the lowest soft penalty")
        elif alg == "Simulated Annealing":
            c = st.columns(3)
            prm["iters"] = int(c[0].number_input("Iterations", 1000, 1000000, 60000, 5000, key="s_it"))
            prm["t0"] = float(c[1].number_input("Start temperature", 0.5, 500.0, 10.0, key="s_t0"))
            prm["swap_prob"] = c[2].slider("Swap move probability", 0.0, 1.0, 0.3, key="s_sw")
        elif alg == "Hill Climbing":
            prm["iters"] = int(st.number_input("Iterations", 1000, 1000000, 60000, 5000, key="hc_it"))
        elif alg == "Hybrid (Backtracking + SA)":
            c = st.columns(2)
            prm["sa_iters"] = int(c[0].number_input("Annealing iterations", 1000, 1000000, 60000, 5000, key="hy_it"))
            prm["t0"] = float(c[1].number_input("Annealing start temperature", 0.5, 100.0, 5.0, key="hy_t0"))
        else:
            st.caption("Plain backtracking has no settings. It is the baseline for the comparison.")
    return prm

def trace_chart(res):
    df = pd.DataFrame({"iteration": range(len(res.trace)), "cost": [max(1, c) for c in res.trace]})
    try:
        import altair as alt
        st.altair_chart(alt.Chart(df).mark_line(color="#0F766E").encode(
            x=alt.X("iteration:Q"), y=alt.Y("cost:Q", scale=alt.Scale(type="log"), title="cost (log scale)")
        ).properties(height=260))
    except ImportError:
        st.line_chart(df.set_index("iteration"))

def show_result(p, res):
    ok = res.success
    st.markdown(f'<span class="ts-pill {"ok" if ok else "no"}">'
                f'{"Valid timetable" if ok else f"{res.hard} hard violation(s)"}</span> '
                f'&nbsp;{html.escape(res.algorithm)} · {html.escape(str(res.info.get("status", "")))}', unsafe_allow_html=True)
    st.write("")
    steps = "Nodes expanded" if res.algorithm.startswith("Backtracking") else "Iterations"
    cards([("Time taken", f"{res.seconds:.3f} s", ""), (steps, f"{res.steps:,}", ""),
           ("Hard violations", res.hard, "must be 0", "good" if res.hard == 0 else "bad"),
           ("Soft penalty", res.soft, "lower is nicer"), ("Total cost", f"{res.cost:,}", "1000 x hard + soft")])
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("**Soft constraints**")
        parts = res.analysis.soft_parts
        st.dataframe(pd.DataFrame([
            {"constraint": "S1 gaps in a class's day", "count": parts["S1"], "weight": WEIGHTS["S1"]},
            {"constraint": "S2 same subject twice a day", "count": parts["S2"], "weight": WEIGHTS["S2"]},
            {"constraint": "S3 lab periods consecutive", "count": 0, "weight": "built in"},
            {"constraint": "S4 teacher preferences", "count": parts["S4"], "weight": WEIGHTS["S4"]},
            {"constraint": "S5 over 2 lectures in a row", "count": parts["S5"], "weight": WEIGHTS["S5"]}]),
            hide_index=True)
    with b:
        st.markdown("**Search details**")
        info = {k: v for k, v in res.info.items() if k != "status"}
        st.dataframe(pd.DataFrame({"measure": list(info), "value": [str(v) for v in info.values()]}), hide_index=True)
    if res.trace and len(res.trace) > 2:
        st.markdown("**Cost against iterations**")
        trace_chart(res)
    if res.analysis.issues:
        with st.expander(f"Hard-constraint violations ({res.hard})", expanded=True):
            for it in res.analysis.issues[:40]:
                st.write(f"**{it['code']}** {HARD_NAMES[it['code']]}: {it['text']}")

def page_generate():
    ss, p = st.session_state, st.session_state.problem
    st.subheader("Generate a timetable")
    left, right = st.columns([2, 1])
    with left:
        alg = st.selectbox("Algorithm", ALGORITHMS)
        st.caption(BLURB[alg])
        prm = algo_params(alg)
    with right:
        seed = st.number_input("Random seed", 0, 10 ** 6, 42, help="Same seed = same result")
        tl = st.slider("Time limit (seconds)", 1, 120, 30)
    if st.button("Generate timetable", type="primary"):
        with st.spinner(f"Running {alg} on {len(p.sessions)} sessions..."):
            res = run_algorithm(p, alg, seed=int(seed), time_limit=float(tl), **prm)
        ss.result, ss.assignment, ss.generated, ss.history = res, dict(res.assignment), dict(res.assignment), []
        ss.edit_ver += 1
        ss.excel = None
        ss.runlog.append({"dataset": p.name[:18], "algorithm": res.algorithm, "seed": int(seed),
                          "time (s)": round(res.seconds, 3), "steps": res.steps, "hard": res.hard, "soft": res.soft})
    if ss.result is not None:
        st.divider()
        show_result(p, ss.result)
        if ss.result.hard == 0:
            st.success("Open the **Timetable** page to see the grids, or **Edit** to move sessions by hand.")
    if ss.runlog:
        with st.expander("Runs in this session"):
            st.dataframe(pd.DataFrame(ss.runlog), hide_index=True)

def page_timetable():
    ss, p = st.session_state, st.session_state.problem
    st.subheader("Timetable")
    if not ss.assignment:
        st.info("No timetable yet. Go to **Generate** and press *Generate timetable*.")
        return
    an = analyze(p, ss.assignment)
    if an.hard_total:
        st.error(f"{an.hard_total} hard-constraint violation(s). Clashing cells have a red outline.")
    c1, c2 = st.columns([1, 2])
    view = c1.radio("View by", list(export.VIEWS), horizontal=True)
    names = export.entities(p, view)
    idx = c2.selectbox(view, range(len(names)), format_func=lambda i: names[i])
    colors = colors_for(p)
    st.markdown(grid_html(p, ss.assignment, view, idx, an.conflicted, colors), unsafe_allow_html=True)
    st.markdown(legend_html(colors, p), unsafe_allow_html=True)

    cells = export.grid_cells(p, ss.assignment, view, idx)
    used = len(cells)
    if view == "Teacher":
        st.caption(f"{names[idx]} teaches {used} periods per week (maximum {p.teacher_max_load[idx]}).")
    elif view == "Class":
        st.caption(f"{names[idx]} has {used} periods per week; {p.n_days * sum(p.usable[idx]) - used} free periods.")
    else:
        st.caption(f"{names[idx]} is in use for {used} of {p.n_days * p.n_periods} periods "
                   f"({used / (p.n_days * p.n_periods):.0%}).")

    st.divider()
    st.markdown("##### Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download CSV", export.to_csv_bytes(p, ss.assignment), "timetable.csv", "text/csv")
    if e2.button("Prepare Excel workbook"):
        ss.excel = export.to_excel_bytes(p, ss.assignment)
    if ss.get("excel"):
        e3.download_button("Download Excel", ss.excel, "timetable.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def page_edit():
    ss, p = st.session_state, st.session_state.problem
    st.subheader("Edit by hand")
    if ss.assignment is None:
        st.info("Generate a timetable first, then adjust it here.")
        return
    cur = analyze(p, ss.assignment)
    cards([("Hard violations", cur.hard_total, "now", "good" if cur.hard_total == 0 else "bad"),
           ("Soft penalty", cur.soft_total, "now"), ("Unscheduled sessions", len(cur.unscheduled), "")])
    st.write("")

    ci = st.selectbox("Class", range(len(p.class_names)), format_func=lambda i: p.class_names[i])
    sids = [s.sid for s in p.sessions if s.cls == ci]

    def fmt(sid):
        where = p.describe(sid, ss.assignment[sid]) if sid in ss.assignment else "not scheduled"
        return f"{p.label(sid, short=True)} · {p.teacher_names[p.sessions[sid].teacher]} · {where}"
    sid = st.selectbox("Session to move", sids, format_func=fmt)
    s = p.sessions[sid]
    old = ss.assignment.get(sid, p.domains[sid][0])
    key = f"{sid}_{ss.edit_ver}"
    c1, c2, c3 = st.columns(3)
    d = c1.selectbox("Day", range(p.n_days), index=old[0], format_func=lambda i: p.days[i], key=f"d_{key}")
    starts = list(range(p.n_periods - s.duration + 1))
    st0 = c2.selectbox("Starting period", starts, index=min(old[1], starts[-1]), format_func=lambda i: f"P{i + 1}",
                       key=f"s_{key}")
    r = c3.selectbox("Room", range(len(p.room_names)), index=old[2], key=f"r_{key}",
                     format_func=lambda i: f"{p.room_names[i]} · {p.room_kinds[i]} · {p.room_caps[i]} seats")
    new = (d, st0, r)

    trial = dict(ss.assignment)
    trial[sid] = new
    ta = analyze(p, trial)
    mine = [it for it in ta.issues if sid in it["sids"]]
    if sid in ss.assignment and new == ss.assignment[sid]:
        st.info("This is the current position.")
    elif mine:
        st.error(f"This move creates {sum(it['count'] for it in mine)} hard-constraint conflict(s):")
        for it in mine:
            st.write(f"**{it['code']}** {HARD_NAMES[it['code']]}: {it['text']}")
    else:
        st.success("No hard-constraint conflicts. This is a legal position.")
    st.caption(f"Effect of the move: hard {cur.hard_total} to {ta.hard_total}, soft {cur.soft_total} to {ta.soft_total}")

    b1, b2, b3 = st.columns(3)
    if b1.button("Apply move", type="primary"):
        ss.history.append(dict(ss.assignment))
        ss.assignment[sid] = new
        ss.edit_ver += 1
        ss.excel = None
        st.rerun()
    if b2.button("Undo last move", disabled=not ss.history):
        ss.assignment = ss.history.pop()
        ss.edit_ver += 1
        st.rerun()
    if b3.button("Reset to generated", disabled=ss.generated is None):
        ss.assignment, ss.history = dict(ss.generated), []
        ss.edit_ver += 1
        st.rerun()

    with st.expander("Conflict-free positions for this session"):
        base = State(p)
        for k, v in ss.assignment.items():
            if k != sid:
                base.add(k, v)
        options = []
        for v in p.domains[sid]:
            if base.value_conflicts(sid, v) == 0:
                base.add(sid, v)
                options.append({"day": p.days[v[0]], "period": f"P{v[1] + 1}", "room": p.room_names[v[2]], "soft penalty": base.soft})
                base.remove(sid)
        options.sort(key=lambda o: o["soft penalty"])
        if options:
            st.dataframe(pd.DataFrame(options[:12]), hide_index=True)
        else:
            st.write("No free position exists for this session. Move another session first.")

    if cur.issues:
        with st.expander(f"All current violations ({cur.hard_total})"):
            for it in cur.issues[:60]:
                st.write(f"**{it['code']}** {HARD_NAMES[it['code']]}: {it['text']}")

def page_experiments():
    ss, p = st.session_state, st.session_state.problem
    st.subheader("Compare algorithms")
    st.caption("Runs every selected algorithm on every selected dataset with seeds 42, 43, ... so results are reproducible.")
    c1, c2 = st.columns(2)
    sets = c1.multiselect("Datasets", ["Small sample", "Large sample", "Current dataset"], ["Small sample", "Large sample"])
    algs = c2.multiselect("Algorithms", ALGORITHMS, CORE)
    c3, c4, c5 = st.columns(3)
    runs = c3.slider("Runs per algorithm", 1, 10, 10)
    tl = c4.slider("Time limit per run (s)", 1, 60, 10)
    seed = c5.number_input("First seed", 0, 10 ** 6, 42)
    if st.button("Run experiments", type="primary"):
        probs = {}
        for name, fname in (("Small sample", "small_dataset.json"), ("Large sample", "large_dataset.json")):
            if name in sets:
                probs[name] = build_problem(json.loads((DATA / fname).read_text()))
        if "Current dataset" in sets:
            probs["Current dataset"] = p
        if not probs or not algs:
            st.warning("Pick at least one dataset and one algorithm.")
            return
        bar = st.progress(0.0, text="Starting")
        summary, raw, traces = run_experiments(probs, algs, runs, float(tl), int(seed),
                                               progress=lambda f, m: bar.progress(min(1.0, f), text=m))
        bar.empty()
        ss.exp = (summary, raw, traces)
    if ss.get("exp"):
        summary, raw, traces = ss.exp
        st.markdown("##### Results (averages over runs)")
        st.dataframe(summary.rename(columns={
            "time_s": "time (s)", "nodes_or_iterations": "nodes / iterations", "hard_violations": "hard violations",
            "soft_penalty": "soft penalty", "success_rate": "success rate"}), hide_index=True)
        st.download_button("Download summary (CSV)", summary.to_csv(index=False), "summary.csv", "text/csv")
        for dname in summary["dataset"].unique():
            sub = summary[summary["dataset"] == dname]
            fastest = sub.loc[sub["time_s"].idxmin(), "algorithm"]
            nicest = sub.loc[sub["soft_penalty"].idxmin(), "algorithm"]
            st.write(f"**{dname}:** fastest on average was *{fastest}*; lowest soft penalty was *{nicest}*.")
        keys = [k for k in traces if k[1] == "Min-Conflicts"]
        if keys:
            st.markdown("##### Min-Conflicts: cost against iterations")
            pick = st.selectbox("Dataset", [k[0] for k in keys])
            tr = traces[(pick, "Min-Conflicts")]
            trace_chart(type("T", (), {"trace": tr})())

def page_about():
    st.subheader("How it works")
    st.markdown(f"""
**Variables.** One session: a single lecture period, or a lab block of 2 periods. A class with 4 weekly hours of Maths has 4 variables.

**Domain.** Every (day, start period, room) that already passes the single-session rules: room type (H5), capacity (H4), teacher availability (H6), and lunch or break periods (H9).
The teacher is fixed in advance for each class and subject, which also settles the teacher load limit (H8).

**Binary constraints.** No teacher (H1), class (H2) or room (H3) may have two sessions in the same period. H7 holds because every session must be placed; an unplaced session counts as a violation.

**Cost.** `1000 x hard violations + soft penalty`, where soft penalty = {WEIGHTS['S1']} x gaps (S1) + {WEIGHTS['S2']} x repeated subjects in a day (S2) + {WEIGHTS['S4']} x teacher-preference periods (S4) + {WEIGHTS['S5']} x periods beyond 2 in a row (S5).
S3 is built in because a lab is one block of consecutive periods.

**Algorithms.** Backtracking (plain, and with MRV, degree, LCV, forward checking and AC-3), Min-Conflicts with random walk and partial restarts,
plus Simulated Annealing, Hill Climbing and a Hybrid of Backtracking and Annealing.
""")

PAGES = {"📥  Data": page_data, "⚙️  Generate": page_generate, "🗓️  Timetable": page_timetable,
         "✏️  Edit": page_edit, "📊  Experiments": page_experiments, "ℹ️  How it works": page_about}

def main():
    init_state()
    ss = st.session_state
    with st.sidebar:
        st.markdown("### 🗓️ Timetable Studio")
        page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
        p = ss.problem
        status = "No timetable yet"
        if ss.get("assignment"):
            a = analyze(p, ss.assignment)
            status = "Valid timetable" if a.hard_total == 0 else f"{a.hard_total} hard violation(s)"
        st.markdown(f'<div class="ts-side-card"><b>{html.escape(p.name[:34])}</b><br>{len(p.sessions)} sessions · '
                    f'{len(p.class_names)} classes · {len(p.teacher_names)} teachers · {len(p.room_names)} rooms'
                    f'<br>{status}</div>', unsafe_allow_html=True)
    hero()
    PAGES[page]()

main()
