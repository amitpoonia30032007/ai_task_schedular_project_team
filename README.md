# AI-Based Timetable Scheduler (Compact Version)

This version shortens the supplied Python files by removing redundant comments, long docstrings, and excess blank lines. The algorithm implementations, function names, constraints, UI pages, and import relationships are retained.

## Project layout
- `app.py` — Streamlit interface
- `scheduler/problem.py` — CSP model, sessions, domains, teacher assignment
- `scheduler/state.py` — hard-conflict and soft-penalty bookkeeping
- `scheduler/analysis.py` — validation and result summaries for H1–H9
- `scheduler/backtracking.py` — plain and improved backtracking, MRV, degree, LCV, forward checking, AC-3
- `scheduler/min_conflicts.py` — Min-Conflicts local search and plateau escape
- `scheduler/annealing.py` — Simulated Annealing, Hill Climbing, Hybrid
- `scheduler/runner.py` — common algorithm entry point
- `scheduler/data_io.py` — JSON/CSV input, validation, data conversion
- `scheduler/experiments.py` — repeated runs and summary metrics
- `scheduler/export.py` — timetable grids, CSV and Excel export

## Run
Install the dependencies used by the project (`streamlit`, `pandas`, and `openpyxl`), place the `data/` folder containing `small_dataset.json` and `large_dataset.json` beside `app.py`, then run:

```bash
streamlit run app.py
```
```
if this doesn't work
try: python -m streamlit run app.py
```

The dataset files were not among the Python files uploaded for this edit, so they are not included in this compact archive.
While using json format file of dataset format of data should match as prescribe in data folder.
While using csv file 4 different files should be upload "classes.csv, subjects.csv,rooms.csv,teachers.csv" format should match individual format of each 
