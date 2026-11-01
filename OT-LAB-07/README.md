# Lab 7 — Multiobjective Optimization: 37 Algorithms, Trade-offs and Applications

**Yash Deshmukh · VJTI · Roll 241070018** · Assigned deep-dive algorithms (R = 18): **18 MOEA/D, 30 MOMA, 5 Value Function**

All 37 algorithms are implemented in NumPy only (scikit-learn is used just to load the breast-cancer data for P9).

## Contents
| Path | What |
|---|---|
| `notebooks/Lab7_MOO.ipynb` | **Main deliverable** – executed notebook, Tasks A–F, runs top to bottom (~11 min) |
| `src/a00_utils.py` … `a13_taskF.py` | Notebook source split by section (`# %%` cells) |
| `tools/build.py` | Builds the notebook from `src/` |
| `tools/run_all.sh` | Runs all sources as one script (`LAB7_QUICK=1` for a ~2 min smoke test) |
| `tools/exec_nb.py` | Executes the notebook with nbclient |

## Task map
| Task | Source | Content |
|---|---|---|
| A | `a00`–`a08` | `moo_utils` (dominance, NDS, crowding, HV, IGD, Spacing), problems P1–P14, algorithms 1–37 with docstrings, unit tests, plots, implementation log |
| B | `a09` | 37-row comparison table (pros/cons with evidence) + family paragraph |
| C | `a10` | Deep dive: study cards, two-iteration hand traces reproduced by code, 3-value × 10-seed sensitivity |
| D | `a11` | Head-to-head (10 seeds): P3, P5, P10 (M=5), P8, TOPSIS/VIKOR on NSGA-II front |
| E | `a12` | Algorithm selection for six scenarios |
| F | `a13` | Viva questions with demos |

## Run
```bash
pip install -r requirements.txt
# quick smoke test (small budgets)
cd tools && LAB7_QUICK=1 bash run_all.sh
# rebuild + execute the notebook (full budget: 10,000 evaluations, seeds 0–9)
python build.py && python exec_nb.py   # also regenerates results/
```
Or upload `notebooks/Lab7_MOO.ipynb` to Google Colab and **Run all**.

## Setup used
Budget 10,000 evaluations (P9: 3,000) · seeds 0–9 · metrics: Hypervolume (ref point 1.1 in normalised space), IGD, Spacing.

## Raw results (`results/`)
Everything behind every table and plot is saved here (written by `src/a14_export.py` at the end of the notebook).
| Path | Content |
|---|---|
| `results/metrics_per_run_seed0.csv` | HV / IGD / Spacing / points / time per (algorithm, problem) run of the implementation log |
| `results/fronts/` | obtained fronts (`algNN_Pk_objectives.csv`), decision vectors (`*_decision.csv`), reference fronts (`reference_*.csv`) and every per-seed front of Tasks C/D |
| `results/per_seed/` | per-seed metrics for D.1 (P3, P5), D.2 (P10 M=5), D.3 (P8 TSP), C.3 sensitivity (MOEA/D T, MOMA p_ls) |
| `results/D4_P4_front_points_risk_return.csv` | the 20 front points ranked by TOPSIS/VIKOR |
| `results/study_numbers.json` | numbers measured in the problem studies used as evidence in Task B/E |
| `results/tables/` | every table printed in the notebook as CSV (named by notebook cell) |
| `results/figures/` | every plot in the notebook as PNG (named by notebook cell) |

