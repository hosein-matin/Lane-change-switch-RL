# Rollout gates and privileged distillation for lane selection with sampled safety

Code for the paper

> H. Nick Zinat Matin, J. Sheng, X. Gong, J. Read, *Rollout Gates and Privileged Distillation for Lane Selection
> with Sampled Safety: When Is Run-Time Simulation Enough?*, submitted to the European Control Conference (ECC) 2027.

Repository: <https://github.com/hosein-matin/Lane-change-switch-RL>

The numerical companion (`docs/numerical_companion.pdf`) describes every experiment in detail: what each run
computes, how each number in the paper is obtained, and the result tables. This repository holds the code that
produced those numbers and the scripts that turn the raw results into the paper's tables and figure.

## Contents

| Folder | What it runs | Paper |
|---|---|---|
| `deterministic/` | Deterministic traffic (version D): longitudinal controller (PPO), lane-selection critic (Double DQN), exact gate, student, cascade, MOBIL, on-policy relabelling and episode-loss certification; two lanes (15 seeds) and three lanes (10 seeds) | Sec. V-B, first paragraph |
| `stochastic_S/` | Stochastic traffic at ν = 1 (package S): teacher labels, student, cascade, routers, Monte-Carlo gates with measured latency, privileged gate, certification (Corollary 3) | Sec. V-E (certification) |
| `stochastic_T/` | Package T, run on top of package S: delay sweep (0–7 steps), critic with the student's inputs, label budget, shifted traffic | Sec. V-B (delay), V-D |
| `stochastic_V/` | Uncertainty sweep ν ∈ {0, 0.5, 1, 2} (package V) and, in the same runner, packages W, W2, Z (information split), X (kernel filter and terminal-safe guards, Theorems 1–2), Y (frozen policies, stale-state sensitivity, decision-charge audit); compiled timing benchmark | Table I, Fig. 1, Sec. V-B, V-C, V-E |
| `analysis/` | Scripts that aggregate the raw results into the paper's numbers | all |
| `data/` | Summary exports of the paper's runs: `cert_perseed.csv` (seed, candidate, ε, n, J, γ, k, U, pass), `summary_final.json` (deterministic traffic), `summary_S.json` (packages S and T, including the label budget), `summary_V.json`, `value_of_sim.json`, `decomp.json` (sweep), `zsplit_*.json` (W, Z), `xfilt.json` (X), `yaudit.json`, `yfix_lat_*.json` (Y), `bench_compiled.json`, `fit.json`, `compute_V.json` (benchmark) | all |
| `docs/numerical_companion.pdf` | The numerical companion | all |

Each runner folder is self-contained: a resumable parallel runner (`run*.py`) and the `code/` it calls. The
stochastic folders bundle the trained longitudinal controllers (`ppo2/`, `ppo3/`), which the deterministic runners
train from scratch (`ppo.py`), so the stochastic packages can be run on their own.

## Setup

Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # numpy is all the simulations need; scipy/matplotlib for analysis; numba for the benchmark
```

## Running

Every runner has the same interface:

```bash
python <runner>.py [--lanes 2|3] --smoke --cores 4     # short end-to-end test on tiny settings (separate folder)
python <runner>.py [--lanes 2|3] --cores 16            # full run
python <runner>.py [--lanes 2|3] --status              # progress
python <runner>.py [--lanes 2|3] --only <text>         # only jobs whose name contains <text>, with their dependencies
```

Jobs write logs to `<results>/logs/` and a marker to `<results>/.done/` on success; re-running the same command
after an interruption skips finished jobs. Full runs take many core-hours; use as many cores as the machine has.

| Runner | Results folder |
|---|---|
| `deterministic/run_two_lanes.py` | `deterministic/results_cn` |
| `deterministic/run_three_lanes.py` | `deterministic/results3` |
| `stochastic_S/runS.py --lanes L` | `stochastic_S/results_sL` |
| `stochastic_T/runT.py --lanes L` | uses and extends `results_sL` (see below) |
| `stochastic_V/runV.py --lanes L` | `stochastic_V/results_vL/u<ν>/` |

**Package T** runs on the results of package S in the same folder: after `runS.py` has finished, copy
`stochastic_T/runT.py` and `stochastic_T/code/*.py` into `stochastic_S/` (replacing the files of the same name) and
run `python runT.py --lanes L --cores N` there. This is how the paper's runs were made.

**Benchmark** (Sec. V-C), on one core:

```bash
cd stochastic_V/code
python bench_compiled.py ../ppo2/ppo_s0/actor.pkl      # writes bench_compiled.json
```

What is timed: one Numba-compiled rollout (longitudinal network, the held-input filter of Proposition 1, OVM
followers, stochastic leader targets) for 3–80 vehicles and 5–30 s horizons; one untimed warm-up call per
configuration triggers compilation; then 200 repetitions (50 for more than 20 vehicles) timed with
`time.perf_counter`, each including the construction of the initial state; single process, Numba's default
single-threaded execution. The paper's timings (`data/bench_compiled.json`) were taken on an Apple-silicon laptop
under macOS 26.4.1 (arm64); the chip model and the Python/Numba versions were not recorded with them. The kernel
filter and plan search of Theorems 1–2 are not compiled and are not part of this benchmark; their NumPy timings
(companion, package X) come from the runs themselves.

## From raw results to the paper's numbers

See `analysis/README.md`: it lists, for each table and paragraph of the paper, the script and its inputs.

## License

- **Code** (everything except `docs/` and `data/`): PolyForm Strict License 1.0.0 (`LICENSE`). You may run the code
  for any noncommercial purpose, including research, teaching and reproducing the paper's results. You may not
  distribute it or make changes or new works based on it without the authors' permission. For other uses, contact
  the corresponding author.
- **Documentation and data** (`docs/`, `data/`): CC BY-NC-ND 4.0 (`LICENSE-docs-data`).

## Versions and checkpoints

The release tagged `v1.0` matches the submitted paper and version 11 of the companion. The trained critics and
students of the runs are not included; each runner retrains them, and the frozen-policy comparisons (packages X
and Y) use the critics trained in the same `runV.py` run.

## Citation

If you use this code, please cite the paper and the numerical companion (`CITATION.cff`).
