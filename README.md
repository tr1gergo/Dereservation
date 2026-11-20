# Efficient de-reservation in matching markets

This study compares reserve mechanisms in two synthetic settings, a stylized
Indian vertical reservation market and a market with overlapping eligibility.
The paper profile contains 32 settings, with 100 paired market draws per setting,
600 students per market, and 20 institutions. Each setting compares six methods.

The [notebook](dereservation_study/experiments.ipynb) regenerates the reports from the bundled runs.
The [manifest](dereservation_study/data/paper/manifest.json) records the full design,
seed, software versions, and source hash of the recorded runs.

## Reproduce the reports

From the repository root, create a Python environment and select it as the
Jupyter kernel. The study uses Python 3.10 or newer.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source
.venv/bin/activate` in Bash, then run:

```sh
python -m pip install -r dereservation_study/requirements.txt
jupyter lab dereservation_study/experiments.ipynb
```

Run all cells with the default `paper` profile. A fresh clone reads the compressed
archive in `dereservation_study/data/paper/`. An existing local run can supply the same observations.
The notebook loads the main results, examples, local compliance frontiers, and
MD timing benchmark, then writes derived tables and PNG/SVG figures to
`dereservation_study/results/notebook_paper/reports/`. The mechanisms run only when a requested
profile lacks complete saved results. Smoke and quick profiles use smaller
markets for new experiments. Keep `RESUME = True` to reuse saved results.

To generate the two main sweep reports from the command line:

```sh
python -m dereservation_study --profile paper
```

For an independent small run, use `python -m dereservation_study --profile smoke`.
The `--output` option sets a separate working directory. Use `--no-resume` to
measure new runs and new timings for the main study. Numerical checkpoints record
the design, algorithm definitions, core source fingerprint, and environment.
Reporting reads the archived provenance and validates the requested data coverage.
The study has no commercial solver dependency.

## Mechanisms

| Method | Setting | Institutional choice |
|---|---|---|
| DA | Indian | Open-first student-proposing deferred acceptance with original quotas |
| Multi-run DA | Indian | Convert vacant OBC quota to open quota between full DA runs |
| BT | Indian | Reset original quotas for each choice, then perform backward transfers inside that choice |
| DA-open | Overlapping | Accept up to capacity in overall school priority order |
| Smart-Hard-Reserves | Overlapping | Greedily retain students in priority order when the cohort fits the original open and reserved slots |
| Soft-Reserves | Overlapping | Reject by demographic soft bounds, falling back to school priority when needed |
| MD | Both | Maximize headcount, then the fixed lexicographic order of category-priority claims |
| RP | Both | Maximize headcount, then seats used by original eligibles, then the fixed claims order |
| Merit | Both | Maximize headcount, then total cohort merit, then the fixed claims order |

The Indian design permits transfers only from OBC. The overlapping design varies
the dependence of three reserved types while holding each marginal probability
at 0.3. The incumbent overlapping baselines use overall school priority. MD, RP,
and Merit assign individual seats using fixed integer weights and alternating-path
updates. Merit can favor a higher-ranked cohort while creating envy under the
extended category priorities. The Indian BT comparison follows the procedure in
the [Aygün and Turhan paper](https://doi.org/10.1287/mnsc.2022.4566).

## Archived observations

The compressed CSVs contain the observations for the reported mechanisms.
`pandas.read_csv` reads them directly. Tables join on `instance_id` and
`algorithm`. Each setting reuses the same seed schedule across mechanisms.

| Table in `data/paper/` | Unit and contents |
|---|---|
| `outcomes.csv.gz` | 19,200 market outcomes with welfare, seat-use, feasibility, blocking, demographic, and timing metrics |
| `groups.csv.gz` | Group populations, admission counts and rates, representation, and ranks |
| `institutions.csv.gz` | Institution capacity, matched students, and protected admissions |
| `welfare.csv.gz` | Within-market counts and shares of students who improve or worsen relative to BT or DA-open |
| `runtime_engines.csv` | 200 timings from paired trials of incremental MD and recomputed-flow MD |
| `paper_example_2_1.csv`, `*_local_frontier.csv` | Small examples and exact local compliance frontiers |

- `average_rank` is one-based. An unmatched student receives list length plus
  one. `average_rank_matched` conditions on being assigned.
- `original_reserved_seats` counts reserved seats occupied by original
  eligibles. `dereserved_seats` counts seats occupied through additional
  eligibility. Administrative OBC quota changes have separate fields.
- The overlapping baselines return cohorts. Their seat labels come from a
  deterministic supporting assignment. `original_reserved_seats_max_for_cohort`
  gives the largest original-seat count attainable holding the cohort fixed.
  Default summaries exclude unsupported cohorts and report their incidence.
- `blocking_pairs` tests whether a preferred institution chooses a student
  added to its current cohort. `category_*` diagnostics instead use reported
  seat labels and extended eligibility. `merit_*` uses overall school priority.
  Envy and vacancies count student–institution pairs.
- `minority_matched` counts unique students across protected types.
  `minority_share_matched` divides by all students matched.
  `minority_admission_rate` divides by all protected applicants.
- Outcome figures show means with approximate 95% confidence intervals. Runtime
  figures show medians and interquartile ranges. The raw timings retain isolated
  long measurements. Paired comparisons use differences on the same market.

## Source map

| File | Role |
|---|---|
| `generation.py`, `model.py` | Seeded market generation and eligibility model |
| `algorithms.py`, `policies.py` | Deferred acceptance, benchmark adapters, and exact institutional choice |
| `metrics.py`, `experiments.py` | Evaluation, experiment design, and numerical checkpoints |
| `reporting.py`, `analysis.py` | Archived data loading, audits, and shared-axis plots |

The geographic preference generator and three overlapping benchmarks are reused
from [`src/`](src/). Generated reports, local checkpoints, cached validation
runs, and Python environments are ignored by Git. The bundled data and source
are sufficient to regenerate the paper reports on another machine.
