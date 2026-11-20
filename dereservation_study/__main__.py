"""Command-line equivalent of the main notebook experiment."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .analysis import SWEEP_METRICS, audit_results, plot_sweep, runtime_statistics
from .algorithms import algorithm_suite
from .reporting import PAPER_ARCHIVE, load_or_run_study
from .experiments import METRICS, paired_differences, proposed_design, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "quick", "paper"), default="smoke")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8-whitegrid")
    scenarios, repetitions = proposed_design(args.profile)
    output = args.output or Path(__file__).resolve().parent / "results" / f"notebook_{args.profile}"
    count = repetitions if args.repetitions is None else args.repetitions
    archive = PAPER_ARCHIVE if args.profile == "paper" and args.seed == 271828 and count == repetitions else None
    suites = {regime: algorithm_suite(regime) for regime in ("indian", "overlapping")}
    result = load_or_run_study(scenarios, count, base_seed=args.seed, suites=suites,
                               output_dir=output / "main", resume=not args.no_resume, archive_dir=archive)
    report = output / "reports"
    report.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in METRICS if m != "runtime"]
    summarize(result.outcomes, metrics).to_csv(report / "metric_summary.csv", index=False)
    feasible = result.outcomes.loc[result.outcomes.eligibility_feasible]
    runtime_statistics(feasible, ["regime", "scenario", "axis", "value", "algorithm"]).to_csv(
        report / "algorithm_runtime.csv", index=False,
    )
    for reference in ("BT", "Soft-Reserves"):
        paired_differences(result.outcomes, reference, metrics).to_csv(report / f"paired_vs_{reference}.csv", index=False)
    for regime, axis in (("indian", "obc_population"), ("overlapping", "overlap")):
        for metrics_group in SWEEP_METRICS.values():
            figure = plot_sweep(result.outcomes, regime, axis, metrics_group, output_dir=report / "figures")
            plt.close(figure)
    print(audit_results(result))
    print(f"Saved study to {output}")


if __name__ == "__main__":
    main()
