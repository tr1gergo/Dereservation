"""Figures, outcome checks, and runtime comparisons for the simulations."""

from collections import defaultdict
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd

from .algorithms import Algorithm
from .experiments import summarize
from .generation import generate_indian_market, generate_overlapping_market
from .policies import Policy
from .reporting import without_wd


LABELS = {
    "average_rank": "Mean student rank",
    "num_matches": "Matched students", "fill_rate": "Filled share of physical capacity",
    "original_reserved_seats": "Original reserved seats filled ($D$)",
    "dereserved_seats": "De-reserved seats filled ($T$)",
    "blocking_pairs": "Choice blocking pairs",
    "category_envy_pairs": "Category priority envy pairs",
    "category_waste_pairs": "Category vacancy blocking pairs",
    "minority_matched": "Unique protected students matched",
    "minority_share_matched": "Protected share of matched students",
    "runtime": "Mechanism runtime (seconds)",
}

SWEEP_METRICS = {
    "welfare_seats": ("average_rank", "num_matches", "original_reserved_seats", "dereserved_seats"),
    "fairness_runtime": ("category_envy_pairs", "category_waste_pairs", "minority_share_matched", "runtime"),
}

AXIS_LABELS = {
    "obc_population": "OBC share of applicants",
    "overlap": "Overlap probability",
    "capacity_ratio": "Seats per applicant",
    "preference_length": "Acceptable institutions per applicant",
    "protected_score_shift": "Merit score shift for protected applicants",
    "reserve_fraction": "Reserved share of seats",
    "priority_noise": "Category priority noise",
    "transferable_fraction": "Share of categories allowing transfers",
    "transfer_access": "Probability of transfer eligibility for nonmembers",
}

COLORS = {
    "DA": "#4b5563", "DA-open": "#4b5563", "Multi-run DA": "#a8611e",
    "BT": "#2865a8", "Smart-Hard-Reserves": "#a8611e", "Soft-Reserves": "#2865a8",
    "MD": "#007f66", "RP": "#993e71", "Merit": "#4b4291", "Demographic": "#a63831",
}
ALGORITHM_ORDER = {name: position for position, name in enumerate(COLORS)}
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
DASHES = ("-", (0, (5, 2)), (0, (1.1, 1.6)), (0, (6, 2, 1, 2)),
          (0, (3, 2)), (0, (1, 1.5, 5, 1.5)))


def save_figure(figure, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{stem}.{extension}", dpi=180, bbox_inches="tight")


def runtime_statistics(data, by):
    """Median and interquartile range of the recorded wall-clock times."""
    return data.groupby(by)["runtime"].agg(
        n="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
    ).reset_index()


def _coincident_markers(points, logarithmic):
    """Place identifying symbols along equal segments at distinct positions."""
    tags = defaultdict(list)
    for left, right in zip(points.index[:-1], points.index[1:]):
        segments = defaultdict(list)
        for name in points.columns:
            y0, y1 = points.loc[left, name], points.loc[right, name]
            if pd.notna(y0) and pd.notna(y1):
                segments[y0, y1].append(name)
        for (y0, y1), names in segments.items():
            if len(names) < 2:
                continue
            for j, name in enumerate(names, start=1):
                fraction = j / (len(names) + 1)
                x = left + fraction * (right - left)
                y = y0 ** (1 - fraction) * y1 ** fraction if logarithmic else y0 + fraction * (y1 - y0)
                tags[name].append((x, y))
    return tags


def plot_sweep(outcomes, regime, axis, metrics=None, output_dir=None):
    """Draw all algorithms together on one set of axes per metric."""
    defaults = tuple(metric for group in SWEEP_METRICS.values() for metric in group)
    metrics = tuple(metrics or defaults)
    data = without_wd(outcomes[(outcomes["regime"] == regime) & (outcomes["axis"] == axis)])
    if data.empty:
        raise ValueError("No outcomes for that regime and sweep")
    feasible = data[data.eligibility_feasible]
    names = sorted(data.algorithm.unique(), key=lambda name: (ALGORITHM_ORDER.get(name, len(COLORS)), name))
    values = sorted(data.value.unique())
    styles = {}
    handles = []
    for j, name in enumerate(names):
        color = COLORS.get(name, plt.get_cmap("tab10")(j % 10))
        # Draw wider strokes first. Narrow dashed strokes leave the wider
        # lines visible even when several means follow the same path.
        width = 1.35 + 2.7 * (len(names) - j - 1) / max(1, len(names) - 1)
        line = dict(color=color, linewidth=width, linestyle=DASHES[j % len(DASHES)])
        marker = MARKERS[j % len(MARKERS)]
        styles[name] = (line, marker)
        handles.append(Line2D([], [], **line, marker=marker, markersize=5, label=name))

    columns = min(2, len(metrics))
    rows = (len(metrics) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 2.9 * rows + .8), squeeze=False)
    x_label = AXIS_LABELS.get(axis, axis.replace("_", " "))
    grouped_by = ["value", "algorithm"]
    for panel, metric in zip(axes.flat, metrics):
        logarithmic = metric == "runtime"
        if logarithmic:
            stats = runtime_statistics(feasible, grouped_by).rename(columns={"median": "center"})
            stats["below"] = stats.center - stats.q1
            stats["above"] = stats.q3 - stats.center
            panel.set_yscale("log")
        else:
            stats = summarize(data, [metric])
            if stats.empty:
                stats = pd.DataFrame(columns=[*grouped_by, "mean", "se"])
            stats = stats.rename(columns={"mean": "center"})
            stats["below"] = stats["above"] = 1.96 * stats.se

        # Align each curve to the full grid so a missing feasible estimate
        # leaves a gap instead of connecting the observations on either side.
        points = stats.pivot(index="value", columns="algorithm", values="center").reindex(
            index=values, columns=names,
        )
        tags = _coincident_markers(points, logarithmic)
        for j, name in enumerate(names):
            line, marker = styles[name]
            group = stats[stats.algorithm.eq(name)].set_index("value").reindex(values)
            panel.errorbar(values, points[name].to_numpy(dtype=float),
                           yerr=group[["below", "above"]].fillna(0).to_numpy(dtype=float).T,
                           **line, marker=marker, markersize=3.7, capsize=3, elinewidth=1,
                           zorder=3 + j, label=name)
        for name, locations in tags.items():
            line, marker = styles[name]
            tag_x, tag_y = zip(*locations)
            panel.plot(tag_x, tag_y, linestyle="none", marker=marker, markersize=5.8,
                       color=line["color"], markeredgecolor="white", markeredgewidth=.65,
                       zorder=4 + len(names))
        panel.set(xlabel=x_label, ylabel=LABELS.get(metric, metric.replace("_", " ")), xticks=values)
        if metric in {"minority_share_matched", "fill_rate"}:
            panel.yaxis.set_major_formatter(PercentFormatter(1))
        panel.grid(axis="y", alpha=.25)
        if not points.notna().any().any():
            panel.text(.5, .5, "No feasible observations", transform=panel.transAxes, ha="center")
    for panel in list(axes.flat)[len(metrics):]:
        panel.set_visible(False)
    description = "Indian vertical reservations" if regime == "indian" else "Overlapping eligibility"
    fig.suptitle(f"{description} - {x_label}", fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .058),
               ncol=min(3, len(names)), frameon=False, fontsize=9, handlelength=3.1)
    interval_note = ("Means and approximate 95% CIs." if "runtime" not in metrics else
                     "Runtime shows median and IQR. Other panels show means and approximate 95% CIs.")
    fig.text(.5, .018, f"{interval_note} Symbols between settings identify coincident curves.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .21 if rows == 1 else .15, 1, .93))
    if output_dir is not None:
        group_names = {group: name for name, group in SWEEP_METRICS.items()}
        group_names[defaults] = "overview"
        name = group_names.get(metrics, "_".join(metrics))
        save_figure(fig, output_dir, f"{regime}_{axis}_{name}")
    return fig


def benchmark_weighted_engines(sizes=(30, 60, 120), repetitions=3, regime="overlapping",
                               num_institutions=4, base_seed=314159, policy=Policy()):
    """Time DA's single-path update against full min-cost flow on every proposal.

    Both timings include weight construction. Their complete original-seat
    assignments must agree on every instance. No metrics/plotting is timed.
    This compares two Python implementations, not theoretical complexity.
    """
    import networkx  # noqa: F401 -- exclude import overhead from timings

    if regime not in {"indian", "overlapping"} or repetitions < 1:
        raise ValueError("Invalid regime or repetition count")
    generator = generate_indian_market if regime == "indian" else generate_overlapping_market
    records = []
    for n in sizes:
        for replicate in range(repetitions):
            market = generator(seed=base_seed + replicate, num_students=n,
                               num_institutions=num_institutions,
                               preference_length=max(1, num_institutions // 2))
            engines = ["incremental", "flow"]
            random.Random(base_seed + replicate + n).shuffle(engines)
            outputs = {}
            for engine in engines:
                algorithm = Algorithm(f"{policy.name}-{engine}", "weighted", policy, engine)
                start = time.perf_counter()
                output = algorithm.run(market)
                runtime = time.perf_counter() - start
                outputs[engine] = output
                records.append({"num_students": n, "num_institutions": num_institutions,
                                "replicate": replicate, "seed": base_seed + replicate,
                                "engine": engine, "policy": policy.name, "runtime": runtime,
                                "proposals": output.diagnostics["proposals"],
                                "num_matches": sum(map(len, output.matching.values()))})
            if outputs["incremental"].assignments != outputs["flow"].assignments:
                raise AssertionError("Incremental and reference flow assignments differ")
    return pd.DataFrame(records)


def audit_results(study):
    """Empirical consistency checks, not proofs or publication claims."""
    data = study.outcomes
    weighted = data.algorithm.isin(["MD", "RP", "Merit", "Demographic"]) | data.algorithm.str.startswith("WD(")
    if not data.loc[weighted, "eligibility_feasible"].all():
        raise AssertionError("A weighted mechanism returned an ineligible assignment")
    if not data.loc[weighted, "choice_stable"].all():
        raise AssertionError("A weighted DA outcome has a choice blocking pair")
    fair = data.algorithm.isin(["MD", "RP"]) | data.algorithm.str.startswith("WD(")
    if not (data.loc[fair, "category_envy_pairs"] == 0).all():
        raise AssertionError("A priority-consistent weighted policy violated admission fairness")
    welfare = study.welfare
    comparison = (welfare[(welfare.regime == "indian") & (welfare.algorithm == "MD") & (welfare.reference == "BT")]
                  if not welfare.empty else welfare)
    if not comparison.empty and not (comparison.students_worse == 0).all():
        raise AssertionError("MD/BT Indian welfare comparison failed")
    valid = data[data.eligibility_feasible]
    if not ((valid.original_reserved_seats + valid.dereserved_seats) == valid.reserved_seats_occupied).all():
        raise AssertionError("Original and transferred reserved-seat counts do not partition occupancy")
    return {"evaluated_mechanisms": len(data), "weighted_runs": int(weighted.sum()),
            "unsupported_legacy_runs": int((~data.eligibility_feasible).sum()),
            "md_bt_welfare_comparisons": len(comparison)}
