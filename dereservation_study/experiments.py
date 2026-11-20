"""Paired designs, mechanism-only timings, checkpoints, and uncertainty tables."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

import pandas as pd

from .algorithms import algorithm_suite, attach_legacy_support, _legacy
from .generation import generate_indian_market, generate_overlapping_market
from .metrics import evaluate_outcome, group_outcomes, institution_outcomes, student_ranks


METRICS = ["average_rank", "average_rank_matched", "num_matches", "fill_rate",
           "original_reserved_seats", "dereserved_seats", "reserved_seats_occupied",
           "unused_reserved_seats", "original_reserved_utilization", "avoidable_original_reservation_loss",
           "blocking_pairs", "choice_envy_pairs", "choice_waste_pairs",
           "category_envy_pairs", "category_waste_pairs", "category_blocking_pairs",
           "merit_envy_pairs", "merit_waste_pairs", "minority_matched",
           "minority_share_matched", "minority_admission_rate", "runtime"]


@dataclass
class Scenario:
    name: str
    regime: str
    parameters: dict = field(default_factory=dict)
    axis: str = "baseline"
    value: float = 0.0


@dataclass
class StudyResults:
    outcomes: pd.DataFrame
    groups: pd.DataFrame
    institutions: pd.DataFrame
    welfare: pd.DataFrame
    manifest: dict

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name in ("outcomes", "groups", "institutions", "welfare"):
            table = getattr(self, name)
            temporary = path / f".{name}.tmp.csv"
            table.to_csv(temporary, index=False)
            temporary.replace(path / f"{name}.csv")
        summary = summarize(self.outcomes)
        summary.to_csv(path / "metric_summary.csv", index=False)
        _write_json(path / "manifest.json", self.manifest)


def proposed_design(profile="smoke"):
    """One-factor-at-a-time studies plus a separate runtime experiment.

    The paper profile is a protocol, not a claim that those runs are complete.
    Use the smoke profile to check the entire notebook in a few seconds.
    """
    if profile not in {"smoke", "quick", "paper"}:
        raise ValueError("profile must be smoke, quick, or paper")
    n, schools = {"smoke": (36, 3), "quick": (150, 8), "paper": (600, 20)}[profile]
    repetitions = {"smoke": 2, "quick": 10, "paper": 100}[profile]
    shared = dict(num_students=n, num_institutions=schools, preference_length=max(2, schools // 2),
                  alpha=0.4, beta=0.4)
    scenarios = []
    def sweep(regime, axis, values, **fixed):
        for value in values:
            scenarios.append(Scenario(f"{regime}_{axis}={value}", regime,
                                      {**shared, **fixed, axis: value}, axis, value))
    sweep("indian", "obc_population", (.02, .27) if profile == "smoke" else (.02, .08, .15, .27, .40))
    sweep("overlapping", "overlap", (0.0, .8) if profile == "smoke" else (0.0, .25, .5, .75, 1.0))
    if profile != "smoke":
        sweep("indian", "capacity_ratio", (.75, 1.0, 1.25))
        sweep("indian", "preference_length", (1, max(2, schools // 2), schools))
        sweep("indian", "protected_score_shift", (-1.0, 0.0, 1.0))
        sweep("overlapping", "reserve_fraction", (.2, .4, .6, .8))
        sweep("overlapping", "priority_noise", (0.0, .5, 1.0), heterogeneous_precedence=True)
        # These robustness cells deliberately expose out-of-domain outcomes
        # of old soft-bound benchmarks; eligibility_feasible is never hidden.
        sweep("overlapping", "transferable_fraction", (0.0, .5, 1.0))
        sweep("overlapping", "transfer_access", (.25, .5, 1.0), priority_noise=.5,
              heterogeneous_precedence=True)
    return scenarios, repetitions


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_clean_json(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _source_hash():
    root = Path(__file__).resolve().parent
    study_sources = ("algorithms.py", "experiments.py", "generation.py", "metrics.py", "model.py", "policies.py")
    files = [root / name for name in study_sources]
    files += [root.parent / "src" / name for name in ("algorithms.py", "instance_generation.py")]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest(scenarios, repetitions, seed, suites):
    packages = {}
    for name in ("numpy", "pandas", "networkx", "matplotlib", "gurobipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"schema_version": 1, "source_sha256": _source_hash(),
            "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "base_seed": seed, "repetitions": repetitions,
            "scenarios": [vars(s) for s in scenarios],
            "algorithms": {regime: [{"name": a.name, "kind": a.kind, "engine": a.engine,
                                       "policy": a.policy.kind if a.policy else None,
                                       "lambda": str(a.policy.lam) if a.policy else None}
                                      for a in algorithms] for regime, algorithms in suites.items()},
            "timing": "perf_counter; mechanism including weights; excludes generation, input copy, metrics, posthoc support, IO",
            "ci": "normal approximation: mean +/- 1.96 * sample SD / sqrt(n); paired differences use paired samples",
            "synthetic": True}


def _paired_welfare(market, outputs, reference, feasibility):
    if reference not in outputs:
        return []
    ref = student_ranks(market, outputs[reference])
    rows = []
    for name, outcome in outputs.items():
        ranks = student_ranks(market, outcome)
        gains = [ref[s] - ranks[s] for s in market.students]
        rows.append({"algorithm": name, "reference": reference,
                     "eligibility_feasible": feasibility[name] and feasibility[reference],
                     "students_better": sum(x > 0 for x in gains),
                     "students_equal": sum(x == 0 for x in gains),
                     "students_worse": sum(x < 0 for x in gains),
                     "share_better": sum(x > 0 for x in gains) / len(gains),
                     "share_worse": sum(x < 0 for x in gains) / len(gains),
                     "mean_rank_gain": sum(gains) / len(gains)})
    return rows


def run_study(scenarios, repetitions=2, base_seed=271828, suites=None,
              output_dir=None, resume=True, verbose=True, save_assignments=False):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    scenarios = list(scenarios)
    if len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("Scenario names must be unique")
    regimes = sorted({s.regime for s in scenarios})
    if not scenarios or not set(regimes) <= {"indian", "overlapping"}:
        raise ValueError("Supply at least one Indian or overlapping scenario")
    suites = suites or {regime: algorithm_suite(regime) for regime in regimes}
    for regime in regimes:
        names = [a.name for a in suites[regime]]
        if not names or len(names) != len(set(names)):
            raise ValueError("Each regime needs uniquely named algorithms")
    # Exclude module-import cold starts from algorithm timings.
    if any(a.kind in {"DA-open", "Smart-Hard-Reserves", "Soft-Reserves"}
           for regime in regimes for a in suites[regime]):
        _legacy()
    manifest = _manifest(scenarios, repetitions, base_seed, suites)
    manifest["save_assignments"] = save_assignments
    directory = Path(output_dir) if output_dir is not None else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and resume:
            if json.loads(manifest_path.read_text(encoding="utf-8")) != _clean_json(manifest):
                raise ValueError("Checkpoint configuration, source, or environment changed. Use a new output_dir or resume=False.")
        _write_json(manifest_path, manifest)
        (directory / "checkpoints").mkdir(exist_ok=True)
    accumulated = {key: [] for key in ("outcomes", "groups", "institutions", "welfare")}
    job = 0
    for scenario in scenarios:
        generator = generate_indian_market if scenario.regime == "indian" else generate_overlapping_market
        for replicate in range(repetitions):
            job += 1
            seed = base_seed + replicate
            identity = f"{scenario.name}__seed={seed}"
            # The configuration hash prevents stale checkpoints being reused
            # after resume=False or after switching between experiment designs.
            key = hashlib.sha256((json.dumps(_clean_json(manifest), sort_keys=True) + identity).encode()).hexdigest()[:24]
            checkpoint = directory / "checkpoints" / f"{key}.json" if directory else None
            if resume and checkpoint is not None and checkpoint.exists():
                batch = json.loads(checkpoint.read_text(encoding="utf-8"))
                if verbose:
                    print(f"[{job}/{len(scenarios)*repetitions}] resume {identity}")
            else:
                market = generator(seed=seed, **scenario.parameters)
                context = {"instance_id": identity, "scenario": scenario.name, "regime": scenario.regime,
                           "axis": scenario.axis, "value": scenario.value, "replicate": replicate, "seed": seed,
                           "num_students": len(market.students), "num_institutions": len(market.institutions)}
                batch = {key: [] for key in accumulated}
                outputs, feasibility = {}, {}
                algorithms = list(suites[scenario.regime])
                random.Random(seed + 13).shuffle(algorithms)
                if verbose:
                    print(f"[{job}/{len(scenarios)*repetitions}] {identity}")
                for algorithm in algorithms:
                    isolated_market = deepcopy(market)
                    start = time.perf_counter()
                    outcome = algorithm.run(isolated_market)
                    runtime = time.perf_counter() - start
                    outcome = attach_legacy_support(market, outcome)
                    metrics = evaluate_outcome(market, outcome, algorithm, runtime)
                    feasibility[algorithm.name] = metrics["eligibility_feasible"]
                    batch["outcomes"].append({**context, "algorithm": algorithm.name, **metrics})
                    batch["groups"].extend({**context, "algorithm": algorithm.name, **row}
                                           for row in group_outcomes(market, outcome))
                    batch["institutions"].extend({**context, "algorithm": algorithm.name, **row}
                                                 for row in institution_outcomes(market, outcome))
                    outputs[algorithm.name] = outcome
                reference = "BT" if scenario.regime == "indian" else "DA-open"
                batch["welfare"] = [{**context, **row} for row in _paired_welfare(market, outputs, reference, feasibility)]
                if checkpoint is not None:
                    if save_assignments:
                        batch["assignments"] = {name: {"matching": out.matching, "assignments": out.assignments,
                                                       "assignment_source": out.assignment_source}
                                                for name, out in outputs.items()}
                    _write_json(checkpoint, batch)
            for key in accumulated:
                accumulated[key].extend(batch[key])
    frames = {}
    preferred = ["instance_id", "regime", "scenario", "axis", "value", "replicate", "seed", "algorithm"]
    for key, rows in accumulated.items():
        frame = pd.DataFrame(rows)
        # JSON checkpoints sort object keys. Keep the CSV schema identical
        # whether every job is fresh, resumed, or a mixture of the two.
        first = [column for column in preferred if column in frame]
        frames[key] = frame.reindex(columns=first + sorted(set(frame.columns) - set(first)))
    result = StudyResults(**frames, manifest=manifest)
    if directory:
        result.save(directory)
    return result


def summarize(outcomes, metrics=METRICS, feasible_only=True):
    data = outcomes[outcomes["eligibility_feasible"]] if feasible_only else outcomes
    if data.empty:
        return pd.DataFrame()
    keys = ["regime", "scenario", "axis", "value", "algorithm"]
    long = data.melt(id_vars=keys, value_vars=[m for m in metrics if m in data], var_name="metric", value_name="measurement")
    summary = long.groupby(keys + ["metric"], dropna=False)["measurement"].agg(n="count", mean="mean", sd="std").reset_index()
    summary["se"] = summary["sd"] / summary["n"].pow(.5)
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["se"]
    return summary


def paired_differences(outcomes, reference, metrics=METRICS):
    """Algorithm minus reference on the SAME scenario/replicate/seed.

    Only mutually feasible runs enter a pair. The SD/CI are computed on paired
    differences, not from two independent algorithm summaries.
    """
    data = outcomes[outcomes["eligibility_feasible"]]
    keys = ["instance_id", "regime", "scenario", "axis", "value", "replicate", "seed"]
    metrics = [m for m in metrics if m in data]
    baseline = data[data["algorithm"] == reference][keys + metrics]
    paired = data.merge(baseline, on=keys, how="inner", suffixes=("", "_reference"), validate="many_to_one")
    for metric in metrics:
        paired[metric] = paired[metric] - paired[f"{metric}_reference"]
    paired["eligibility_feasible"] = True
    summary = summarize(paired, metrics)
    summary["reference"] = reference
    return summary
