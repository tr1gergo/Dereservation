"""Read archived observations for reporting without rerunning mechanisms.

Archived source hashes remain provenance of the measured runs, rather than a
requirement that today's plotting code has the same hash. New computations
continue to use the strict source checks in experiments.run_study.
"""

from pathlib import Path
import json

import pandas as pd

from .experiments import StudyResults, run_study

PAPER_ARCHIVE = Path(__file__).resolve().parent / "data" / "paper"
TABLE_NAMES = ("outcomes", "groups", "institutions", "welfare")


def without_wd(frame):
    if "algorithm" not in frame:
        return frame.copy()
    return frame.loc[~frame.algorithm.astype(str).str.startswith("WD")].copy()


def load_or_run_study(scenarios, repetitions, base_seed, suites, output_dir,
                      resume=True, save_assignments=False, archive_dir=None):
    """Reuse a complete archived design and any subset of its algorithms.

    No files in the archive are rewritten on a cache hit. Different seeds,
    scenarios, replications or algorithm definitions cannot silently reuse it.
    Changes to reporting code or the removal of unwanted algorithms are safe.
    """
    output = Path(output_dir)
    candidates = [output]
    if archive_dir is not None:
        candidates.append(Path(archive_dir))
    directory = None
    for candidate in candidates if resume else ():
        if not (candidate / "manifest.json").is_file():
            continue
        paths = {name: _csv_path(candidate, name) for name in TABLE_NAMES}
        if all(path.is_file() for path in paths.values()):
            directory = candidate
            break
    if directory is None:
        return run_study(scenarios, repetitions, base_seed=base_seed, suites=suites,
                         output_dir=output, resume=resume,
                         save_assignments=save_assignments)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_scenarios = json.loads(json.dumps([vars(s) for s in scenarios]))
    if (manifest["scenarios"] != expected_scenarios or
            manifest["repetitions"] != repetitions or manifest["base_seed"] != base_seed):
        raise ValueError("Saved design/seeds do not match the requested report. Select the matching profile or a new output directory.")
    names = {}
    for regime, suite in suites.items():
        cached = {a["name"]: a for a in manifest["algorithms"][regime]}
        names[regime] = set()
        for a in suite:
            definition = dict(name=a.name, kind=a.kind, engine=a.engine,
                              policy=a.policy.kind if a.policy else None,
                              **{"lambda": str(a.policy.lam) if a.policy else None})
            if cached.get(a.name) != definition:
                raise ValueError(f"No matching archived definition for {regime}/{a.name}")
            names[regime].add(a.name)
    tables = {}
    for name, path in paths.items():
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if "algorithm" in frame:
            mask = pd.Series(False, index=frame.index)
            for regime, algorithms in names.items():
                mask |= frame.regime.eq(regime) & frame.algorithm.isin(algorithms)
            frame = without_wd(frame.loc[mask])
        tables[name] = frame
    observed = tables["outcomes"]
    if observed.duplicated(["scenario", "seed", "algorithm"]).any():
        raise ValueError("Duplicate archived observations")
    expected = {(s.name, base_seed + r, name) for s in scenarios
                for r in range(repetitions) for name in names[s.regime]}
    actual = set(observed[["scenario", "seed", "algorithm"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("Archived study is incomplete for the requested report; no mechanisms were rerun.")
    print(f"Loaded {len(observed):,} cached observations from {directory}.")
    return StudyResults(**tables, manifest=manifest)


def _csv_path(directory, name):
    plain = directory / f"{name}.csv"
    return plain if plain.is_file() else directory / f"{name}.csv.gz"


def load_runtime_cache(path, sizes, repetitions, num_institutions):
    """Validate and reuse measured MD runtime observations, without retiming."""
    data = pd.read_csv(path)
    expected = {(n, r, engine) for n in sizes for r in range(repetitions)
                for engine in ("incremental", "flow")}
    keys = ["num_students", "replicate", "engine"]
    if (data.duplicated(keys).any() or
            set(data[keys].itertuples(index=False, name=None)) != expected or
            not data.num_institutions.eq(num_institutions).all() or
            not data.policy.eq("MD").all()):
        raise ValueError("Runtime cache does not match the requested MD benchmark")
    print(f"Loaded {len(data)} saved runtime observations; no benchmarks rerun.")
    return data
