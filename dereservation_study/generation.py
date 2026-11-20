"""Paired synthetic markets, reusing the project's spatial preference generator."""

from contextlib import contextmanager
from math import floor
import random

from src.instance_generation import generate_instance

from .model import Market


@contextmanager
def _legacy_seed(seed):
    # The original generator uses the global RNG. Preserve the caller's state.
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _apportion(total, shares):
    """Hamilton apportionment with deterministic ties; sums exactly to total."""
    if total < 0 or any(x < 0 for x in shares.values()) or abs(sum(shares.values()) - 1) > 1e-9:
        raise ValueError("Shares must be nonnegative and sum to one")
    raw = {c: total * p for c, p in shares.items()}
    counts = {c: floor(x) for c, x in raw.items()}
    order = sorted(shares, key=lambda c: -(raw[c] - counts[c]))
    for c in order[:total - sum(counts.values())]:
        counts[c] += 1
    return counts


def _base(seed, num_students, num_institutions, alpha, beta,
          capacity_ratio, preference_length, fam_type):
    if capacity_ratio <= 0:
        raise ValueError("capacity_ratio must be positive")
    if preference_length is not None and not 0 <= preference_length <= num_institutions:
        raise ValueError("preference_length must lie between zero and the number of schools")
    with _legacy_seed(seed):
        base = generate_instance(num_students, num_institutions, fam_type, alpha, beta)
    total = round(num_students * capacity_ratio)
    rng = random.Random(seed + 811)
    order = list(base["institutions"])
    rng.shuffle(order)
    base["capacities"] = {i: total // num_institutions + (order.index(i) < total % num_institutions)
                          for i in base["institutions"]}
    if preference_length is not None:
        base["student_prefs"] = {s: prefs[:preference_length]
                                 for s, prefs in base["student_prefs"].items()}
    base["metadata"].update(seed=seed, capacity_ratio=capacity_ratio,
                            preference_length=preference_length)
    return base


def _legacy_projection(market):
    """Physical reserve slots become lower quotas for the two old benchmarks.

    Type memberships are original reserved-category eligibility, so one student
    may count toward several soft bounds. Hard-reserve slots still consume one
    seat per student. These are distinct interpretations, kept explicit.
    """
    base = market.base
    cats = [c for c in market.precedence[market.institutions[0]] if c in market.reserved]
    base["all_categories"] = cats
    base["categories"] = {
        s: {c for c in cats if any(s in market.original[i][c] for i in market.institutions)}
        for s in market.students
    }
    base["bounds"] = {
        i: {"lower": {c: market.quotas[i][c] for c in cats},
            "upper": {c: base["capacities"][i] for c in cats}}
        for i in market.institutions
    }
    return market.validate()


def generate_indian_market(seed=0, num_students=200, num_institutions=10,
                           obc_population=0.12, sc_population=0.15,
                           st_population=0.075, quota_shares=None,
                           capacity_ratio=1.0, preference_length=None,
                           alpha=0.4, beta=0.4, fam_type="independent",
                           priority_mode="common_merit", protected_score_shift=0.0):
    """Stylized vertical reservations, not an empirical Indian dataset.

    Only OBC seats can transfer; SC/ST remain exclusive. Default seat shares
    are illustrative 50.5/27/15/7.5 percent. OBC supply varies independently.
    Demographics are sampled per student; only location/preferences reuse the
    old family model. A positive score shift favors protected students in merit.
    """
    shares = quota_shares or {"g": .505, "OBC": .27, "SC": .15, "ST": .075}
    if set(shares) != {"g", "OBC", "SC", "ST"}:
        raise ValueError("Indian quotas require g, OBC, SC, ST")
    population = {"OBC": obc_population, "SC": sc_population, "ST": st_population,
                  "g": 1 - obc_population - sc_population - st_population}
    _apportion(1, population)  # validate even if the generated capacity is zero
    if priority_mode not in {"common_merit", "school_merit"}:
        raise ValueError("Unknown Indian priority mode")
    base = _base(seed, num_students, num_institutions, alpha, beta,
                 capacity_ratio, preference_length, fam_type)
    rng = random.Random(seed + 1907)
    types = dict(zip(base["students"], rng.choices(list(population), weights=list(population.values()), k=num_students)))
    groups = {c: {s for s in base["students"] if types[s] == c} for c in ("OBC", "SC", "ST")}
    merit = {s: rng.gauss(0, 1) + protected_score_shift * (types[s] != "g") for s in base["students"]}
    common = sorted(base["students"], key=lambda s: (-merit[s], s))
    if priority_mode == "common_merit":
        base["inst_prefs"] = {i: list(common) for i in base["institutions"]}
    else:
        # Reuse school-specific walk-zone priorities unless a score shift was requested.
        if protected_score_shift:
            raise ValueError("protected_score_shift requires common_merit")
    quotas, original, transfers, priorities, precedence = {}, {}, {}, {}, {}
    for i in base["institutions"]:
        quotas[i] = _apportion(base["capacities"][i], shares)
        precedence[i] = ["g", "OBC", "SC", "ST"]
        original[i] = {"g": set(base["students"]), **{c: set(g) for c, g in groups.items()}}
        transfers[i] = {c: set() for c in shares}
        transfers[i]["OBC"] = set(base["students"]) - groups["OBC"]
        priorities[i] = {c: [s for s in base["inst_prefs"][i] if s in original[i][c]] +
                              [s for s in base["inst_prefs"][i] if s in transfers[i][c]] for c in shares}
    base["social_types"] = types
    base["metadata"].update(regime="indian", obc_population=obc_population,
                            sc_population=sc_population, st_population=st_population,
                            priority_mode=priority_mode, protected_score_shift=protected_score_shift,
                            quota_shares=dict(shares))
    return _legacy_projection(Market(base, quotas, original, transfers, priorities,
                                     precedence, set(groups), groups))


def generate_overlapping_market(seed=0, num_students=200, num_institutions=10,
                                num_types=3, type_probability=0.3, overlap=0.3,
                                reserve_fraction=0.6, transferable_fraction=1.0,
                                transfer_access=1.0, priority_noise=0.0,
                                heterogeneous_precedence=False,
                                capacity_ratio=1.0, preference_length=None,
                                alpha=0.4, beta=0.4, fam_type="independent"):
    """Overlapping types and optional category-specific priorities/permissions.

    ``overlap`` mixes independent Bernoulli type membership with a common
    Bernoulli draw, preserving each type's marginal probability. A fraction of
    reserved categories permits transfers; ``transfer_access`` is the fraction
    of nonmembers additionally eligible at those categories. At the default
    full transfer access all old soft-bound outcomes are implementable.
    """
    if type(num_types) is not int or num_types < 1:
        raise ValueError("num_types must be a positive integer")
    for name, p in {"type_probability": type_probability, "overlap": overlap,
                    "reserve_fraction": reserve_fraction, "transferable_fraction": transferable_fraction,
                    "transfer_access": transfer_access, "priority_noise": priority_noise}.items():
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    base = _base(seed, num_students, num_institutions, alpha, beta,
                 capacity_ratio, preference_length, fam_type)
    rng = random.Random(seed + 2909)
    cats = [f"T{t+1}" for t in range(num_types)]
    groups = {c: set() for c in cats}
    for s in base["students"]:
        common = rng.random() < type_probability
        correlated = rng.random() < overlap
        for c in cats:
            if common if correlated else rng.random() < type_probability:
                groups[c].add(s)
    shares = {"g": 1 - reserve_fraction, **{c: reserve_fraction / num_types for c in cats}}
    quotas, original, transfers, priorities, precedence = {}, {}, {}, {}, {}
    for i in base["institutions"]:
        quotas[i] = _apportion(base["capacities"][i], shares)
        order = list(cats)
        rng.shuffle(order)
        enabled = set(order[:round(transferable_fraction * num_types)])
        precedence[i] = ["g"] + (order if heterogeneous_precedence else cats)
        original[i] = {"g": set(base["students"]), **{c: set(groups[c]) for c in cats}}
        transfers[i] = {"g": set(), **{c: {s for s in base["students"] if s not in groups[c]
                                          and c in enabled and rng.random() < transfer_access} for c in cats}}
        priorities[i] = {}
        merit_rank = {s: r for r, s in enumerate(base["inst_prefs"][i])}
        for c in shares:
            # A convex mixture of merit rank and independent category lottery.
            score = {s: (1 - priority_noise) * merit_rank[s] / num_students + priority_noise * rng.random()
                     for s in base["students"]}
            key = lambda s: (score[s], merit_rank[s])
            priorities[i][c] = sorted(original[i][c], key=key) + sorted(transfers[i][c], key=key)
    base["metadata"].update(regime="overlapping", num_types=num_types,
                            type_probability=type_probability, overlap=overlap,
                            reserve_fraction=reserve_fraction, transferable_fraction=transferable_fraction,
                            transfer_access=transfer_access, priority_noise=priority_noise,
                            heterogeneous_precedence=heterogeneous_precedence)
    return _legacy_projection(Market(base, quotas, original, transfers, priorities,
                                     precedence, set(cats), groups))
