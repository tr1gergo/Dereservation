"""Welfare, physical seat use, demographics, and three stability diagnostics."""

from math import nan

from .model import validate_placements, validate_contracts
from .policies import Policy, WeightedChoice


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else nan


def student_ranks(market, outcome):
    assigned = {s: i for i, cohort in outcome.matching.items() for s in cohort}
    return {s: (market.base["student_prefs"][s].index(assigned[s]) + 1 if s in assigned
                else len(market.base["student_prefs"][s]) + 1) for s in market.students}


def blocking_statistics(market, outcome, algorithm, contracts_valid=True):
    """Count student-school edges, never seats or envy triples.

    Choice blocking follows Section 7, allowing all category reassignments.
    Choice waste adds an admission; choice envy replaces an admission.
    Category envy/waste inspect the reported seat labeling without exchanges.
    Merit envy/waste ignore reserve protections and use published school merit.
    Raw envy and raw waste may overlap; their union is counted separately.
    """
    oracle = algorithm.oracle(market, outcome)
    held_choices = {i: oracle.choose(i, outcome.matching[i]) for i in market.institutions}
    held_optimal = {i: set(held_choices[i]) == set(outcome.matching[i]) for i in market.institutions}
    assigned = {s: i for i, cohort in outcome.matching.items() for s in cohort}
    ranks = {i: {s: r for r, s in enumerate(market.base["inst_prefs"][i])} for i in market.institutions}
    category_rank = {i: {c: {s: r for r, s in enumerate(market.priorities[i][c])}
                         for c in market.quotas[i]} for i in market.institutions}
    counts = dict(blocking_pairs=0, choice_envy_pairs=0, choice_waste_pairs=0,
                  category_envy_pairs=0, category_waste_pairs=0, category_blocking_pairs=0,
                  merit_envy_pairs=0, merit_waste_pairs=0, merit_blocking_pairs=0)
    for s in market.students:
        for i in market.base["student_prefs"][s]:
            if assigned.get(s) == i:
                break
            if s not in ranks[i]:
                continue
            cohort = outcome.matching[i]
            if isinstance(oracle, WeightedChoice) and held_optimal[i] and contracts_valid:
                chosen = oracle.update(i, held_choices[i], s)
            else:
                chosen = oracle.choose(i, set(cohort) | {s})
            if s in chosen:
                counts["blocking_pairs"] += 1
                counts["choice_waste_pairs" if len(chosen) > len(cohort) else "choice_envy_pairs"] += 1
            merit_waste = len(cohort) < market.base["capacities"][i]
            merit_envy = any(ranks[i][s] < ranks[i][t] for t in cohort)
            counts["merit_waste_pairs"] += int(merit_waste)
            counts["merit_envy_pairs"] += int(merit_envy)
            counts["merit_blocking_pairs"] += int(merit_waste or merit_envy)
            if contracts_valid:
                envy, waste = False, False
                for c, q in market.quotas[i].items():
                    if s not in category_rank[i][c]:
                        continue
                    occupants = [t for t, label in outcome.assignments[i].items() if label == c]
                    waste |= len(occupants) < q
                    envy |= any(category_rank[i][c][s] < category_rank[i][c][t] for t in occupants)
                counts["category_envy_pairs"] += int(envy)
                counts["category_waste_pairs"] += int(waste)
                counts["category_blocking_pairs"] += int(envy or waste)
    if not contracts_valid:
        for key in ("category_envy_pairs", "category_waste_pairs", "category_blocking_pairs"):
            counts[key] = nan
    counts["choice_individually_rational"] = all(held_optimal.values())
    counts["choice_stable"] = all(held_optimal.values()) and counts["blocking_pairs"] == 0
    return counts


def evaluate_outcome(market, outcome, algorithm, runtime):
    validate_placements(market, outcome.matching)
    try:
        validate_contracts(market, outcome)
        feasible = True
    except ValueError:
        feasible = False
    matched = set().union(*(set(v) for v in outcome.matching.values()))
    ranks = student_ranks(market, outcome)
    n, k = len(market.students), len(matched)
    total = sum(market.base["capacities"].values())
    minority = len(matched & market.protected)
    reserved_capacity = sum(q for i in market.institutions for c, q in market.quotas[i].items() if c in market.reserved)
    metrics = {
        "num_matches": k, "unmatched_students": n - k,
        "average_rank": sum(ranks.values()) / n,
        "average_rank_matched": _ratio(sum(ranks[s] for s in matched), k),
        "fill_rate": _ratio(k, total), "empty_seats": total - k,
        "minority_matched": minority, "minority_share_matched": _ratio(minority, k),
        "minority_admission_rate": _ratio(minority, len(market.protected)),
        "majority_admission_rate": _ratio(len(matched - market.protected), n - len(market.protected)),
        "reserved_capacity": reserved_capacity, "total_capacity": total,
        "eligibility_feasible": feasible,
        "unsupported_students": sum(len(outcome.matching[i]) - len(outcome.assignments.get(i, {}))
                                    for i in market.institutions),
        "assignment_source": outcome.assignment_source,
        "runtime": runtime, **outcome.diagnostics,
    }
    d = transferred = occupied = 0
    max_d = 0
    if feasible:
        protecting = WeightedChoice(market, Policy("RP"))
        for i in market.institutions:
            for s, c in outcome.assignments[i].items():
                if c in market.reserved:
                    occupied += 1
                    d += s in market.original[i][c]
                    transferred += s in market.transfers[i][c]
            best = protecting.choose(i, outcome.matching[i])
            max_d += sum(c in market.reserved and s in market.original[i][c] for s, c in best.items())
    metrics.update(
        original_reserved_seats=d if feasible else nan,
        dereserved_seats=transferred if feasible else nan,
        reserved_seats_occupied=occupied if feasible else nan,
        unused_reserved_seats=reserved_capacity - occupied if feasible else nan,
        original_reserved_utilization=_ratio(d, reserved_capacity) if feasible else nan,
        original_reserved_seats_max_for_cohort=max_d if feasible else nan,
        avoidable_original_reservation_loss=max_d - d if feasible else nan,
        # BT/multi-run quotas record administrative conversion, distinct from
        # an ED assignment (a converted slot may still be occupied by OBC).
        obc_quota_converted=(sum(market.quotas[i]["OBC"] - outcome.effective_quotas[i]["OBC"]
                                for i in market.institutions) if outcome.effective_quotas else nan),
        converted_seats_occupied=(sum(map(len, outcome.converted.values())) if outcome.effective_quotas else nan),
    )
    metrics.update(blocking_statistics(market, outcome, algorithm, feasible))
    return metrics


def group_outcomes(market, outcome):
    matched = set().union(*(set(v) for v in outcome.matching.values()))
    ranks = student_ranks(market, outcome)
    groups = {**market.groups, "ANY_PROTECTED": market.protected,
              "UNPROTECTED": set(market.students) - market.protected}
    return [{"group": label, "group_population": len(group),
             "group_matched": len(group & matched),
             "group_admission_rate": _ratio(len(group & matched), len(group)),
             "group_share_matched": _ratio(len(group & matched), len(matched)),
             "group_average_rank": _ratio(sum(ranks[s] for s in group), len(group))}
            for label, group in groups.items()]


def institution_outcomes(market, outcome):
    return [{"institution": i, "capacity": market.base["capacities"][i],
             "matched": len(outcome.matching[i]),
             "protected_matched": len(set(outcome.matching[i]) & market.protected)}
            for i in market.institutions]
