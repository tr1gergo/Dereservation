"""DA mechanisms, Indian transfer rules, and adapters to the original backends."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

from .model import Outcome
from .policies import Policy, WeightedChoice


def _legacy():
    from src import algorithms

    return algorithms


def _require_indian(market):
    if market.base["metadata"].get("regime") != "indian":
        raise ValueError("DA with vertical quotas, multi-run DA, and BT require the Indian regime")
    if any(market.precedence[i] != ["g", "OBC", "SC", "ST"] for i in market.institutions):
        raise ValueError("Indian benchmarks require g > OBC > SC > ST precedence")
    for i in market.institutions:
        e0, ed = market.original[i], market.transfers[i]
        merit = market.base["inst_prefs"][i]
        if set(market.quotas[i]) != {"g", "OBC", "SC", "ST"} or e0["g"] != set(merit):
            raise ValueError("Indian benchmarks need open seats available to every acceptable student")
        if any(e0[a] & e0[b] for a, b in (("OBC", "SC"), ("OBC", "ST"), ("SC", "ST"))):
            raise ValueError("Indian reserved social types must be disjoint")
        if any(ed[c] for c in ("g", "SC", "ST")) or ed["OBC"] != set(merit) - e0["OBC"]:
            raise ValueError("Only OBC seats may transfer, with universal additional access")
        for c in market.quotas[i]:
            expected = [s for s in merit if s in e0[c]] + [s for s in merit if s in ed[c]]
            if market.priorities[i][c] != expected:
                raise ValueError("Indian benchmarks require common merit within eligibility tiers")


class IndianChoice:
    """Open-first sequential choice; BT resets quotas on EVERY invocation."""

    def __init__(self, market, backward=False, quotas=None):
        _require_indian(market)
        self.market, self.backward = market, backward
        self.quotas = deepcopy(market.quotas if quotas is None else quotas)
        self.last_quotas, self.last_converted = {}, {}

    def choose(self, i, applicants):
        pool, market = set(applicants), self.market
        q = dict(self.quotas[i])
        merit = market.base["inst_prefs"][i]
        while True:
            selected, remaining, by_category = {}, set(pool), {}
            for c in market.precedence[i]:
                chosen = [s for s in merit if s in remaining and s in market.original[i][c]][:q[c]]
                by_category[c] = chosen
                selected.update({s: c for s in chosen})
                remaining.difference_update(chosen)
            vacancies = q["OBC"] - len(by_category["OBC"])
            if not self.backward or not vacancies:
                break
            q["g"] += vacancies
            q["OBC"] -= vacancies
        # Keep original physical seat labels, including converted OBC positions.
        # Original open slots are consumed first, then converted OBC slots.
        converted = set(by_category["g"][market.quotas[i]["g"]:])
        for s in converted:
            selected[s] = "OBC"
        self.last_quotas[i], self.last_converted[i] = q, converted
        return selected


def deferred_acceptance(market, oracle, engine="incremental"):
    """Student-proposing DA over schools; category assignments stay internal."""
    held = {i: {} for i in market.institutions}
    next_proposal = {s: 0 for s in market.students}
    free = deque(market.students)
    proposals = 0
    while free:
        s = free.popleft()
        prefs = market.base["student_prefs"][s]
        if next_proposal[s] >= len(prefs):
            continue
        i = prefs[next_proposal[s]]
        next_proposal[s] += 1
        proposals += 1
        old = held[i]
        if isinstance(oracle, WeightedChoice) and engine == "incremental":
            chosen = oracle.update(i, old, s)
        else:
            pool = set(old) | {s}
            chosen = (oracle.choose(i, pool, engine=engine) if isinstance(oracle, WeightedChoice)
                      else oracle.choose(i, pool))
        held[i] = chosen
        # At most one new student and one eviction for weighted rules; the
        # generic loop also supports choice rules that reject several students.
        for rejected in market.students:
            if (rejected in old or rejected == s) and rejected not in chosen:
                if next_proposal[rejected] < len(market.base["student_prefs"][rejected]):
                    free.append(rejected)
    diagnostics = {"proposals": proposals, "da_rounds": 1}
    if isinstance(oracle, WeightedChoice):
        diagnostics["path_updates"] = oracle.path_updates
    effective = deepcopy(oracle.last_quotas) if isinstance(oracle, IndianChoice) else None
    converted = deepcopy(oracle.last_converted) if isinstance(oracle, IndianChoice) else {}
    # Schools receiving no proposals still have their unchanged original quotas.
    if effective is not None:
        for i in market.institutions:
            effective.setdefault(i, dict(oracle.quotas[i]))
    return Outcome({i: [s for s in market.students if s in held[i]] for i in market.institutions},
                   held, diagnostics, effective, converted)


def multi_run_da(market):
    """Transfer unfilled OBC quotas between complete fresh DA runs."""
    _require_indian(market)
    quotas, rounds, proposals = deepcopy(market.quotas), 0, 0
    while True:
        result = deferred_acceptance(market, IndianChoice(market, quotas=quotas))
        rounds += 1
        proposals += result.diagnostics["proposals"]
        changed = False
        for i in market.institutions:
            # Converted open slots are labeled OBC in the supporting assignment;
            # subtract those before calculating remaining actual OBC occupancy.
            obc_occupied = sum(c == "OBC" for c in result.assignments[i].values()) - len(result.converted.get(i, ()))
            vacant = quotas[i]["OBC"] - obc_occupied
            if vacant:
                quotas[i]["g"] += vacant
                quotas[i]["OBC"] -= vacant
                changed = True
        if not changed:
            result.diagnostics.update(da_rounds=rounds, proposals=proposals)
            return result


class LegacyChoice:
    """Replay the original choice routine on a single institution's pool.

    The old Soft-Reserves has a fallback for overlapping types; no general
    substitutes/stability theorem is asserted for that fallback.
    """
    def __init__(self, market, kind):
        self.market, self.kind = market, kind

    def choose(self, i, applicants):
        legacy = _legacy()
        base = self.market.base
        pool = set(applicants)
        order = [s for s in self.market.students if s in pool]
        if self.kind == "DA-open":
            chosen = [s for s in base["inst_prefs"][i] if s in pool][:base["capacities"][i]]
        elif self.kind == "Smart-Hard-Reserves":
            ranks = {s: r for r, s in enumerate(base["inst_prefs"][i])}
            chosen = legacy._smart_hard_reserve_choice(base, i, order, ranks)
        else:
            local = {**base, "students": order, "institutions": [i],
                     "capacities": {i: base["capacities"][i]},
                     "student_prefs": {s: [i] for s in order},
                     "inst_prefs": {i: base["inst_prefs"][i]}}
            chosen = legacy.alg_soft_reserves_da(local)[i]
        return {s: None for s in chosen}


@dataclass(frozen=True)
class Algorithm:
    name: str
    kind: str
    policy: Policy | None = None
    engine: str = "incremental"

    def run(self, market):
        if self.policy is not None:
            return deferred_acceptance(market, WeightedChoice(market, self.policy), self.engine)
        if self.kind == "Multi-run DA":
            return multi_run_da(market)
        if self.kind in {"DA", "BT"}:
            return deferred_acceptance(market, IndianChoice(market, backward=self.kind == "BT"))
        legacy = _legacy()
        fn = {"DA-open": legacy.alg0_simple_da,
              "Smart-Hard-Reserves": legacy.alg_smart_hard_reserves_da,
              "Soft-Reserves": legacy.alg_soft_reserves_da}[self.kind]
        # The experiment runner isolates every algorithm's input before timing.
        matching = fn(market.base)
        return Outcome(matching, {}, assignment_source="posthoc_MD_support",
                       diagnostics={"da_rounds": 1})

    def oracle(self, market, outcome=None):
        if self.policy is not None:
            return WeightedChoice(market, self.policy)
        if self.kind in {"DA", "BT", "Multi-run DA"}:
            q = outcome.effective_quotas if self.kind == "Multi-run DA" and outcome is not None else None
            return IndianChoice(market, backward=self.kind == "BT", quotas=q)
        return LegacyChoice(market, self.kind)


def algorithm_suite(regime, lambdas=(), include_merit=True,
                    include_demographic=False):
    if regime == "indian":
        algorithms = [Algorithm(name, name) for name in ("DA", "Multi-run DA", "BT")]
    elif regime == "overlapping":
        algorithms = [Algorithm(name, name) for name in ("DA-open", "Smart-Hard-Reserves", "Soft-Reserves")]
    else:
        raise ValueError("regime must be indian or overlapping")
    policies = [Policy("MD"), Policy("RP")]
    if include_merit:
        policies.append(Policy("Merit"))
    if include_demographic:
        policies.append(Policy("Demographic"))
    policies.extend(Policy("WD", lam) for lam in lambdas)
    return algorithms + [Algorithm(policy.name, "weighted", policy) for policy in policies]


def attach_legacy_support(market, outcome):
    """Supply a deterministic feasible seat labeling AFTER mechanism timing.

    Old routines return cohorts only. When the cohort has no support, retain
    its actual matching and let evaluation mark it ineligible; never silently
    repair the benchmark's student placements.
    """
    if outcome.assignment_source == "posthoc_MD_support":
        oracle = WeightedChoice(market)
        outcome.assignments = {i: oracle.choose(i, outcome.matching[i]) for i in market.institutions}
    return outcome
