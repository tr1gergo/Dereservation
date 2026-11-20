"""Institutional placements and their supporting original-seat contracts."""

from dataclasses import dataclass, field


@dataclass
class Market:
    base: dict
    quotas: dict                  # institution -> original category -> capacity
    original: dict                # institution -> category -> original eligibles
    transfers: dict               # institution -> category -> additional eligibles
    priorities: dict              # institution -> category -> two-tier priority list
    precedence: dict              # institution -> category order
    reserved: set
    groups: dict                  # demographic label -> set of students (may overlap)

    @property
    def students(self):
        return self.base["students"]

    @property
    def institutions(self):
        return self.base["institutions"]

    @property
    def protected(self):
        return set().union(*self.groups.values()) if self.groups else set()

    def eligible(self, i, c):
        return self.original[i][c] | self.transfers[i][c]

    def validate(self):
        """Reject inconsistent capacities, tiers, identities, and strict orders."""
        students, schools = set(self.students), set(self.institutions)
        if len(students) != len(self.students) or len(schools) != len(self.institutions):
            raise ValueError("Student and institution identifiers must be unique")
        if not students or not schools:
            raise ValueError("A market needs students and institutions")
        if set(self.base["student_prefs"]) != students:
            raise ValueError("A preference list is required for every student")
        for s, prefs in self.base["student_prefs"].items():
            if len(prefs) != len(set(prefs)) or not set(prefs) <= schools:
                raise ValueError(f"Invalid strict preferences for {s}")
        for label, group in self.groups.items():
            if not group <= students:
                raise ValueError(f"Unknown students in demographic group {label}")
        for i in self.institutions:
            categories = set(self.quotas[i])
            if any(type(q) is not int or q < 0 for q in self.quotas[i].values()):
                raise ValueError("Category quotas must be nonnegative integers")
            if sum(self.quotas[i].values()) != self.base["capacities"][i]:
                raise ValueError("Original categories must partition physical capacity")
            if (set(self.precedence[i]) != categories or
                    len(self.precedence[i]) != len(categories)):
                raise ValueError("Precedence must be a strict order of all categories")
            merit = self.base["inst_prefs"][i]
            if len(merit) != len(set(merit)) or not set(merit) <= students:
                raise ValueError("Institution priorities must be strict")
            for c in categories:
                e0, ed = self.original[i][c], self.transfers[i][c]
                priority = self.priorities[i][c]
                if e0 & ed or not (e0 | ed) <= set(merit):
                    raise ValueError("Eligibility tiers must be disjoint and acceptable")
                if len(priority) != len(set(priority)) or set(priority) != e0 | ed:
                    raise ValueError("Category priorities must list exactly eligible students")
                tier = [s in ed for s in priority]
                if tier != sorted(tier):
                    raise ValueError("Original eligibles must precede transfer eligibles")
        return self


@dataclass
class Outcome:
    matching: dict                # institution -> admitted student list
    assignments: dict             # institution -> {student: original seat category}
    diagnostics: dict = field(default_factory=dict)
    effective_quotas: dict | None = None
    converted: dict = field(default_factory=dict)  # operationally converted OBC slots
    assignment_source: str = "mechanism"


def validate_placements(market, matching):
    """Physical capacity and mutual acceptability, independent of seat labels."""
    if set(matching) != set(market.institutions):
        raise ValueError("Matching must contain every institution, and no others")
    seen = set()
    for i, cohort in matching.items():
        if len(cohort) > market.base["capacities"][i]:
            raise ValueError(f"Capacity exceeded at {i}")
        for s in cohort:
            if s not in market.base["student_prefs"] or s in seen:
                raise ValueError("Unknown or multiply matched student")
            if i not in market.base["student_prefs"][s] or s not in market.base["inst_prefs"][i]:
                raise ValueError("Individually unacceptable placement")
            seen.add(s)


def validate_contracts(market, outcome):
    """Require one eligible original-seat contract per admitted student."""
    validate_placements(market, outcome.matching)
    for i in market.institutions:
        assigned = outcome.assignments[i]
        if set(assigned) != set(outcome.matching[i]):
            raise ValueError(f"No complete supporting assignment at {i}")
        for c, q in market.quotas[i].items():
            if sum(value == c for value in assigned.values()) > q:
                raise ValueError("Original seat quota exceeded")
        for s, c in assigned.items():
            if c not in market.quotas[i] or s not in market.eligible(i, c):
                raise ValueError("Ineligible supporting contract")
