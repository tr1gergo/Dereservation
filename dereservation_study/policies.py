"""Exact fixed-weight policies and the single-path update in paper Section 8.

All comparisons use Python integers. Lexicographic weights are fixed on the
ENTIRE market, never renormalized for a changing applicant pool. Category
nodes have original capacities; no seat expansion is used by the update.
"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Policy:
    kind: str = "MD"
    lam: Fraction = Fraction(0)

    def __post_init__(self):
        if self.kind not in {"MD", "RP", "Merit", "Demographic", "WD"}:
            raise ValueError(f"Unknown weighted policy: {self.kind}")
        object.__setattr__(self, "lam", Fraction(str(self.lam)))
        if self.lam < 0:
            raise ValueError("lambda must be nonnegative")

    @property
    def name(self):
        return f"WD(lambda={self.lam})" if self.kind == "WD" else self.kind


class WeightedChoice:
    def __init__(self, market, policy=Policy()):
        self.market, self.policy = market, policy
        self.weights, self.bits, self.units = {}, {}, {}
        self.path_updates = 0
        n = len(market.students)
        for i in market.institutions:
            order = [(s, c) for c in market.precedence[i] for s in market.priorities[i][c]]
            unit = 1 << len(order)
            self.units[i] = unit
            self.bits[i] = {edge: 1 << (len(order) - j - 1) for j, edge in enumerate(order)}
            q = min(sum(market.quotas[i].values()), n)
            merit = {s: n - r for r, s in enumerate(market.base["inst_prefs"][i])}
            weights = {}
            for (s, c), bit in self.bits[i].items():
                d = int(c in market.reserved and s in market.original[i][c])
                if policy.kind == "MD":
                    primary = 1
                elif policy.kind == "RP":
                    primary = q + 1 + d
                elif policy.kind == "Merit":
                    primary = q * n + 1 + merit[s]
                elif policy.kind == "Demographic":
                    primary = q + 1 + int(s in market.protected)
                else:
                    # W_lambda = |X| + lambda D(X), followed by the MD tie-break.
                    primary = policy.lam.denominator + policy.lam.numerator * d
                weights[s, c] = primary * unit + bit
            self.weights[i] = weights

    def update(self, i, held, newcomer):
        """Update an optimal student->category assignment after one arrival.

        The residual graph has no negative cycle. A cheapest negative path
        starts at the newcomer and ends at a vacancy or an evicted student.
        Strict relaxation and fixed arc ordering also make the implementation
        deterministic. ``held`` is not mutated.
        """
        if newcomer in held:
            raise ValueError("The arriving student must not already be held")
        self.path_updates += 1
        market, weights = self.market, self.weights[i]
        students = [s for s in market.students if s in held or s == newcomer]
        sink, start = ("sink",), ("s", newcomer)
        arcs = []
        occupants = {c: [] for c in market.quotas[i]}
        for s, c in held.items():
            occupants[c].append(s)
        for s in students:
            for c in market.precedence[i]:
                if (s, c) in weights and held.get(s) != c:
                    arcs.append((("s", s), ("c", c), -weights[s, c]))
        for c in market.precedence[i]:
            for s in occupants[c]:
                arcs.append((("c", c), ("s", s), weights[s, c]))
            if len(occupants[c]) < market.quotas[i][c]:
                arcs.append((("c", c), sink, 0))
        arcs.extend((("s", s), sink, 0) for s in students if s in held)
        dist, parent = {start: 0}, {}
        vertices = len(students) + len(market.quotas[i]) + 1
        for _ in range(vertices - 1):
            changed = False
            for u, v, cost in arcs:
                if u in dist and (v not in dist or dist[u] + cost < dist[v]):
                    dist[v], parent[v] = dist[u] + cost, u
                    changed = True
            if not changed:
                break
        if sink not in dist or dist[sink] >= 0:
            return dict(held)
        path, v = [], sink
        while v != start:
            if v not in parent or len(path) >= vertices:
                raise RuntimeError("Invalid residual path; held assignment was not optimal")
            u = parent[v]
            path.append((u, v))
            v = u
        chosen = dict(held)
        # Remove reversed contracts before installing forward contracts.
        for u, v in path:
            if u[0] == "c" and v[0] == "s":
                del chosen[v[1]]
        for u, v in path:
            if u[0] == "s" and v[0] == "c":
                chosen[u[1]] = v[1]
        return chosen

    def choose(self, i, applicants, engine="incremental"):
        pool = set(applicants)
        if engine == "flow":
            return self.choose_flow(i, pool)
        if engine != "incremental":
            raise ValueError("engine must be incremental or flow")
        held = {}
        for s in self.market.students:
            if s in pool:
                held = self.update(i, held, s)
        return held

    def choose_flow(self, i, applicants):
        """Independent, from-scratch min-cost flow reference implementation.

        One unit per applicant is sent either to a category or an unmatched
        arc. Forcing maximum flow through categories would incorrectly turn
        WD into a cardinality-first policy.
        """
        import networkx as nx

        pool = set(applicants)
        source, sink = ("source",), ("sink",)
        graph = nx.DiGraph()
        graph.add_node(source, demand=-len(pool))
        graph.add_node(sink, demand=len(pool))
        for c, q in self.market.quotas[i].items():
            graph.add_edge(("c", c), sink, capacity=q, weight=0)
        for s in self.market.students:
            if s not in pool:
                continue
            graph.add_edge(source, ("s", s), capacity=1, weight=0)
            graph.add_edge(("s", s), sink, capacity=1, weight=0)
            for c in self.market.precedence[i]:
                if (s, c) in self.weights[i]:
                    graph.add_edge(("s", s), ("c", c), capacity=1, weight=-self.weights[i][s, c])
        flow = nx.min_cost_flow(graph)
        return {s: c for s in self.market.students if s in pool for c in self.market.precedence[i]
                if flow[("s", s)].get(("c", c), 0)}


def compliance_frontier(market, institution, applicants=None):
    """Exact local D*(k) at every feasible headcount, not a DA frontier.

    Each solution uses a fixed pool at ONE institution. Discrete concavity can
    be checked from the reported marginal D increments. Other schools and
    students' placement preferences do not enter this local bound.
    """
    import networkx as nx

    i = institution
    pool = set(market.students if applicants is None else applicants)
    oracle = WeightedChoice(market)
    max_k = len(oracle.choose(i, pool))
    unit = oracle.units[i]
    records = []
    for k in range(max_k + 1):
        graph = nx.DiGraph()
        source, sink = ("source",), ("sink",)
        graph.add_node(source, demand=-k)
        graph.add_node(sink, demand=k)
        for c, q in market.quotas[i].items():
            graph.add_edge(("c", c), sink, capacity=q, weight=0)
        for s in market.students:
            if s not in pool:
                continue
            graph.add_edge(source, ("s", s), capacity=1, weight=0)
            for c in market.precedence[i]:
                if (s, c) in oracle.bits[i]:
                    d = int(c in market.reserved and s in market.original[i][c])
                    graph.add_edge(("s", s), ("c", c), capacity=1,
                                   weight=-(d * unit + oracle.bits[i][s, c]))
        flow = nx.min_cost_flow(graph)
        assignment = {s: c for s in market.students if s in pool for c in market.precedence[i]
                      if flow[("s", s)].get(("c", c), 0)}
        d = sum(c in market.reserved and s in market.original[i][c] for s, c in assignment.items())
        records.append({"institution": i, "headcount": k, "original_reserved_seats": d,
                        "assignment": assignment})
    return records
