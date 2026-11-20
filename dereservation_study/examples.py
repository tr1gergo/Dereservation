"""Small explicit examples transcribed from the supplied manuscript."""

from .model import Market


def _single_school(students, quotas, original, transfers, regime):
    priorities = {c: [s for s in students if s in original[c]] +
                     [s for s in students if s in transfers[c]] for c in quotas}
    reserved = set(quotas) - {"g"}
    base = {"students": students, "institutions": [0], "capacities": {0: sum(quotas.values())},
            "student_prefs": {s: [0] for s in students}, "inst_prefs": {0: students},
            "metadata": {"regime": regime}}
    return Market(base, {0: quotas}, {0: original}, {0: transfers}, {0: priorities},
                  {0: list(quotas)}, reserved, {c: original[c] for c in reserved}).validate()


def bt_avoidable_vacancies():
    """Example 2.1: DA-BT/multi-run admit six; MD admits all eight."""
    students = list(range(1, 9))
    return _single_school(students, {"g": 3, "OBC": 3, "SC": 1, "ST": 1},
                          {"g": set(students), "OBC": {3, 4}, "SC": {5}, "ST": {6}},
                          {"g": set(), "OBC": {1, 2, 5, 6, 7, 8}, "SC": set(), "ST": set()}, "indian")


def compliance_tradeoff():
    """Example 6.4: three admissions with D=1 versus two with D=2."""
    return _single_school(["a", "b", "c"], {"U": 1, "V": 1, "W": 1},
                          {"U": {"a"}, "V": {"b"}, "W": {"b"}},
                          {"U": {"c"}, "V": {"a"}, "W": set()}, "overlapping")
