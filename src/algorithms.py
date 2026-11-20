import copy

# --- Core Stateful DA Algorithm ---

def run_da(students, institutions, capacities, student_prefs, inst_prefs,
           forbidden_edges=None, initial_matches=None, initial_proposals=None):
    """
    A stateful Student-Proposing Deferred Acceptance algorithm.
    Can be resumed from a previous state by passing initial_matches and initial_proposals.
    """
    if forbidden_edges is None:
        forbidden_edges = set()

    # State tracking
    proposals_made = initial_proposals.copy() if initial_proposals else {s: 0 for s in students}
    matches = {i: [] for i in institutions}
    if initial_matches:
        for i, matched_students in initial_matches.items():
            matches[i] = list(matched_students)

    # Rank dictionaries for O(1) comparison: smaller rank is better
    inst_rank = {i: {s: rank for rank, s in enumerate(prefs)} for i, prefs in inst_prefs.items()}

    # Identify free students who still have institutions to propose to
    free_students = []
    for s in students:
        matched_to_s = any(s in matches[i] for i in institutions)
        if not matched_to_s and proposals_made[s] < len(student_prefs[s]):
            free_students.append(s)

    exhausted_students = set(s for s in students if proposals_made[s] >= len(student_prefs[s]) and not any(s in matches[i] for i in institutions))

    while free_students:
        s = free_students.pop(0)

        # Propose down the preference list
        proposed = False
        while proposals_made[s] < len(student_prefs[s]):
            i = student_prefs[s][proposals_made[s]]
            proposals_made[s] += 1

            # Skip forbidden edges strictly
            if (s, i) in forbidden_edges:
                continue

            # If s is acceptable to i
            if s in inst_rank[i]:
                matches[i].append(s)
                # Sort current matches at i by i's preferences (best to worst)
                matches[i].sort(key=lambda x: inst_rank[i][x])

                # Over capacity? Reject the worst
                if len(matches[i]) > capacities[i]:
                    rejected_s = matches[i].pop(-1)
                    if proposals_made[rejected_s] < len(student_prefs[rejected_s]):
                        free_students.append(rejected_s)
                    else:
                        exhausted_students.add(rejected_s)

                proposed = True
                break # Wait for next round if rejected later

        if not proposed and not any(s in matches[i] for i in institutions):
            exhausted_students.add(s)

    return matches, proposals_made, exhausted_students


# --- ALG 0: Standard DA ---

def alg0_simple_da(instance):
    """Computes the student-optimal stable matching."""
    M, _, _ = run_da(
        instance['students'], instance['institutions'], instance['capacities'],
        instance['student_prefs'], instance['inst_prefs']
    )
    return M


# --- Benchmark DA Variants with Reserves ---

def _student_categories(instance, s):
    return set(instance["categories"].get(s, set()))


def _can_match_smart_reserves(instance, institution, students):
    """
    Check whether the selected students can fill the school's lower-bound
    category reserves plus its open reserve seats.
    """

    capacity = instance["capacities"][institution]

    if len(students) > capacity:
        return False

    lower = instance["bounds"][institution]["lower"]
    slots = []
    category_slots = {}

    lower_total = 0
    for category in instance["all_categories"]:
        reserve_count = max(0, int(lower.get(category, 0)))
        lower_total += reserve_count
        category_slots[category] = []

        for index in range(reserve_count):
            category_slots[category].append(len(slots))
            slots.append(("category", category, index))

    # The smart-reserve model assumes lower reserves fit within capacity.
    # The guard above enforces the school capacity if overlapping generated
    # categories make the reserve sum larger.
    open_reserve_count = max(0, capacity - lower_total)
    open_slots = []

    for index in range(open_reserve_count):
        open_slots.append(len(slots))
        slots.append(("open", None, index))

    if not students:
        return True

    if not slots:
        return False

    adjacency = {}
    for student in students:
        edges = list(open_slots)

        for category in _student_categories(instance, student):
            edges.extend(category_slots.get(category, []))

        if not edges:
            return False

        adjacency[student] = edges

    matched_student_by_slot = {}

    def augment(student, seen_slots):
        for slot in adjacency[student]:
            if slot in seen_slots:
                continue

            seen_slots.add(slot)
            current_student = matched_student_by_slot.get(slot)

            if current_student is None or augment(current_student, seen_slots):
                matched_student_by_slot[slot] = student
                return True

        return False

    for student in students:
        if not augment(student, set()):
            return False

    return True


def _smart_hard_reserve_choice(instance, institution, proposals, inst_rank):
    """
    Priority-greedy smart-reserve choice from the current proposal set.
    """

    chosen = []

    for student in sorted(proposals, key=lambda s: inst_rank[s]):
        candidate = chosen + [student]

        if _can_match_smart_reserves(
            instance=instance,
            institution=institution,
            students=candidate,
        ):
            chosen = candidate

    return chosen


def alg_smart_hard_reserves_da(instance):
    """
    Deferred acceptance with smart hard reserves.

    Each school chooses from its current proposals by scanning students from
    highest to lowest priority and keeping a student exactly when the kept set
    remains matchable to category lower-bound reserve seats and open seats.
    """

    students = instance["students"]
    institutions = instance["institutions"]
    student_prefs = instance["student_prefs"]
    inst_prefs = instance["inst_prefs"]

    matches = {i: [] for i in institutions}
    proposals_made = {s: 0 for s in students}
    free_students = list(students)
    inst_rank = {
        i: {s: rank for rank, s in enumerate(inst_prefs[i])}
        for i in institutions
    }

    while free_students:
        s = free_students.pop(0)

        while proposals_made[s] < len(student_prefs[s]):
            i = student_prefs[s][proposals_made[s]]
            proposals_made[s] += 1

            if s not in inst_rank[i]:
                continue

            proposal_set = list(matches[i])

            if s not in proposal_set:
                proposal_set.append(s)

            chosen = _smart_hard_reserve_choice(
                instance=instance,
                institution=i,
                proposals=proposal_set,
                inst_rank=inst_rank[i],
            )
            chosen_set = set(chosen)
            rejected_students = [
                student
                for student in proposal_set
                if student not in chosen_set
            ]

            matches[i] = chosen

            for rejected in rejected_students:
                if proposals_made[rejected] < len(student_prefs[rejected]):
                    free_students.append(rejected)

            break

    return matches


def _category_counts(assigned, instance):
    return {
        c: sum(1 for s in assigned if c in _student_categories(instance, s))
        for c in instance["all_categories"]
    }


def _worst_student(students, inst_rank):
    return max(students, key=lambda s: inst_rank[s])


def _worst_upper_violation_student(assigned, instance, inst_rank, upper):
    counts = _category_counts(assigned, instance)
    over_categories = {
        c for c, count in counts.items()
        if count > upper.get(c, float("inf"))
    }

    candidates = [
        s for s in assigned
        if _student_categories(instance, s) & over_categories
    ]

    if not candidates:
        return None

    return _worst_student(candidates, inst_rank)


def _worst_lower_safe_student(assigned, instance, inst_rank, lower):
    counts = _category_counts(assigned, instance)

    candidates = []
    for s in assigned:
        categories = _student_categories(instance, s)

        if all(
            counts[c] - 1 >= lower.get(c, 0)
            for c in categories
        ):
            candidates.append(s)

    if candidates:
        return _worst_student(candidates, inst_rank)

    # Overlapping categories can make the single-category lower-quota
    # existence argument fail. Fall back to termination-preserving DA rejection.
    return _worst_student(assigned, inst_rank)


def alg_soft_reserves_da(instance):
    students = instance["students"]
    institutions = instance["institutions"]
    capacities = instance["capacities"]
    student_prefs = instance["student_prefs"]
    inst_prefs = instance["inst_prefs"]
    bounds = instance["bounds"]

    matches = {i: [] for i in institutions}
    proposals_made = {s: 0 for s in students}
    free_students = list(students)
    inst_rank = {
        i: {s: rank for rank, s in enumerate(inst_prefs[i])}
        for i in institutions
    }

    while free_students:
        s = free_students.pop(0)

        while proposals_made[s] < len(student_prefs[s]):
            i = student_prefs[s][proposals_made[s]]
            proposals_made[s] += 1

            if s not in inst_rank[i]:
                continue

            assigned = matches[i]
            assigned.append(s)
            assigned.sort(key=lambda x: inst_rank[i][x])

            upper = bounds[i]["upper"]
            lower = bounds[i]["lower"]

            if len(assigned) > capacities[i]:
                rejected = _worst_upper_violation_student(
                    assigned,
                    instance,
                    inst_rank[i],
                    upper,
                )

                if rejected is None:
                    rejected = _worst_lower_safe_student(
                        assigned,
                        instance,
                        inst_rank[i],
                        lower,
                    )

                assigned.remove(rejected)
                if proposals_made[rejected] < len(student_prefs[rejected]):
                    free_students.append(rejected)

            break

    return matches


# --- ALG 1: Bound-Driven DA (Diversity Only) ---

def alg1_bound_driven_da(instance):
    """Dynamically updates forbidden edges to satisfy category upper/lower bounds."""
    students = instance['students']
    institutions = instance['institutions']
    capacities = instance['capacities']
    student_prefs = instance['student_prefs']
    inst_prefs = instance['inst_prefs']
    categories_dict = instance['categories']
    all_categories = instance['all_categories']
    base_bounds = instance['bounds']

    # We need M_init for the inverse upper bounds u_neq_c
    M_init, _, _ = run_da(students, institutions, capacities, student_prefs, inst_prefs)

    relaxation_level = 0

    while True:
        # 1. Initialization
        F = set()
        j = 1

        # Run standard DA to find student-optimal M_0
        M_0, proposals, exhausted = run_da(students, institutions, capacities, student_prefs, inst_prefs)
        M = copy.deepcopy(M_0)

        # \rural and \rurals definitions
        rural_insts = {i for i in institutions if len(M_0[i]) < capacities[i]}
        rurals_students = set(exhausted)

        # Bounds logic with current relaxation
        u_c = {i: {c: base_bounds[i]['upper'].get(c, float('inf')) + relaxation_level for c in all_categories} for i in institutions}
        l_c = {i: {c: max(0, base_bounds[i]['lower'].get(c, 0) - relaxation_level) for c in all_categories} for i in institutions}
        u_neq_c = {i: {c: len(M_init[i]) - l_c[i][c] for c in all_categories} for i in institutions}

        # Check if any i in \rural does not satisfy a diversity bound
        rural_violation = False
        for i in rural_insts:
            for c in all_categories:
                n_i_c = sum(1 for s in M_0[i] if c in categories_dict[s])
                if n_i_c < l_c[i][c] or n_i_c > u_c[i][c]:
                    rural_violation = True
                    break
            if rural_violation:
                break

        if rural_violation:
            relaxation_level += 1
            continue

        inner_solution_found = False
        while True:
            # 2. Check Bounds
            violated_i, violated_c, bound_type = None, None, None
            for i in institutions:
                n_i_c_map = {c: sum(1 for s in M[i] if c in categories_dict[s]) for c in all_categories}
                for c in all_categories:
                    if n_i_c_map[c] > u_c[i][c]:
                        violated_i, violated_c, bound_type = i, c, 'u_c'
                        break
                    n_i_neq_c = len(M[i]) - n_i_c_map[c]
                    if n_i_neq_c > u_neq_c[i][c]:
                        violated_i, violated_c, bound_type = i, c, 'u_neq_c'
                        break
                if violated_i is not None:
                    break

            # If all bounds are satisfied, we found our matching
            if violated_i is None:
                inner_solution_found = True
                break

            # 3. Identify Violation & Expand F
            i = violated_i
            c = violated_c
            inst_rank = {s: rank for rank, s in enumerate(inst_prefs[i])}

            if bound_type == 'u_c':
                # Least preferred student of category c currently in M[i]
                c_students = [s for s in M[i] if c in categories_dict[s]]
                target_s = max(c_students, key=lambda s: inst_rank[s])
            else:
                # Least preferred student of category != c currently in M[i]
                not_c_students = [s for s in M[i] if c not in categories_dict[s]]
                target_s = max(not_c_students, key=lambda s: inst_rank[s])

            # Expand forbidden edge set
            target_rank = inst_rank[target_s]
            for s in inst_prefs[i]:
                if inst_rank[s] >= target_rank:
                    F.add((s, i))

            # 4. Break Marriages & Resume DA
            broken = [s for s in M[i] if (s, i) in F]
            for s in broken:
                M[i].remove(s)

            M, proposals, exhausted = run_da(
                students, institutions, capacities, student_prefs, inst_prefs,
                forbidden_edges=F, initial_matches=M, initial_proposals=proposals
            )

            # Check strict termination criteria:
            # 1. Any student NOT in \rurals exhausts their list
            newly_exhausted = exhausted - rurals_students
            if newly_exhausted:
                break # "stop and output no solution" for this inner loop iteration

            # 2. Applies to an institution in \rural
            # (By Rural Hospitals Theorem, any valid stable matching must match the exact same subset to rural hospitals)
            rural_violation_inner = False
            for ri in rural_insts:
                if set(M[ri]) != set(M_0[ri]):
                    rural_violation_inner = True
                    break

            if rural_violation_inner:
                break # "stop and output no solution"

        if inner_solution_found:
            return relaxation_level, M
        else:
            relaxation_level += 1


# --- Subroutine: DA with Forbidden Edges (Single Institution Stable Sets) ---

def compute_stable_sets(i, instance):
    """Computes the collection of stable sets E_i^1 ... E_i^{k_i} for institution i."""
    students = instance['students']
    institutions = instance['institutions']
    capacities = instance['capacities']
    student_prefs = instance['student_prefs']
    inst_prefs = instance['inst_prefs']

    inst_rank = {s: rank for rank, s in enumerate(inst_prefs[i])}

    # 1. Initialization
    M_0, _, exhausted_0 = run_da(students, institutions, capacities, student_prefs, inst_prefs)

    # 2. Base Case Check
    if not M_0[i]:
        return [[]]

    stable_sets = [list(M_0[i])]
    F = set()
    current_M = copy.deepcopy(M_0)

    # Track invariants to verify stability lattice bounds
    rural_insts = {inst for inst in institutions if len(M_0[inst]) < capacities[inst]}
    rurals_students = set(exhausted_0)

    # 3. Iterative Step
    while True:
        if not current_M[i]:
            break

        s_worst = max(current_M[i], key=lambda s: inst_rank[s])
        worst_rank = inst_rank[s_worst]

        for s in inst_prefs[i]:
            if inst_rank[s] >= worst_rank:
                F.add((s, i))

        # Re-run DA
        M_next, _, exhausted_next = run_da(students, institutions, capacities, student_prefs, inst_prefs, forbidden_edges=F)

        # 4. Teration Condition: No stable matching exists
        # This occurs if the bounds of the stable matching lattice are breached
        newly_exhausted = exhausted_next - rurals_students
        if newly_exhausted:
            break

        rural_violation = False
        for ri in rural_insts:
            if set(M_next[ri]) != set(M_0[ri]):
                rural_violation = True
                break

        if rural_violation:
            break

        if not M_next[i]:
            stable_sets.append([])
            break

        stable_sets.append(list(M_next[i]))
        current_M = M_next

    distinct_sets = []
    for s_set in stable_sets:
        if set(s_set) not in [set(ds) for ds in distinct_sets]:
            distinct_sets.append(s_set)

    return distinct_sets


def alg2_LP(instance, g_i_func, objective_type, M_DA=None):
    """Computes Minimum Weight Stable Matching via Gurobi with optimized model building."""
    import gurobipy as gp
    from gurobipy import GRB

    institutions = instance['institutions']
    students = instance['students']
    capacities = instance['capacities']
    student_prefs = instance['student_prefs']
    inst_prefs = instance['inst_prefs']

    # ------------------------------------------------------------------
    # Step 1: Compute Stable Sets
    # ------------------------------------------------------------------
    stable_sets_dict = {
        i: compute_stable_sets(i, instance)
        for i in institutions
    }

    # ------------------------------------------------------------------
    # Step 2: Translate g_i into edge weights
    # ------------------------------------------------------------------
    weights = {}

    for i in institutions:
        E_i = stable_sets_dict[i][::-1]  # student-pessimal -> student-optimal

        if not E_i or not E_i[0]:
            for s in students:
                weights[(s, i)] = 0
            continue

        inst_rank = {s: rank for rank, s in enumerate(inst_prefs[i])}

        for E_i_j in E_i:
            if not E_i_j:
                continue

            s_j = max(E_i_j, key=lambda s: inst_rank[s])

            M_dummy = {i: E_i_j}

            g_val = g_i_func(
                i,
                M_dummy,
                objective_type,
                instance=instance
            )

            sum_w = sum(
                weights.get((e, i), 0)
                for e in E_i_j
                if e != s_j
            )

            weights[(s_j, i)] = g_val - sum_w

        for s in inst_prefs[i]:
            if (s, i) not in weights:
                weights[(s, i)] = 0

    # ------------------------------------------------------------------
    # Step 3: Solve via Gurobi
    # ------------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("MinWeightStableMatching", env=env)

    x = {}

    inst_copies = {}
    copy_to_original = {}

    copy_idx = 0
    for i in institutions:
        inst_copies[i] = []
        for _ in range(capacities[i]):
            inst_copies[i].append(copy_idx)
            copy_to_original[copy_idx] = i
            copy_idx += 1

    all_copies = list(copy_to_original.keys())

    student_prefs_copies = {s: [] for s in students}

    for s in students:
        k = 1
        for i in student_prefs[s]:
            for i_j in inst_copies[i]:
                student_prefs_copies[s].append((i_j, k))
                k += 1

    obj_expr = gp.LinExpr()

    for s in students:
        for i_j, k in student_prefs_copies[s]:

            orig_i = copy_to_original[i_j]

            w = weights.get((s, orig_i), 0)

            penalty = -1.0 / (k * len(students) * 100.0)

            x[s, i_j] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{s}_{i_j}"
            )

            obj_expr += (w + penalty) * x[s, i_j]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------
    if M_DA is not None:

        used_copies = set()

        for i, assigned_students in M_DA.items():

            sorted_assigned = sorted(
                assigned_students,
                key=lambda s: inst_prefs[i].index(s)
            )

            for s in sorted_assigned:

                for i_j in inst_copies[i]:

                    if i_j not in used_copies:
                        x[s, i_j].Start = 1.0
                        used_copies.add(i_j)
                        break

    # ------------------------------------------------------------------
    # Acceptable students
    # ------------------------------------------------------------------
    valid_students_for_inst = {}

    for i in institutions:
        valid_students_for_inst[i] = [
            s for s in inst_prefs[i]
            if i in student_prefs[s]
        ]

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # each student matched at most once
    for s in students:
        model.addConstr(
            gp.quicksum(
                x[s, i_j]
                for i_j, _ in student_prefs_copies[s]
            ) <= 1
        )

    # each copy matched at most once
    for i_j in all_copies:

        orig_i = copy_to_original[i_j]

        model.addConstr(
            gp.quicksum(
                x[s, i_j]
                for s in valid_students_for_inst[orig_i]
            ) <= 1
        )

    # stability constraints
    for s in students:

        better_copies_expr = gp.LinExpr()

        for i_j, _ in student_prefs_copies[s]:

            orig_i = copy_to_original[i_j]

            if (
                orig_i not in student_prefs[s]
                or s not in inst_prefs[orig_i]
            ):
                continue

            better_students_expr = gp.LinExpr()

            for s_prime in valid_students_for_inst[orig_i]:
                if s_prime == s:
                    break
                better_students_expr.add(x[s_prime, i_j])

            model.addConstr(
                x[s, i_j]
                + better_students_expr
                + better_copies_expr
                >= 1
            )

            better_copies_expr.add(x[s, i_j])

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # Recover matching
    # ------------------------------------------------------------------
    M_out = {i: [] for i in institutions}

    if model.status == GRB.OPTIMAL:

        for s in students:

            for i_j, _ in student_prefs_copies[s]:

                if x[s, i_j].X > 0.5:
                    M_out[copy_to_original[i_j]].append(s)

    total_obj = sum(
        g_i_func(
            i,
            M_out,
            objective_type,
            instance=instance
        )
        for i in institutions
    )

    return total_obj, M_out


# --- ALG 3: Extension to Max Violation Case ---

def alg3_LP(instance, g_i_func, objective_type):
    """
    Repeatedly relax diversity bounds until a feasible minimum-weight
    stable matching is found.
    """
    current_instance = copy.deepcopy(instance)
    relaxation_steps = 0

    while True:

        obj_val, M = alg2_LP(
            current_instance,
            g_i_func,
            objective_type
        )

        # Families objective never requires relaxation.
        if objective_type == "families" or obj_val <= 0:
            return relaxation_steps, M

        # Relax diversity bounds
        relaxation_steps += 1

        for i in current_instance["institutions"]:
            for c in current_instance["all_categories"]:

                current_instance["bounds"][i]["lower"][c] = max(
                    0,
                    current_instance["bounds"][i]["lower"][c] - 1
                )

                current_instance["bounds"][i]["upper"][c] += 1


# --- Verification Module ---

def verify_stability(M, instance):
    """Checks for any blocking pairs in the given matching."""
    students = instance['students']
    institutions = instance['institutions']
    capacities = instance['capacities']
    student_prefs = instance['student_prefs']
    inst_prefs = instance['inst_prefs']

    student_match = {s: None for s in students}
    for i, matched in M.items():
        for s in matched:
            student_match[s] = i

    inst_rank = {i: {s: rank for rank, s in enumerate(prefs)} for i, prefs in inst_prefs.items()}
    student_rank = {s: {i: rank for rank, i in enumerate(prefs)} for s, prefs in student_prefs.items()}

    for s in students:
        for i in institutions:
            if student_match[s] == i:
                continue

            if i not in student_rank[s] or s not in inst_rank[i]:
                continue # Unacceptable

            # Does s strictly prefer i?
            s_current_i = student_match[s]
            s_prefers_i = (s_current_i is None) or (student_rank[s][i] < student_rank[s][s_current_i])

            if s_prefers_i:
                # Does i prefer s or have empty seats?
                if len(M[i]) < capacities[i]:
                    return False # Blocking pair

                worst_s_in_i = max(M[i], key=lambda x: inst_rank[i][x])
                if inst_rank[i][s] < inst_rank[i][worst_s_in_i]:
                    return False # Blocking pair

    return True

# Rotation based version of ALG2
import networkx as nx

def min_weight_stable_fast(
    students,
    all_copies,
    student_prefs_copies,
    s_rank_dict,
    inst_prefs_copies,
    i_rank_dict,
    W_matrix,
):
    """
    Core subroutine for minimum-weight stable matching in a one-to-one market.

    Parameters
    ----------
    students : list

    all_copies : list

    student_prefs_copies : dict

    s_rank_dict : dict

    inst_prefs_copies : dict

    i_rank_dict : dict

    W_matrix : dict
        Edge weight dictionary:
            W_matrix[(student, institution_copy)] -> weight
    """

    ####################################################################
    # Step 1. Student-optimal stable matching
    ####################################################################

    free_students = list(students)

    proposals_idx = {
        s: 0
        for s in students
    }

    M_student = {
        s: None
        for s in students
    }

    M_copy = {
        i: None
        for i in all_copies
    }

    while free_students:

        s = free_students.pop()

        if proposals_idx[s] >= len(student_prefs_copies[s]):
            continue

        i = student_prefs_copies[s][proposals_idx[s]]
        proposals_idx[s] += 1

        current = M_copy[i]

        if current is None:
            M_copy[i] = s
            M_student[s] = i

        elif i_rank_dict[i][s] < i_rank_dict[i][current]:

            M_copy[i] = s
            M_student[s] = i

            free_students.append(current)

        else:
            free_students.append(s)

    ####################################################################
    # Step 2. Enumerate rotations
    ####################################################################

    rotations = []

    current_student = M_student.copy()
    current_copy = M_copy.copy()

    proposal_pointer = {
        s: s_rank_dict[s][current_student[s]] + 1
        for s in students
        if current_student[s] is not None
    }

    while True:

        next_copy = {}
        next_student = {}

        for s in students:

            if current_student[s] is None:
                continue

            idx = proposal_pointer[s]

            while idx < len(student_prefs_copies[s]):

                i = student_prefs_copies[s][idx]

                worst = current_copy[i]

                if i_rank_dict[i][s] < i_rank_dict[i][worst]:
                    next_copy[s] = i
                    next_student[s] = worst
                    proposal_pointer[s] = idx
                    break

                idx += 1

            proposal_pointer[s] = idx

        if not next_student:
            break

        visited = set()

        found = False

        for start in next_student:

            if start in visited:
                continue

            path = []
            index = {}

            cur = start

            while cur not in visited and cur in next_student:

                visited.add(cur)

                index[cur] = len(path)

                path.append(cur)

                cur = next_student[cur]

            if cur not in index:
                continue

            cycle = path[index[cur]:]

            rho = {
                "s_old": {},
                "s_new": {},
                "i_old": {},
                "i_new": {},
            }

            weight = 0

            for s in cycle:

                old_i = current_student[s]
                new_i = next_copy[s]

                rho["s_old"][s] = old_i
                rho["s_new"][s] = new_i
                rho["i_old"][old_i] = s
                rho["i_new"][new_i] = s

                weight += (
                    W_matrix[(s, new_i)]
                    - W_matrix[(s, old_i)]
                )

            rho["weight"] = weight

            rotations.append(rho)

            for s in cycle:
                new_i = rho["s_new"][s]
                current_student[s] = new_i
                current_copy[new_i] = s

            found = True
            break

        if not found:
            break

    ####################################################################
    # Step 3. Rotation digraph
    ####################################################################

    edges = set()

    for rho in rotations:

        rho["s_new_items"] = list(rho["s_new"].items())
        rho["i_new_items"] = list(rho["i_new"].items())

        rho["rank_interval"] = {}

        for s, new_i in rho["s_new_items"]:

            rho["rank_interval"][s] = (
                s_rank_dict[s].get(rho["s_old"][s], float("inf")),
                s_rank_dict[s].get(new_i, float("inf")),
            )

    for a, rho in enumerate(rotations):

        for b, rho2 in enumerate(rotations):

            if a == b:
                continue

            for s, new_i in rho["s_new_items"]:

                if rho2["s_old"].get(s) == new_i:
                    edges.add((a, b))

            for s, _ in rho["s_new_items"]:

                low, high = rho["rank_interval"][s]

                for i, new_student in rho2["i_new_items"]:

                    rank = s_rank_dict[s].get(i, float("inf"))

                    if not (low < rank <= high):
                        continue

                    old_student = rho2["i_old"][i]

                    if (
                        i_rank_dict[i].get(old_student, float("inf"))
                        > i_rank_dict[i].get(s, float("inf"))
                        >= i_rank_dict[i].get(new_student, float("inf"))
                    ):
                        edges.add((b, a))

    ####################################################################
    # Step 4. Minimum cut
    ####################################################################

    G = nx.DiGraph()

    G.add_node("source")
    G.add_node("terminal")

    for idx, rho in enumerate(rotations):

        if rho["weight"] >= 0:
            G.add_edge(
                idx,
                "terminal",
                capacity=rho["weight"],
            )
        else:
            G.add_edge(
                "source",
                idx,
                capacity=-rho["weight"],
            )

    INF = 10 ** 9

    for u, v in edges:
        G.add_edge(v, u, capacity=INF)

    _, partition = nx.minimum_cut(
        G,
        "source",
        "terminal",
    )

    reachable, _ = partition

    selected = {
        v
        for v in reachable
        if v != "source"
    }

    ####################################################################
    # Step 5. Recover matching
    ####################################################################

    result = M_student.copy()

    for s in students:

        if result[s] is None:
            continue

        best_rank = s_rank_dict[s][result[s]]

        for idx in selected:

            if s not in rotations[idx]["s_new"]:
                continue

            new_i = rotations[idx]["s_new"][s]
            rank = s_rank_dict[s][new_i]

            if rank > best_rank:
                best_rank = rank
                result[s] = new_i

    return result

import copy

def alg2_very_fast(
    instance,
    g_i_func=None,
    objective_type=None,
):
    """
    Compute a minimum-weight stable matching using the fast rotation algorithm.
    """

    institutions = instance["institutions"]
    students = instance["students"]
    capacities = instance["capacities"]
    student_prefs = instance["student_prefs"]
    inst_prefs = instance["inst_prefs"]

    # ============================================================
    # 🔥 COMPATIBILITY FIX: allow old calls (test.py safe)
    # ============================================================
    if g_i_func is None:
        g_i_func = instance.get("g_i")
    if objective_type is None:
        objective_type = instance.get("objective", "diversity")

    ####################################################################
    # Step 0. Stable sets
    ####################################################################
    stable_sets_dict = {
        i: compute_stable_sets(i, instance)
        for i in institutions
    }

    ####################################################################
    # Step 1. Compute edge weights
    ####################################################################
    weights = {}

    for i in institutions:

        E_i = stable_sets_dict[i][::-1]

        if not E_i or not E_i[0]:
            for s in students:
                weights[(s, i)] = 0
            continue

        inst_rank = {
            s: r
            for r, s in enumerate(inst_prefs[i])
        }

        for S in E_i:

            if not S:
                continue

            worst = max(S, key=lambda s: inst_rank[s])

            g = g_i_func(
                i=i,
                matching={i: S},
                objective_type=objective_type,
                instance=instance,
            )

            previous = sum(
                weights.get((x, i), 0)
                for x in S
                if x != worst
            )

            weights[(worst, i)] = g - previous

        for s in inst_prefs[i]:
            weights.setdefault((s, i), 0)

    ####################################################################
    # Step 2 ~ Step 6 (unchanged core logic)
    ####################################################################
    inst_copies = {}
    copy_to_original = {}

    idx = 0
    for i in institutions:
        inst_copies[i] = []
        for _ in range(capacities[i]):
            inst_copies[i].append(idx)
            copy_to_original[idx] = i
            idx += 1

    all_copies = list(copy_to_original.keys())

    student_prefs_copies = {s: [] for s in students}
    s_rank_dict = {s: {} for s in students}

    for s in students:
        rank = 0
        for i in student_prefs[s]:
            for copy in inst_copies[i]:
                student_prefs_copies[s].append(copy)
                s_rank_dict[s][copy] = rank
                rank += 1

    inst_prefs_copies = {}
    i_rank_dict = {}

    for copy in all_copies:
        i = copy_to_original[copy]
        prefs = [
            s for s in inst_prefs[i]
            if i in student_prefs[s]
        ]
        inst_prefs_copies[copy] = prefs
        i_rank_dict[copy] = {s: r for r, s in enumerate(prefs)}

    SCALE = 10*len(students)*len(institutions)
    W_matrix = {}

    for s in students:
        for copy, rank in s_rank_dict[s].items():
            i = copy_to_original[copy]
            w = weights.get((s, i), 0)
            W_matrix[(s, copy)] = int(SCALE * w + rank)

    matching_copy = min_weight_stable_fast(
        students=students,
        all_copies=all_copies,
        student_prefs_copies=student_prefs_copies,
        s_rank_dict=s_rank_dict,
        inst_prefs_copies=inst_prefs_copies,
        i_rank_dict=i_rank_dict,
        W_matrix=W_matrix,
    )

    matching = {i: [] for i in institutions}

    for s, copy in matching_copy.items():
        if copy is None:
            continue
        matching[copy_to_original[copy]].append(s)

    objective = sum(
        g_i_func(
            i=i,
            matching=matching,
            objective_type=objective_type,
            instance=instance,
        )
        for i in institutions
    )

    return objective, matching


# ============================================================
# ALG3 (same compatibility fix)
# ============================================================

def alg3_very_fast(
    instance,
    g_i_func=None,
    objective_type=None,
):
    """
    Repeatedly relax diversity bounds until feasible stable matching is found.
    """

    current_instance = copy.deepcopy(instance)
    relaxation_steps = 0

    if g_i_func is None:
        g_i_func = instance.get("g_i")
    if objective_type is None:
        objective_type = instance.get("objective", "diversity")

    while True:

        obj_val, matching = alg2_very_fast(
            current_instance,
            g_i_func,
            objective_type,
        )

        # families: no relaxation needed
        if objective_type == "families":
            return relaxation_steps, matching

        # diversity feasible
        if obj_val <= 0:
            return relaxation_steps, matching

        relaxation_steps += 1

        for i in current_instance["institutions"]:
            for c in current_instance["all_categories"]:
                current_instance["bounds"][i]["lower"][c] = max(
                    0,
                    current_instance["bounds"][i]["lower"][c] - 1,
                )
                current_instance["bounds"][i]["upper"][c] += 1
