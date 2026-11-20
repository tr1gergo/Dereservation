import random
import math


def _normalize_fam_type(fam_type):
    if fam_type not in {"independent", "correlated"}:
        raise ValueError(
            "fam_type must be 'independent' or 'correlated'"
        )

    return fam_type


def _validate_generation_parameters(
    num_students,
    num_institutions,
    alpha,
    beta,
):
    if num_students <= 0:
        raise ValueError("num_students must be positive")

    if num_institutions <= 0:
        raise ValueError("num_institutions must be positive")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")


def _distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def _balanced_capacities(num_students, institutions):
    if not institutions:
        raise ValueError("num_institutions must be positive")

    base_capacity = num_students // len(institutions)
    extra_seats = num_students % len(institutions)

    capacities = {
        i: base_capacity
        for i in institutions
    }

    shuffled_institutions = list(institutions)
    random.shuffle(shuffled_institutions)

    for i in shuffled_institutions[:extra_seats]:
        capacities[i] += 1

    return capacities


def _rank_institutions_by_section3_utility(
    student,
    institutions,
    student_coordinates,
    inst_coordinates,
    common_school_shocks,
    alpha,
    beta,
):
    utilities = []

    for i in institutions:
        distance = _distance(
            student_coordinates[student],
            inst_coordinates[i],
        )

        utility = (
            -beta * distance
            + (1.0 - beta)
            * (
                alpha * common_school_shocks[i]
                + (1.0 - alpha) * random.normalvariate(0.0, 1.0)
            )
        )

        utilities.append(
            (utility, random.random(), i)
        )

    utilities.sort(reverse=True)

    return [
        i
        for _, _, i in utilities
    ]


def generate_families_and_demographics(students, coordinates_dict):
    """
    Generate

    - families
    - demographic categories
    - sibling coordinates

    Siblings share the same demographic profile and home location.
    """

    families = []
    unassigned = list(students)
    random.shuffle(unassigned)

    categories = {}

    while unassigned:

        ############################################################
        # Family size
        ############################################################

        r = random.random()

        if r < 0.4:
            size = 1
        elif r < 0.9:
            size = 2
        else:
            size = 3

        size = min(size, len(unassigned))

        family = tuple(unassigned[:size])
        unassigned = unassigned[size:]

        families.append(family)

        ############################################################
        # Shared home location
        ############################################################

        base_student = family[0]
        base_coords = coordinates_dict[base_student]

        ############################################################
        # Shared demographic profile
        ############################################################

        cats = set()

        # SES (mutually exclusive)

        r = random.random()

        if r < 0.50:
            cats.add(0)          # Low SES
            p_ell = 0.25

        elif r < 0.80:
            cats.add(1)          # Mid SES
            p_ell = 0.05

        else:
            cats.add(2)          # High SES
            p_ell = 0.01

        # ELL

        if random.random() < p_ell:
            cats.add(3)

        # SPED

        if random.random() < 0.12:
            cats.add(4)

        ############################################################
        # Apply to all siblings
        ############################################################

        for student in family:
            coordinates_dict[student] = base_coords
            categories[student] = set(cats)

    return families, categories, coordinates_dict


def generate_instance(
    num_students,
    num_institutions,
    fam_type="independent",
    alpha=0.1,
    beta=0.1,
):
    """
    Generate one random many-to-one matching instance.

    Student preferences follow the Section 3 simulation utility:
    U_ix = -beta * distance(l_i, l_x)
           + (1 - beta) * (alpha * Z_0x + (1 - alpha) * Z_ix)

    fam_type:
        "independent" -> siblings share location and demographics, but draw
        separate preference shocks.
        "correlated" -> siblings share location, demographics, and the same
        preference list.
    """

    fam_type = _normalize_fam_type(fam_type)
    _validate_generation_parameters(
        num_students,
        num_institutions,
        alpha,
        beta,
    )

    ############################################################
    # Basic sets
    ############################################################

    students = list(range(num_students))
    institutions = list(range(num_institutions))

    all_categories = [0, 1, 2, 3, 4, 5]

    ############################################################
    # Coordinates
    ############################################################

    student_coordinates = {
        s: (random.random(), random.random())
        for s in students
    }

    inst_coordinates = {
        i: (random.random(), random.random())
        for i in institutions
    }

    ############################################################
    # Families and demographics
    ############################################################

    (
        families,
        categories,
        student_coordinates
    ) = generate_families_and_demographics(
        students,
        student_coordinates
    )

    ############################################################
    # Capacities
    ############################################################

    capacities = _balanced_capacities(
        num_students,
        institutions,
    )

    ############################################################
    # Diversity bounds
    ############################################################

    category_count = {
        c: sum(
            1
            for s in students
            if c in categories[s]
        )
        for c in all_categories
    }

    bounds = {
        i: {
            "lower": {},
            "upper": {}
        }
        for i in institutions
    }

    for i in institutions:

        for c in all_categories:

            expected = (
                category_count[c]
                * capacities[i]
                / num_students
            )

            lower = max(
                0,
                math.floor(0.9*expected)
            )

            upper = min(
                capacities[i],
                math.ceil(1.1 * expected)
            )

            if lower > upper:
                lower = upper

            bounds[i]["lower"][c] = lower
            bounds[i]["upper"][c] = upper

    ############################################################
    # Student preferences
    ############################################################

    student_prefs = {}
    common_school_shocks = {
        i: random.normalvariate(0.0, 1.0)
        for i in institutions
    }

    for family in families:

        if fam_type == "correlated":
            pref_list = _rank_institutions_by_section3_utility(
                student=family[0],
                institutions=institutions,
                student_coordinates=student_coordinates,
                inst_coordinates=inst_coordinates,
                common_school_shocks=common_school_shocks,
                alpha=alpha,
                beta=beta,
            )

            for student in family:
                student_prefs[student] = list(pref_list)

        else:
            for student in family:
                student_prefs[student] = (
                    _rank_institutions_by_section3_utility(
                        student=student,
                        institutions=institutions,
                        student_coordinates=student_coordinates,
                        inst_coordinates=inst_coordinates,
                        common_school_shocks=common_school_shocks,
                        alpha=alpha,
                        beta=beta,
                    )
                )

    ############################################################
    # Walk zones
    ############################################################

    walk_zones = {
        s: min(
            institutions,
            key=lambda i: _distance(
                student_coordinates[s],
                inst_coordinates[i],
            ),
        )
        for s in students
    }

    ############################################################
    # Institution preferences
    ############################################################

    inst_prefs = {}
    inst_tie_break_ranks = {}

    for i in institutions:
        tie_breaking = list(students)
        random.shuffle(tie_breaking)

        tie_break_rank = {
            s: rank
            for rank, s in enumerate(tie_breaking)
        }

        inst_tie_break_ranks[i] = tie_break_rank

        inst_prefs[i] = sorted(
            students,
            key=lambda s: (
                0 if walk_zones[s] == i else 1,
                tie_break_rank[s],
            ),
        )

    ############################################################
    # Instance
    ############################################################

    instance = {

        "metadata": {
            "num_students": num_students,
            "num_institutions": num_institutions,
            "fam_type": fam_type,
            "alpha": alpha,
            "beta": beta,
        },

        "students": students,

        "institutions": institutions,

        "capacities": capacities,

        "student_prefs": student_prefs,

        "inst_prefs": inst_prefs,

        "categories": categories,

        "families": families,

        "student_coordinates": student_coordinates,

        "inst_coordinates": inst_coordinates,

        "walk_zones": walk_zones,

        "inst_tie_break_ranks": inst_tie_break_ranks,

        "all_categories": all_categories,

        "bounds": bounds,

        ########################################################
        # Default optimization objective
        ########################################################

        "objective": "diversity",
    }

    return instance

def compute_g_i(
    i,
    matching,
    objective_type,
    instance
):
    """
    Compute g_i(M) for institution i.

    Parameters
    ----------
    i : int
        Institution.

    matching : dict
        institution -> list of students

    objective_type : str
        "diversity"
        "families"

    instance : dict
    """

    assigned_students = matching.get(i, [])

    ############################################################
    # Diversity objective
    ############################################################

    if objective_type == "diversity":

        bounds = instance["bounds"][i]
        categories = instance["categories"]
        all_categories = instance["all_categories"]

        score = 0

        for c in all_categories:

            num = sum(
                1
                for s in assigned_students
                if c in categories[s]
            )

            lower = bounds["lower"].get(c, 0)
            upper = bounds["upper"].get(c, float("inf"))

            score += max(0, lower - num)
            score += max(0, num - upper)

        return score

    ############################################################
    # Family objective
    ############################################################

    elif objective_type == "families":

        score = 0

        assigned = set(assigned_students)

        for family in instance["families"]:

            if set(family).issubset(assigned):
                score -= 1

        return score

    ############################################################

    else:

        raise ValueError(
            f"Unknown objective_type: {objective_type}"
        )

def calc_max_violation(
    matching,
    instance
):
    """
    Compute the maximum (L_inf) diversity violation.

    Parameters
    ----------
    matching : dict
        institution -> list of students

    instance : dict

    Returns
    -------
    int
    """

    max_violation = 0

    for i in instance["institutions"]:

        assigned_students = matching.get(i, [])

        bounds = instance["bounds"][i]

        for c in instance["all_categories"]:

            num = sum(
                1
                for s in assigned_students
                if c in instance["categories"][s]
            )

            lower = bounds["lower"].get(c, 0)
            upper = bounds["upper"].get(c, float("inf"))

            violation = (
                max(max(0, lower - num), max(0, num - upper))
            )

            max_violation = max(
                max_violation,
                violation
            )

    return max_violation
