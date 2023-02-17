import itertools
from random import random
from typing import Tuple, List

import numpy as np
from gurobipy import GRB

from relu_hull.lp import solve_lp


def generate_one_random_octahedron(dim: int) -> np.ndarray:
    constraints = []
    for coeffs in itertools.product([-1, 0, 1], repeat=dim):
        if all(c == 0 for c in coeffs):
            continue
        constraint = [random() * 10] + [-c for c in coeffs]
        constraints.append(constraint)
    return np.asarray(constraints)


def generate_box_constraints(dimension: int, lower_bound: float, upper_bound: float) -> np.ndarray:
    lbs, ubs = [lower_bound] * dimension, [upper_bound] * dimension
    lb, ub = -np.array(lbs).reshape((-1, 1)), np.array(ubs).reshape((-1, 1))
    v1, v2 = np.identity(dimension), -np.identity(dimension)

    return np.vstack([np.hstack([lb, v1]), np.hstack([ub, v2])])


def generate_random_constraints(dimension: int, lower_bound: float, upper_bound: float, number: int) -> np.ndarray:
    constraints = []
    r = upper_bound - lower_bound
    for _ in range(number):
        constraint = [r * random() + lower_bound for __ in range(dimension + 1)]
        constraint[0] = abs(constraint[0])
        constraints.append(constraint)

    return np.asarray(constraints)


def get_bounds_of_variables(constraints: np.ndarray) -> Tuple[List, List]:
    upper_bounds, lower_bounds = [], []
    vars_num = constraints.shape[1] - 1
    for i in range(1, vars_num + 1):
        obj_func = np.zeros((1, vars_num + 1))
        obj_func[0, i] = 1
        _, upper_bound = solve_lp(constraints, obj_func[0], GRB.MAXIMIZE)
        _, lower_bound = solve_lp(constraints, obj_func[0], GRB.MINIMIZE)
        assert upper_bound is not None and lower_bound is not None, \
            "The polytope is unbounded or the LPP is infeasible."
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return lower_bounds, upper_bounds


def get_octahedral_approximation(constraints: np.ndarray) -> np.ndarray:
    oct_constraints = []
    dim = constraints.shape[1] - 1
    for coeffs in itertools.product([-1, 0, 1], repeat=dim):
        coeffs = list(coeffs)
        if all(c == 0 for c in coeffs):
            continue
        obj_func = [0] + coeffs
        _, upper_bound = solve_lp(constraints, np.asarray(obj_func), GRB.MAXIMIZE)
        oct_constraints.append([upper_bound] + [-c for c in coeffs])

    return np.asarray(oct_constraints)

