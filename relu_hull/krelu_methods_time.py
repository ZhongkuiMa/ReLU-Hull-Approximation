"""
This is for recording the computation time of different parts of algorithm.
"""

import os
import time
from typing import List

import cdd
import numpy as np

from relu_hull.krelu_methods import krelu_with_triangle



# Sometimes in some machines, there will obtain uncorrected answer by cdd with a few constraints.
# This parameter is used to detect this situation.
RISK_THRESHOLD = {3: 28, 4: 250, 5: 3808}
# RISK_THRESHOLD = {3: 46, 4: 382, 5: 3808}

def krelu_with_sci(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float],
                   add_triangle: bool = False, check=True) -> np.ndarray:
    """
    This is an approximate method to calculate the convex hull of a group of ReLU functions.

    This method only use the default order of variables to extend output variables.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        f"constraints {constraints.shape[1] - 1}, lower_bounds {len(lower_bounds)} and the " \
        f"upper_bounds {len(upper_bounds)}should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."
    start = time.time()

    constraints = constraints.copy()
    vars_num = constraints.shape[1] - 1

    triangle_constraints = None
    if add_triangle or vars_num == 1:
        triangle_constraints = krelu_with_triangle(constraints, lower_bounds, upper_bounds)
    if vars_num == 1:
        return triangle_constraints
    start2 = time.time()
    v = cal_vertices(constraints, check=check)

    file_path = f"./{vars_num}d_vertex_times_{constraints.shape[0]}.txt"
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode=mode) as f:
        f.write(f"{time.time() - start2}\t")

    c = np.ascontiguousarray(constraints)
    x = np.ascontiguousarray(np.transpose(v))
    d = np.matmul(c, x)
    c, x, d = _cal_betas(c, x, d)
    c = np.vstack((c, triangle_constraints)) if add_triangle else c

    file_path = f"./{vars_num}d_total_times_{constraints.shape[0]}.txt"
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode=mode) as f:
        f.write(f"{time.time() - start}\t")

    return c


def krelu_with_sciplus(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float],
                       add_triangle: bool = False, check=True) -> np.ndarray:
    """
        This is an approximate method to calculate the convex hull of a group of ReLU functions.

        This method use both the default order and its reverse order of variables to extend output variables.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        f"constraints {constraints.shape[1] - 1}, lower_bounds {len(lower_bounds)} and the " \
        f"upper_bounds {len(upper_bounds)}should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."

    constraints = constraints.copy()
    vars_num = constraints.shape[1] - 1

    triangle_constraints = None
    if add_triangle or vars_num == 1:
        triangle_constraints = krelu_with_triangle(constraints, lower_bounds, upper_bounds)
    if vars_num == 1:
        return triangle_constraints

    constraints = np.ascontiguousarray(constraints)
    vertices = np.ascontiguousarray(np.transpose(cal_vertices(constraints, check=check)))
    m = np.matmul(constraints, vertices)

    reversed_order = list(range(vars_num, 0, -1))
    inversed_order = reversed_order + [o + vars_num for o in reversed_order]

    new_constraints = np.empty((0, 2 * vars_num + 1))

    for k in range(2):
        c = constraints[:, [0] + reversed_order].copy() if k == 1 else constraints.copy()
        x = vertices[[0] + reversed_order].copy() if k == 1 else vertices.copy()
        d = m.copy()
        c, x, d = _cal_betas(c, x, d)

        if k == 1:
            c[:, 1:] = c[:, inversed_order]
        new_constraints = np.vstack((new_constraints, c))

    if triangle_constraints is not None:
        return np.vstack((new_constraints, triangle_constraints))
    return new_constraints


def cal_vertices(constraints: np.ndarray, check=True) -> np.ndarray:
    """This method calculate the vertices of the polyhedron defined by the given constraints. """

    vars_num = constraints.shape[1] - 1
    try:
        mat = cdd.Matrix(constraints, number_type="float")
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        vertices = poly.get_generators()

        if check:
            vertices_num = len(vertices)
            if vars_num in RISK_THRESHOLD and vertices_num < RISK_THRESHOLD[vars_num]:
                mat = cdd.Matrix(constraints, number_type="fraction")
                mat.rep_type = cdd.RepType.INEQUALITY
                poly = cdd.Polyhedron(mat)
                vertices = poly.get_generators()
                if vertices_num == len(vertices):
                    print(vertices_num, "->", len(vertices), "This is no risk.")

    except Exception:
        mat = cdd.Matrix(constraints, number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        vertices = poly.get_generators()

    # mat = cdd.Matrix(constraints, number_type="fraction")
    # mat.rep_type = cdd.RepType.INEQUALITY
    # poly = cdd.Polyhedron(mat)
    # vertices = poly.get_generators()

    return np.asarray(vertices, dtype=np.float64)


def _cal_betas(c, x, d, tol=1e-8):
    vars_num = c.shape[1] - 1
    x_greater_zero, x_smaller_zero = (x > tol), (x < -tol)
    y = x.copy()
    y[y < 0] = 0

    for i in range(1, vars_num + 1):
        vertices_greater_zero, vertices_smaller_zero = x_greater_zero[i], x_smaller_zero[i]
        beta1 = beta2 = np.zeros((c.shape[0], 1))

        if np.any(vertices_greater_zero):
            beta1 = d[:, vertices_greater_zero] / x[i, vertices_greater_zero]
            beta1 = np.max(-beta1, axis=1).reshape((-1, 1))

        if np.any(vertices_smaller_zero):
            beta2 = d[:, vertices_smaller_zero] / x[i, vertices_smaller_zero]
            beta2 = np.max(beta2, axis=1).reshape((-1, 1))
        # print(f"beta1={beta1.copy().reshape((1,-1))}")
        # print(f"beta2={beta2.copy().reshape((1,-1))}")
        c = np.hstack((c, beta1 + beta2))
        c[:, [i]] -= beta2
        x = np.vstack((x, y[i]))
        d += np.outer(c[:, -1], y[i]) + np.outer(-beta2, x[i])
        # print(c)
    return c, x, d

