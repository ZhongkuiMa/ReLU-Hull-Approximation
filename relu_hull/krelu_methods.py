import contextlib
import itertools
from typing import List

import cdd
import numpy as np

with contextlib.suppress(Exception):
    from ELINA.python_interface import fconv

# Sometimes in some machines, there will obtain uncorrected answer by cdd with a few constraints due to numerical bugs.
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

    constraints = constraints.copy()
    vars_num = constraints.shape[1] - 1

    triangle_constraints = None
    if add_triangle or vars_num == 1:
        triangle_constraints = krelu_with_triangle(constraints, lower_bounds, upper_bounds)
    if vars_num == 1:
        return triangle_constraints

    c = np.ascontiguousarray(constraints)
    x = np.ascontiguousarray(np.transpose(cal_vertices(constraints, check=check)))
    d = np.matmul(c, x)
    c, x, d = _cal_betas(c, x, d)

    return np.vstack((c, triangle_constraints)) if add_triangle else c


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


def krelu_with_triangle(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    """
    This is a approximate method to get the convec hull of a group of ReLU functions.

    This function calculate all triangle relaxation constraints of variables.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "constraints, lower_bounds and the upper_bounds should have the same number of variables."
    constraints = constraints.copy()
    vars_num = constraints.shape[1] - 1

    new_constraints = np.empty((0, 2 * vars_num + 1))
    # Add lower constraints.
    # y_i >= 0
    y = np.hstack((np.zeros((vars_num, vars_num + 1)), np.identity(vars_num)))
    new_constraints = np.vstack((new_constraints, y))
    # y_i >= x_i
    yx = np.concatenate((np.zeros((vars_num, 1)), -np.identity(vars_num), np.identity(vars_num)), axis=1)
    new_constraints = np.vstack((new_constraints, yx))

    # Add upper constraints.
    lbs, ubs = np.array([lower_bounds]), np.array([upper_bounds])
    k = ubs / (ubs - lbs)
    b = (- lbs * k).T
    kx = np.diag(k[0])
    y = np.identity(vars_num)
    new_constraints = np.vstack((new_constraints, np.concatenate((b, kx, -y), axis=1)))

    return new_constraints


def krelu_with_pycdd(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    """
    This is an exact method to get the convex hull of a group of ReLU functions.

    This method is written in Python.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "constraints, lower_bounds and the upper_bounds should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."

    try:
        h_repr = cdd.Matrix(constraints, number_type="float")
        h_repr.rep_type = cdd.RepType.INEQUALITY
        vertices = _get_vertices_of_each_orthant(h_repr)
        v_repr = [[1] + r + [max(x, 0) for x in r] for r in vertices]
        v_repr = cdd.Matrix(v_repr, number_type="float")
        v_repr.rep_type = cdd.RepType.GENERATOR
        new_poly = cdd.Polyhedron(v_repr)
        h_repr = [[float(v) for v in c] for c in new_poly.get_inequalities()]
    except Exception:
        h_repr = cdd.Matrix(constraints, number_type="fraction")
        h_repr.rep_type = cdd.RepType.INEQUALITY
        vertices = _get_vertices_of_each_orthant(h_repr)
        v_repr = [([1] + r + [x if x > 0 else 0 for x in r]) for r in vertices]
        v_repr = cdd.Matrix(v_repr, number_type="fraction")
        v_repr.rep_type = cdd.RepType.GENERATOR
        new_poly = cdd.Polyhedron(v_repr)
        h_repr = [[float(v) for v in c] for c in new_poly.get_inequalities()]
    h_repr = np.asarray(h_repr)
    # print(h_repr)
    return h_repr


def _get_vertices_of_each_orthant(h_repr) -> List[List]:
    """
    Get vertices in each orthant, which is all the vertices including the vertices of the polyhedron and the
    intersection points with each axis.

    Reference: krelu.py of ERAN
    """
    var_num = len(h_repr[0]) - 1
    vertices = []
    for polarity in itertools.product([-1, 1], repeat=var_num):
        new_mat = h_repr.copy()
        for i in range(var_num):
            constraint = [polarity[i] if j == i + 1 else 0 for j in range(var_num + 1)]
            new_mat.extend([constraint])
        new_mat.canonicalize()  # Remove redundant h constraints.
        v_repr = cdd.Polyhedron(new_mat).get_generators()
        vertices += [list(v[1:]) for v in v_repr]

    return vertices


def krelu_with_cdd(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    """
        This is an exact method to get the convex hull of a group of ReLU functions.

        This method is written in C from ELINA.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "constraints, lower_bounds and the upper_bounds should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."
    return fconv.krelu_with_cdd(constraints)


def fkrelu(constraints: np.ndarray, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    """
        This is an approximate method to get the convex hull of a group of ReLU functions.

        This method is written in C from ELINA.
    """
    assert constraints.shape[1] - 1 == len(lower_bounds) == len(upper_bounds), \
        "constraints, lower_bounds and the upper_bounds should have the same number of variables."
    assert all(lb < 0 for lb in lower_bounds), "All lower bound should be negative."
    assert all(ub > 0 for ub in upper_bounds), "All upper bound should be positive."
    return fconv.fkrelu(constraints)
