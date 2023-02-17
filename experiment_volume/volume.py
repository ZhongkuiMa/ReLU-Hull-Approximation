from itertools import product
from typing import List, Tuple

import numpy as np
from relu_hull.constraints import get_bounds_of_variables


class Cache:
    random_points = None


def estimate_polytope_volume(points_num: int, constraints: np.ndarray, random_points: np.ndarray = None,
                             lower_bounds: List[float] = None, upper_bounds: List[float] = None, tol=1e-8) -> int:
    points_in = 0

    if random_points is None:
        if lower_bounds is None and upper_bounds is None:
            lower_bounds, upper_bounds = get_bounds_of_variables(constraints)
        random_points = generate_random_points_in_box(points_num, lower_bounds, upper_bounds)

    points_in += is_in_polyhedron(random_points, constraints, tol=tol)

    return points_in


def estimate_polytope_volume2(constraints: np.ndarray,
                              lower_bounds: List[float] = None, upper_bounds: List[float] = None,
                              step_length=0.1, tol=1e-8) -> Tuple[int, int]:
    Cache.random_points = None
    points_in = 0
    points_num = 0
    if len(lower_bounds) == 2:
        l1 = list(np.arange(lower_bounds[0], upper_bounds[0] + step_length, step_length))
        l2 = list(np.arange(lower_bounds[1], upper_bounds[1] + step_length, step_length))
        for p in product(l1, l2):
            p = np.asarray([p])
            a, b = cal_is_in_polyhedron(p, constraints, tol)
            points_in += a
            points_num += b

    elif len(lower_bounds) == 4:
        l1 = list(np.arange(lower_bounds[0], upper_bounds[0] + step_length, step_length))
        l2 = list(np.arange(lower_bounds[1], upper_bounds[1] + step_length, step_length))
        l3 = list(np.arange(lower_bounds[2], upper_bounds[2] + step_length, step_length))
        l4 = list(np.arange(lower_bounds[3], upper_bounds[3] + step_length, step_length))
        for p in product(l1, l2, l3, l4):
            p = np.asarray([p])
            a, b = cal_is_in_polyhedron(p, constraints, tol)
            points_in += a
            points_num += b

    elif len(lower_bounds) == 6:
        l1 = list(np.arange(lower_bounds[0], upper_bounds[0] + step_length, step_length))
        l2 = list(np.arange(lower_bounds[1], upper_bounds[1] + step_length, step_length))
        l3 = list(np.arange(lower_bounds[2], upper_bounds[2] + step_length, step_length))
        l4 = list(np.arange(lower_bounds[3], upper_bounds[3] + step_length, step_length))
        l5 = list(np.arange(lower_bounds[4], upper_bounds[4] + step_length, step_length))
        l6 = list(np.arange(lower_bounds[5], upper_bounds[5] + step_length, step_length))
        for p in product(l1, l2, l3, l4, l5, l6):
            p = np.asarray([p])
            a, b = cal_is_in_polyhedron(p, constraints, tol)
            points_in += a
            points_num += b

    elif len(lower_bounds) == 8:
        l1 = list(np.arange(lower_bounds[0], upper_bounds[0] + step_length, step_length))
        l2 = list(np.arange(lower_bounds[1], upper_bounds[1] + step_length, step_length))
        l3 = list(np.arange(lower_bounds[2], upper_bounds[2] + step_length, step_length))
        l4 = list(np.arange(lower_bounds[3], upper_bounds[3] + step_length, step_length))
        l5 = list(np.arange(lower_bounds[4], upper_bounds[4] + step_length, step_length))
        l6 = list(np.arange(lower_bounds[5], upper_bounds[5] + step_length, step_length))
        l7 = list(np.arange(lower_bounds[6], upper_bounds[6] + step_length, step_length))
        l8 = list(np.arange(lower_bounds[7], upper_bounds[7] + step_length, step_length))
        for p in product(l1, l2, l3, l4, l5, l6, l7, l8):
            p = np.asarray([p])
            a, b = cal_is_in_polyhedron(p, constraints, tol)
            points_in += a
            points_num += b
    else:
        raise "The dimension is not supported"
    return points_in, points_num


def cal_is_in_polyhedron(p, constraints, tol):
    if Cache.random_points is None or Cache.random_points.shape[1] != p.shape[1]:
        Cache.random_points = p
    else:
        Cache.random_points = np.vstack([Cache.random_points, p])

    if Cache.random_points is not None and Cache.random_points.shape[0] > 100:
        points_num = Cache.random_points.shape[0]
        points_in = is_in_polyhedron(Cache.random_points, constraints, tol=tol)
        Cache.random_points = None
        return points_in, points_num

    return 0, 0


def is_in_polyhedron(points: np.ndarray, constraints: np.ndarray, tol=1e-8) -> int:
    points_in = 0
    p = points
    ax = np.dot(constraints[:, 1:], p.T) + np.tile(np.array([constraints[:, 0]]).T, (1, p.shape[0]))
    points_in += np.nonzero(np.all(ax > -tol, 0))[0].shape[0]

    return points_in


def generate_random_points_in_box(points_num: int, lower_bounds: List[float], upper_bounds: List[float]) -> np.ndarray:
    vars_num = len(lower_bounds)
    lbs, ubs = np.array([lower_bounds]), np.array([upper_bounds])
    r = np.random.random((points_num, vars_num))
    return np.tile(lbs, (points_num, 1)) + r * np.tile(ubs - lbs, (points_num, 1))
