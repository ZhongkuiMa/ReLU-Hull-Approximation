import os
import sys
from random import sample

from relu_hull.krelu_methods import krelu_with_cdd, fkrelu, krelu_with_sciplus

sys.path.insert(0, '../')
sys.path.insert(0, '../../')

import time
from datetime import datetime
from typing import Callable
import numpy as np
from relu_hull.krelu_methods_time import krelu_with_sci, krelu_with_triangle
from relu_hull.constraints import generate_box_constraints, generate_random_constraints, get_bounds_of_variables, \
    get_octahedral_approximation
from volume import estimate_polytope_volume, generate_random_points_in_box, estimate_polytope_volume2

REPEAT_SAMPLING_NUM = 10000

def _run_experiment(dimension: int, lower_bound: int, upper_bound: int,
                    total_added_constraints_num: int, added_constraints_num: int,
                    samples_num: int, methods: [Callable], points_num: int, sample_method: str,
                    step_length: float = 0.01):
    # lower_bounds = [lower_bound] * dimension + [0] * dimension
    # upper_bounds = [upper_bound] * (dimension * 2)
    print("Generate box constraints...")
    box_constraints = generate_box_constraints(dimension, lower_bound, upper_bound)
    print("Generate random constriants...")
    added_constraints = generate_random_constraints(dimension, -1, 1, total_added_constraints_num)

    method_names = [method.__name__ for method in methods]
    created_time = str(datetime.now().strftime("%d_%m_%Y_%H-%M-%S"))

    for i in range(samples_num):
        print(f"Sample-{i}-{dimension}d".center(100, "="))
        print("Choose random constraints...", end="")
        random_ids = sample(range(total_added_constraints_num), added_constraints_num)
        # random_ids = list(range(added_constraints_num))
        input_constraints = np.vstack([box_constraints, added_constraints[random_ids, :]])

        print(f"Input Constr. {input_constraints.shape[0]}")
        print("Calculate the convex hull/approximation...")
        print("".center(100, "-"))
        lb, ub = get_bounds_of_variables(input_constraints)

        print("Input Polytop Bounds:")
        print(f"Lower bounds: {lb}")
        print(f"Upper bounds: {ub}")
        print(f"Sample method: {sample_method} Step Length: {step_length}")
        random_points_list = []
        if sample_method == "random" and points_num > 0:
            print("Generate random sample points...", end="", flush=True)
            start = time.time()
            for _ in range(REPEAT_SAMPLING_NUM):
                # random_points = generate_random_points_in_box(points_num, lower_bounds, upper_bounds)
                random_points = generate_random_points_in_box(points_num, lb * 2, ub * 2)
                random_points_list.append(random_points)
            print(f"{time.time() - start:.4f}s")

        volumes, used_times, constrs_nums = [], [], []
        for method in methods:
            print(f"{method.__name__}".ljust(22), end="", flush=True)
            if method.__name__ == "fkrelu":
                c_temp = get_octahedral_approximation(input_constraints)
                start = time.time()
                constraints = method(c_temp, lb, ub)
            elif method.__name__ in ["krelu_with_cdd", "krelu_with_triangle"]:
                start = time.time()
                constraints = method(input_constraints, lb, ub)
            else:
                start = time.time()
                constraints = method(input_constraints, lb, ub, add_triangle=True, check=False)

            used_time = time.time() - start
            print(f"Constr.: {len(constraints)} ".ljust(18), end="")
            constrs_nums.append(constraints.shape[0])
            print(f"Time: {used_time:.4f}s".ljust(12), end="", flush=True)
            used_times.append(used_time)

            volume = 0
            start = time.time()
            if len(random_points_list):
                for random_points in random_points_list:
                    volume += estimate_polytope_volume(points_num, constraints, random_points=random_points)
            elif points_num > 0:
                volume, points_num = estimate_polytope_volume2(constraints, lb * 2, ub * 2, step_length)
                print(f"  points_num={points_num}  ", end="")

            total_points_num = 0
            if points_num > 0:
                total_points_num = points_num * REPEAT_SAMPLING_NUM if sample_method == "random" else points_num

                volume /= total_points_num
                print(f"Volume: {volume}".ljust(20), end="")
                volumes.append(volume)

            print(f"({time.time() - start:.4f})s".ljust(15))

        outline = f"\nSAMPLE: {i}\n" + \
                  f"METHODS: {method_names}\n" + \
                  f"TOTAL ADDED CONSTRAINTS NUMBER:{total_added_constraints_num}\n" + \
                  f"ADDED CONSTRAINTS NUMBER: {added_constraints_num}\n" + \
                  f"DIMENSION:       {dimension}\n" \
                  f"LOWER BOUND:     {lower_bound}\n" \
                  f"UPPER BOUND:     {upper_bound}\n" + \
                  f"SAMPLES NUMBER:  {samples_num}\n" \
                  f"POINTS NUMBER:   {total_points_num}\n" + \
                  f"SAMPLING METHOD: {sample_method}\n"

        file_path = f"./results/{dimension}d_volumes_{added_constraints_num}_{created_time}.txt"
        mode = "a" if os.path.exists(file_path) else "w"
        with open(file_path, mode=mode) as f:
            if mode == "w":
                f.write(outline)
            f.write("[")
            for v in volumes:
                f.write(f"{v},")
            f.write("],\n")

        file_path = f"./results/{dimension}d_used_times_{added_constraints_num}_{created_time}.txt"
        mode = "a" if os.path.exists(file_path) else "w"
        with open(file_path, mode=mode) as f:
            if mode == "w":
                f.write(outline)
            f.write("[")
            for t in used_times:
                f.write(f"{t},")
            f.write("],\n")

        file_path = f"./results/{dimension}d_constrs_num_{added_constraints_num}_{created_time}.txt"
        mode = "a" if os.path.exists(file_path) else "w"
        with open(file_path, mode=mode) as f:
            if mode == "w":
                f.write(outline)
            f.write("[")
            for c in constrs_nums:
                f.write(f"{c},")
            f.write("],\n")

        file_path = f"./results/{dimension}d_input_constraints_{added_constraints_num}_{created_time}.txt"
        with open(file_path, mode=mode) as f:
            if mode == "w":
                f.write(outline)
            for input_constraints in input_constraints:
                f.write(str(input_constraints) + "\n")


def run_experiment(methods: [Callable], dimension: int, points_num: int, groups_num: int,
                   lower_bound: int, upper_bound: int, added_constraints_num: int, sample_method: str,
                   step_length: float = 0.01):
    np.set_printoptions(precision=4, suppress=True, threshold=np.inf, linewidth=250)
    total_added_constraints_num = 1100000
    _run_experiment(dimension, lower_bound, upper_bound,
                    total_added_constraints_num, added_constraints_num,
                    groups_num, methods, points_num, sample_method, step_length=step_length)


import itertools

if __name__ == '__main__':
    group_num = 5
    dimensions = [10]
    lower_bound, upper_bound = -10, 10
    random_points_num = {2: 100, 3: 10000, 4: 10000, 5: 10000, 6: 100000}
    # random_points_num = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    methods = {2: [krelu_with_triangle, krelu_with_cdd, fkrelu, krelu_with_sci, krelu_with_sciplus],
               3: [krelu_with_triangle, krelu_with_cdd, fkrelu, krelu_with_sci, krelu_with_sciplus],
               4: [krelu_with_triangle, fkrelu, krelu_with_sci, krelu_with_sciplus],
               5: [krelu_with_triangle, krelu_with_sci, krelu_with_sciplus],
               6: [krelu_with_triangle, krelu_with_sci, krelu_with_sciplus]}
    # methods = {1: [krelu_with_mlf],
    #            2: [krelu_with_mlf],
    #            3: [krelu_with_mlf],
    #            4: [krelu_with_mlf],
    #            5: [krelu_with_mlf],
    #            6: [krelu_with_mlf],
    #            7: [krelu_with_mlf],
    #            8: [krelu_with_mlf],
    #            9: [krelu_with_mlf],
    #            10: [krelu_with_mlf]}

    sample_method = "random"
    for dimension, _ in itertools.product(dimensions, range(group_num)):
        run_experiment(methods[dimension], dimension, random_points_num[dimension], 1, lower_bound, upper_bound,
                       2 ** dimension, sample_method)
    # for dimension, _ in itertools.product(dimensions, range(group_num)):
    #     run_experiment(methods[dimension], dimension, random_points_num[dimension], 1, lower_bound, upper_bound,
    #                    3 ** dimension, sample_method)
    # for dimension,_ in itertools.product(dimensions, range(group_num)):
    #     run_experiment(methods[dimension], dimension, random_points_num[dimension], 1, lower_bound, upper_bound,
    #                    4 ** dimension, sample_method)
    # for dimension in dimensions:
    #     run_experiment(methods[dimension], dimension, random_points_num[dimension], group_num, lower_bound, upper_bound,
    #                    5 ** dimension, sample_method)

    # step_length = 0.2
    # sample_method = "uniform"
    # for dimension in dimensions:
    #     run_experiment(methods[dimension], dimension, random_points_num[dimension], group_num, lower_bound, upper_bound,
    #                    5 ** dimension, sample_method, step_length=step_length)
    # for dimension in dimensions:
    #     run_experiment(methods[dimension], dimension, random_points_num[dimension], group_num, lower_bound, upper_bound,
    #                    4 ** dimension, sample_method, step_length=step_length)
    # for dimension in dimensions:
    #     run_experiment(methods[dimension], dimension, random_points_num[dimension], group_num, lower_bound, upper_bound,
    #                    3 ** dimension, sample_method, step_length=step_length)
    # for dimension in dimensions:
    #     run_experiment(methods[dimension], dimension, random_points_num[dimension], group_num, lower_bound, upper_bound,
    #                    2 ** dimension, sample_method, step_length=step_length)
