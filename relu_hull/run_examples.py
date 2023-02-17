import contextlib

import numpy as np

from relu_hull.constraints import generate_one_random_octahedron, get_bounds_of_variables
from relu_hull.krelu_methods import krelu_with_sci, krelu_with_pycdd, krelu_with_sciplus, krelu_with_cdd, fkrelu

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, threshold=np.inf, linewidth=250)

    print("Run examples using different krelu methods.")

    print("Generate a 3-dimensional octahedron.")
    dim = 3
    constraints = generate_one_random_octahedron(dim)
    print("Constraints:")
    print(constraints)

    print("Calculate lower and upper bounds of variables by linear programming.")
    lower_bounds, upper_bounds = get_bounds_of_variables(constraints)
    print("Lower bounds:")
    print(lower_bounds)
    print("Upper bounds:")
    print(upper_bounds)

    print("Exact method krelu_with_pycdd:")
    c = krelu_with_pycdd(constraints, lower_bounds, upper_bounds)
    print(c)

    with contextlib.suppress(Exception):
        c = krelu_with_cdd(constraints, lower_bounds, upper_bounds)
        print("Exact method krelu_with_cdd:")
        print(c)

        c = fkrelu(constraints, lower_bounds, upper_bounds)
        print("Exact method krelu_with_fast:")
        print(c)

    print("Approximate Method krelu_with_sci:")
    c = krelu_with_sci(constraints, lower_bounds, upper_bounds, add_triangle=True)
    print(c)

    print("Approximate Method krelu_with_sciplus:")
    c = krelu_with_sciplus(constraints, lower_bounds, upper_bounds, add_triangle=True)
    print(c)
