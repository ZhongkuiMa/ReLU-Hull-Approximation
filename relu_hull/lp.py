from typing import Optional, Tuple, List

import gurobipy as grb
import numpy as np
from gurobipy import GRB


def solve_lp(constraints: np.ndarray, obj_func: np.ndarray, obj_type: GRB) -> Optional[Tuple[List, float]]:
    model = grb.Model("Solve LP by GUROBI")
    model.setParam("OutputFlag", False)
    model.setParam("LogToConsole", 0)
    model.setParam("Method", 0)  # Simplex method
    vars_num = constraints.shape[1] - 1
    # The default ub is GRB.INFINITY and the default lb is 0, here change the lb.
    x = np.asarray([1] + [model.addVar(lb=-GRB.INFINITY) for _ in range(vars_num)]).reshape((vars_num + 1, 1))

    for constraint in constraints:
        model.addConstr(grb.LinExpr(np.dot(constraint, x)[0]) >= 0)

    model.setObjective(grb.LinExpr(np.dot(obj_func, x)[0]), obj_type)
    model.optimize()
    return (model.x, model.objVal) if model.status == GRB.OPTIMAL else None
