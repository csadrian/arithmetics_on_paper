import numpy as np

from mathematics_dataset import generate
from mathematics_dataset.modules import modules

from mathematics_dataset.util.display import Decimal as _Decimal
from sympy.core.numbers import Integer, Rational

import gen_utils
from plot import *

def _sampling_func_from_module(mname, entropy_args):
    pname = mname.split('__')
    func = modules.train(
        generate._make_entropy_fn(*entropy_args))[pname[0]][pname[1]]
    return func

def sample_from_module(mname, entropy_args=(0,3)):
    func = _sampling_func_from_module(mname, entropy_args)
    res = func()[0]
    return res

def plot_example_md(mname, Solver=False, grid_size=100,
                    ncols=None, entropy_args=(0,1),
                    skip_ntypes=('rational', 'decimal'),
                    solver_kwargs=dict()):
    ntype_map = {
        'integer': Integer,
        'decimal': _Decimal,
        'rational': Rational
    }
    skip = [ntype_map[ntype] for ntype in skip_ntypes]
    if Solver is False:
        problem_name = gen_utils.module_to_problem(mname).__name__
        Solver = gen_utils.problem_to_solver(problem_name)
    # avoid infinite loop
    for i in range(1000):
        problem = sample_from_module(mname)
        wrong_values = False
        for value in problem.params.values():
            if type(value) in skip:
                wrong_values = True
                break
        if not wrong_values:
            break
    solver = Solver(grid_size, **solver_kwargs)
    solver.play(problem)
    steps = solver.get_steps()
    plot_steps(steps, ncols)
