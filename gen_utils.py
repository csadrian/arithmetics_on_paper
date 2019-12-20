import solvers
import problems

module_to_problem_dict = {'add_or_sub': problems.arithmetic_problems.AddOrSubProblem}
problem_to_solver_dict = {'AddOrSubProblem': solvers.arithmetic_solvers.AddOrSubSolver}


def module_to_problem(module_name):
    return module_to_problem_dict[module_name]


def problem_to_solver(problem_name):
    return problem_to_solver_dict[problem_name]
