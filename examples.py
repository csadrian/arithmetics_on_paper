from utils import *
from solvers import *
from problems import *
from solvers_milan import *

GRID_SIZE = 10

def generator(Solver_, Problem_, N=10):
    problem_generator = Problem_().generator()
    solver = Solver_(GRID_SIZE)
    iterator = iter(solver.generator(problem_generator))
    solutions = []
    for i in range(N):
        solutions.append(next(iterator))
    return solutions

def prime_solver_example():
    prime_solutions = solutions(IsPrimeSolver, IsPrimeProblem)
    plot_steps(prime_solutions[-1])

def add_solver_example():
    add_solutions = solutions(Solver_=AddSolver_,
                              Problem_=HardAdditionProblem)
    plot_steps(add_solutions[-1])

def paper_example():
    paper = PaperWithNumbers(10)
    paper.make_step()
    x, y = paper.print_symbol(SIGN_ADD, 5, 5, attention=True)
    paper.make_step()
    paper.print_number(121, x-1, y)
    paper.make_step()
    plot_steps(paper.steps)
    # return paper

def plot_example(Solver_=AddSolver,
                 Problem_=HardAdditionProblem, steps=True):
    generator = Solver_(10).generator(
        Problem_().generator())
    example = next(iter(generator))
    plot_steps(example)
