import numpy as np
from solvers import Solver, AddSolver, SubtractSolver
from common import Symbols as S

class AddSubMultipleSolver(Solver):

    def play(self, problem):
        coefs = problem['coefs']
        ops = problem['ops']

        start = (0, 5)
        x, y = start

        x, y = self.paper.print_number(coefs[0], x, y, attention=True, orientation=1, reset=True)

        x, y = x + 1, y - 1
        
        marginx = x
        marginy = y
        
        total = coefs[0]

        for i in range(len(coefs) - 1):
            
            x = marginx
            y = marginy
            x, y = self.paper.print_number(coefs[i+1], x, y, attention=True)
            
            self.print_symbol(ops[i], x, y, attention=True)

            if ops[i] is 11:
                self.make_step(solver=AddSolver)
                total = total + coefs[i+1]
            else:
                self.make_step(solver=SubtractSolver)
                total = total - coefs[i+1]

            marginx = marginx + 1
            x = marginx
            y = marginy
            
            
            total = total + coefs[i+1]
            x, y = self.paper.print_number(total, x, y, attention=True, reset=True, step_by_step=True)
            
            marginx = marginx + 1

