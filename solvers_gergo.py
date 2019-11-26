import numpy as np
from solvers import Solver, AddSolver, SubtractSolver
from common import Symbols as S

#Need to rewrite to solver handled coordinates
class AddSubMultipleSolver(Solver):

    def play(self, problem):
        coefs = problem['coefs']
        ops = problem['ops']

        self.paper._set_position(0, 5)
        self.paper.mark_current_pos('start')

        self.paper.print_number(coefs[0], orientation=1, reset=True)
    
        self.paper.move_down()
        self.paper.move_left()
        
        self.paper.mark_current_pos('margin')
        
        total = coefs[0]

        for i in range(len(coefs) - 1):
            
            self.paper.go_to_mark('margin')
            
            self.paper.print_number(coefs[i+1])
            
            self.print_symbol(ops[i])

            if ops[i] is 11:
                self.make_step(solver=AddSolver)
                total = total + coefs[i+1]
            else:
                self.make_step(solver=SubtractSolver)
                total = total - coefs[i+1]
            
            self.paper.go_to_mark('margin')
            self.paper.move_down()
            self.paper.mark_current_pos('margin')

            self.paper.print_number(total, reset=True, step_by_step=True)
            
            self.paper.go_to_mark('margin')
            self.move_down()
            self.mark_current_pos('margin')

            

class PlaceValueSolver(Solver):

    def play(self, problem):
        number = problem['number']
        place = problem['place']

        self.paper._set_position(0, 0)

        self.paper.print_number(number, orientation=1, reset=True)

        self.print_symbol(S.last_digit)

        self.paper.print_number(place, orientation=1)

        for i in range(1, place + 1):

            self.paper.make_step()

            self.paper._set_position(i, 0)

            for _ in range(i):
                self.paper.print_symbol(0)

            self.paper.print_number(int(str(number)[i:]), orientation=1)

            self.paper.print_symbol(S.last_digit)
            
            self.paper.print_number(i, orientation=1)

