from paper import PaperWithNumbers


class Solver:

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.set_paper()

    def __getattr__(self, attr):
        return getattr(self.paper, attr)

    def set_paper(self):
        self.paper = PaperWithNumbers(self.grid_size)

    def get_steps(self):
        return self.paper.get_steps()

    def generator(self, problem_generator):
        for problem in problem_generator:
            self.set_paper()
            self.play(problem)
            yield self.get_steps()

    def play(self, problem):
        raise NotImplementedError
