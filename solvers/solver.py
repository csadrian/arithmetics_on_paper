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

    def __getattribute__(self, name, recursive=False):
        if not recursive and name == 'play':
            return self._play
        else:
            return super().__getattribute__(name)

    def _play(self, problem, *args, **kwargs):
        if type(problem) != dict:
            params = problem.params
        else:
            params = problem
        return self.__getattribute__('play', recursive=True)(
            params, *args, **kwargs)

    def play(self, problem):
        raise NotImplementedError
