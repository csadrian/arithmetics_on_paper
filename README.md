# arithmetics_on_paper

1. step: add Problem and Solver to the dictionaries in gen_util.py

```
  module_to_problem_dict = {
     ...
     'arithmetic__add_or_sub': problems.arithmetic_problems.AddOrSubProblem
     ...
     }
  problem_to_solver_dict = {
     ...
     'AddOrSubProblem': solvers.arithmetic_solvers.AddOrSubSolver
     ...
     }
```

Here "add_or_sub" is the name of function that generates a specific problem type in mathematics_dataset.

2. step: run the generator: python -m mathematics_dataset.generate --filter=arithmetic --save_plots=True --per_train_module=5 --per_test_module=5 --paper_size=40

Note the options!
