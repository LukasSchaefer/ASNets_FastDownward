from .. import dependencies

if dependencies.problem_sorter:
    from .base_sorters import InvalidSorterInput

    from .base_sorters import ProblemSorter
    from .base_sorters import DifficultySorter, LexicographicIterableSorter