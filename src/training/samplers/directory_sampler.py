from .base_sampler import saregister, Sampler

from .. import parser
from .. import parser_tools as parset

from ..parser_tools import main_register, ArgumentException
from ..variable import Variable
from ..problem_sorter import ProblemSorter

import os
import re


class DirectorySampler(Sampler):
    arguments = parset.ClassArguments('DirectorySampler', Sampler.arguments,
                                      ('directory', False, None, str),
                                      ('problem_sorter', False, None,
                                       main_register.get_register(ProblemSorter)),
                                      ('ignore', True, "", str),
                                      order=["sampler_bridge","directory", "problem_sorter",
                                             "ignore", "variables", "id"]
                                      )

    def __init__(self, sampler_bridge, directory, problem_sorter, ignore="", variables={}, id=None):
        Sampler.__init__(self, sampler_bridge, variables, id)
        self._directory = directory

        regex = re.compile(ignore) if ignore != "" else None
        files = []
        for file in os.listdir(self._directory):
            if parser.is_problem_file(os.path.join(self._directory, file)):
                if regex is None or regex.match(file):
                files.append(file)

        problem_sorter.sort(files)

        self._linearized_problems = problem_sorter.linearize()
        self._next_problem = 0
        self._sorted_problems = problem_sorter.get_output()

    def _next(self):
        if self._next_problem < len(self._linearized_problems):
            return self._linearized_problems[self._next_problem]
            self._next_problem += 1
        return None


main_register.append_register(DirectorySampler, "directorysampler")


class BatchDirectorySampler(DirectorySampler):
    arguments = parset.ClassArguments('BatchDirectorySampler',
                                      DirectorySampler.arguments,
                                      ('batch', False, None, parser.convert_int_or_inf),
                                      order=["sampler_bridge",
                                             "batch", "directory",
                                             "problem_sorter", "ignore",
                                             "variables", "id"]
                                      )

    def __init__(self, sampler_bridge, batch, directory, problem_sorter, ignore="", variables={}, id=None):
        DirectorySampler.__init__(self, sampler_bridge, directory, problem_sorter, ignore, variables, id)
        self._batch = batch

    def _initialize(self):
        pass

    def _sample(self):
        for i in range(self._batch):
            p = self._next()
            if p is None:
                return

    def _finalize(self):
        pass

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  DirSingleSampler)


main_register.append_register(BatchDirectorySampler, "batchdirsampler")
