from .base_sampler import saregister, Sampler

from .. import parser
from .. import parser_tools as parset
from .. import SampleBatchData

from ..parser_tools import main_register, ArgumentException
from ..variable import Variable
from ..problem_sorter import ProblemSorter, LexicographicIterableSorter

import os
import re


class IterableFileSampler(Sampler):
    arguments = parset.ClassArguments('IterableFileSampler', Sampler.arguments,
                                      ('iterable', False, [], str),
                                      ('batch', True, None, int),
                                      ('problem_sorter', True, None,
                                       main_register.get_register(ProblemSorter)),
                                      ('merge', True, True, parser.convert_bool,
                                       "If true merges the data of all sampling"
                                       " runs in a single container otherwise,"
                                       "each run has its own container."),
                                      order=["sampler_bridge", "iterable",
                                             "batch", "problem_sorter", "merge",
                                             "variables", "id"]
                                      )

    def __init__(self, sampler_bridge, iterable=[],
                 batch=None, problem_sorter=None, merge=True,
                 variables={}, id=None):
        Sampler.__init__(self, sampler_bridge, variables, id)

        self._iterable = [x for x in iterable]  # if it would be a set or s.th
        self._batch = batch
        self._problem_sorter = problem_sorter
        self._merge = merge


    def _initialize(self):
        if self._problem_sorter is not None:
            self._iterable = self._problem_sorter.sort(self._iterable, linearize=True)
        self._next_problem = 0

    def _next(self):
        if self._next_problem < len(self._iterable):
            self._next_problem += 1
            return self._iterable[self._next_problem - 1]

        return None

    def _sample(self,**kwargs):
        nb = len(self._iterable) if self._batch is None else self._batch

        datas = []
        merge_container = None
        for i in range(nb):
            problem = self._next()
            for sb in self.sbridges:
                datas.append(sb.sample(problem, data_container=merge_container))
                if self._merge:
                    merge_container = datas[-1]

        return [merge_container] if self._merge else datas

    def _finalize(self):
        pass

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  IterableFileSampler)


main_register.append_register(IterableFileSampler, "ifsampler")


class DirectorySampler(IterableFileSampler):
    arguments = parset.ClassArguments('DirectorySampler', IterableFileSampler.arguments,
        ('root', False, None, str, "List of directories within which problem files are searched"),
        ("filter_dir", True, None, str),
        ("filter_file", True, None, str),
        ("max_depth", True, None, int),
        ("selection_depth", True, None, int),
        ("iterable", True, None, str,
        "List of problem files independently of the directory traversing which is sampled"),
        ("ignore", True, None, str,
         "Iterable of file or directory paths to ignore if found"),
        order=["sampler_bridge", "root",
             "filter_dir", "filter_file", "ignore",
             "max_depth", "selection_depth",
             "batch", "problem_sorter", "merge",
             "variables", "id", "iterable"]
        )


    def __init__(self, sampler_bridge, root, filter_dir=None, filter_file=None,
                 ignore=None,
                 max_depth=None, selection_depth=None,
                 batch=None, problem_sorter=None, merge=True,
                 variables={}, id=None, iterable=None):
        iterable = [] if iterable is None else iterable
        IterableFileSampler.__init__(self, sampler_bridge, iterable, batch,
                                     problem_sorter, merge, variables, id)

        if not isinstance(root, list):
            root = [root]
        self._root = root
        self._filter_dir = DirectorySampler.compile_regexes(filter_dir)
        self._filter_file = DirectorySampler.compile_regexes(filter_file)
        self._ignore = set() if ignore is None else set(ignore)
        self._max_depth = max_depth
        self._selection_depth = selection_depth

        for root in self._root:
            todo = [(root, 0)]
            while len(todo) > 0:
                (dir, depth) = todo[0]
                del todo[0]

                for item in os.listdir(dir):
                    path_item = os.path.join(dir, item)
                    if path_item in self._ignore:
                        continue

                    if os.path.isdir(path_item):
                        if ((
                                self._max_depth is None or depth < self._max_depth)
                                and DirectorySampler.check_regexes(
                                    item, self._filter_dir)):
                            todo.append((path_item, depth + 1))
                    else:
                        if ((self._selection_depth is None or depth >= self._selection_depth)
                                and DirectorySampler.check_regexes(item, self._filter_file)
                                and parser.is_problem_file(path_item)):
                            self._iterable.append(path_item)


    @staticmethod
    def compile_regexes(regexes):
        if regexes is None:
            return []

        for idx in range(len(regexes)):
            regexes[idx] = re.compile(regexes[idx])

        return regexes

    @staticmethod
    def check_regexes(name, regexes):
        for regex in regexes:
            if not regex.match(name):
                return False
        return True

main_register.append_register(DirectorySampler, "directorysampler")
