from .. import parser
from .. import parser_tools as parset

from ..parser_tools import main_register, ArgumentException
from ..variable import Variable
from ..problem_sorter import ProblemSorter

import abc
from future.utils import with_metaclass
import os


class InvalidMethodCallException(Exception):
    pass


def is_problem_file(path):
    if not os.path.isfile(path):
        return False
    file = os.path.basename(path)
    if not file.endswith(".pddl"):
        return False
    if file.find("domain") != -1:
        return False
    return True


class Sampler(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all sampler.
    Do not forget to register your network subclass in this packages 'register'
    dictionary via 'append_register' of the main package.
    """

    arguments = parset.ClassArguments("Sampler", None,
                                      ('variables', True, {},
                                       main_register.get_register(Variable)),
                                      ('id', True, None, str),
                                      variables=[('sample_calls', 0, int)],
                                      )

    def __init__(self, variables={}, id=None):
        if not isinstance(variables, dict):
            raise ArgumentException("The provided variables have to be a map. "
                                    "Please define them as {name=VARIABLE,...}.")
        self.variables = variables
        self.id = id

        self.var_sample_calls, = Sampler.arguments.validate_and_return_variables(variables)

        self.initialized = False
        self.finalized = False

    def initialize(self):
        if not self.initialized:
            self._initialize()
            self.initialized = True
        else:
            raise InvalidMethodCallException("Multiple initializations of"
                                             "sampler.")

    def sample(self, msgs):
        if self.var_sample_calls is not None:
            self.var_sample_calls.value += 1
        self._sample(msgs)

    def finalize(self):
        if not self.finalized:
            self._finalize()
            self.finalized = True
        else:
            raise InvalidMethodCallException("Mutliple finalization calls of"
                                             "sampler.")

    @abc.abstractmethod
    def _initialize(self):
        pass

    @abc.abstractmethod
    def _sample(self, msgs):
        pass

    @abc.abstractmethod
    def _finalize(self):
        pass

    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Sampler, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base sampler can "
                                    "only be used for look up of any previously"
                                    " defined schema via 'Sampler(id=ID)'")


main_register.append_register(Sampler, "sampler")
saregister = main_register.get_register(Sampler)


class DirectorySampler(Sampler):
    arguments = parset.ClassArguments('DirectorySampler', Sampler.arguments,
                                      ('directory', False, None, str),
                                      ('problem_sorter', False, None,
                                       main_register.get_register(ProblemSorter)),
                                      order=["directory", "problem_sorter",
                                             "variables", "id"]
                                      )

    def __init__(self, directory, problem_sorter, variables={}, id=None):
        Sampler.__init__(self, variables, id)
        self._directory = directory

        files = []
        for file in os.listdir(self._directory):
            if is_problem_file(os.path.join(self._directory, file)):
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
                                      order=["batch", "directory",
                                             "problem_sorter",
                                             "variables", "id"]
                                      )

    def __init__(self, batch, directory, problem_sorter, variables={}, id=None):
        DirectorySampler.__init__(self, directory, problem_sorter, variables, id)
        self._batch = batch

    def _initialize(self):
        pass

    def _sample(self, msgs):
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
