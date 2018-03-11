import abc
from future.utils import with_metaclass
import os

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

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def sample(self, msg_self, msg_network):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass


class DirectorySampler(Sampler):
    def __init__(self, directory, problem_sorter):
        self._directory = directory
        
        files = []
        for file in os.listdir(self._directory):
            if is_problem_file(os.path.join(self._directory, file)):
                files.append(file)

        problem_sorter.sort(files)

        self._linearized_problems = problem_sorter.linearize()
        self._sorted_problems = problem_sorter.get_output()


