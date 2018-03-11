import abc
from future.utils import with_metaclass
import re


class InvalidSorterInput(Exception):
    pass


class ProblemSorter(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, feed=None):
        self._input = None
        self._output = None

        feed = feed.get_output() if isinstance(feed, ProblemSorter) else feed

        if feed is not None:
            self.sort(feed)

    def sort(self, feed):
        self._input = feed
        self._output = self._sort(feed)

    def get_input(self):
        return self._input

    def get_output(self):
        return self._output

    @abc.abstractmethod
    def _sort(self):
        pass

    @abc.abstractmethod
    def linearize(self):
        pass


class LexicographicArraySorter(ProblemSorter):
    def __init__(self, feed):
        ProblemSorter.__init__(self, feed)

    def _sort(self, feed):
        return sorted(feed)

    def linearize(self):
        return self.get_output()


class DifficultySorter(ProblemSorter):
    patterns_difficulty = [
        (re.compile("difficulty(-)?(\d)+"), 1),
        (re.compile("diff(-)?(\d)+"),1),
        (re.compile("d(-)?(\d)+"), 1),

    ]

    def __init__(self, feed):
        ProblemSorter.__init__(self, feed)

    def _sort(self, feed):
        diffs = {}
        for s in feed:
            diff = None
            for (pattern, idx) in DifficultySorter.patterns_difficulty:
                matches = pattern.findall(s)
                if len(matches) == 2:
                    diff = int(matches[0][idx])
                    break
            if diff is None:
                raise InvalidSorterInput("The 'DifficultySorter' was unable " +
                                         "to identify the difficult for item:" +
                                         s)
            else:
                if not diff in diffs:
                    diffs[diff] = []
                diffs[diff].append(s)

        return diffs

    def linearize(self):
        l = []
        for key in sorted(self.get_output().keys()):
            for elem in self.get_output()[key]:
                l.append(elem)
        return l
