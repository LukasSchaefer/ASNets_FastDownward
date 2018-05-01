from .. import main_register
from .. import parser
from .. import parser_tools as parset

from ..parser_tools import ArgumentException

import abc
from future.utils import with_metaclass
import re


class InvalidSorterInput(Exception):
    pass


class ProblemSorter(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all problem sorter.
    Do not forget to register your problem sorter subclass.
    via 'main_register.append_register' of the main package.
    """

    # needs reference to ClassRegister of ProblemSorter, thus, has to be
    # defined after class ProblemSorter.
    arguments = None

    def __init__(self, prev_sorter=None, id=None):
        """
        CONSTRUCTOR
        :param prev_sorter: if not None, then on calling sort the input is
                            first given to prev_sorter and the output of prev_
                            sorter is used as input
        :param id: id of the ProblemSorter
        """
        self.id = id
        self._prev = prev_sorter
        self._output = None

    def sort(self, feed, linearize=False):
        """
        Sorts the problems given to the function. The expected data format of
        feed depends on the implementation of _sort in concrete subclasses.
        If the sorter has a previous sorter registered, then feed is given
        to the previous sorter and the output is further processed by this
        sorter.
        :param feed: problems to sort
        :return: sorted structure of problems (not necessarily vector)
        """
        if self._prev is not None:
            feed = self._prev.sort(feed)

        self._output = self._sort(feed)

        return self.linearize() if linearize else self._output

    def linearize(self, feed=None):
        """
        Linearizes data in the format of the output of this sorter into a
        vector.
        :param feed: if None, then the previously sorted output is linearized,
                     otherwise feed is linearized
        :return: vector of sorted problems
        """
        if feed is None:
            feed = self._output
        return self._linearize(feed)

    def get_prev(self):
        return self._prev

    def get_output(self):
        return self._output

    @abc.abstractmethod
    def _sort(self):
        pass

    @abc.abstractmethod
    def _linearize(self, feed):
        pass

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, ProblemSorter, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base problem sorter "
                                    "can only be used for look up of any "
                                    "previously defined condition via "
                                    "'ProblemSorter(id=ID)'")


main_register.append_register(ProblemSorter, "problemsorter", "sorter")
pregister = main_register.get_register(ProblemSorter)

ProblemSorter.arguments = parset.ClassArguments('ProblemSorter', None,
        ('prev_sorter', True, None, pregister),
        ('id', True, None, str)
    )



class LexicographicIterableSorter(ProblemSorter):

    arguments = parset.ClassArguments('LexicographicIterableSorter',
                                      ProblemSorter.arguments)

    def __init__(self, prev_sorter=None, id=None):
        ProblemSorter.__init__(self, prev_sorter, id)

    def _sort(self, feed):
        return sorted(feed)

    def _linearize(self, feed):
        return feed

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  LexicographicIterableSorter)


main_register.append_register(LexicographicIterableSorter, "lexicographic_sorter",
                "s_lex")


class DifficultySorter(ProblemSorter):

    arguments = parset.ClassArguments('DifficultySorter',
                                      ProblemSorter.arguments)

    patterns_difficulty = [
        (re.compile("difficulty(-)?(\d)+"), 1),
        (re.compile("diff(-)?(\d)+"),1),
        (re.compile("d(-)?(\d)+"), 1),

    ]

    def __init__(self, prev_sorter=None, id=None):
        ProblemSorter.__init__(self, prev_sorter, id)

    def _sort(self, feed):
        diffs = {}
        for s in feed:
            diff = None
            for (pattern, idx) in DifficultySorter.patterns_difficulty:
                matches = pattern.findall(s)
                if len(matches) == 1:
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

    def _linearize(self, feed):
        l = []
        for key in sorted(feed):
            for elem in feed[key]:
                l.append(elem)
        return l

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  DifficultySorter)

main_register.append_register(DifficultySorter, "difficulty_sorter", "s_diff")