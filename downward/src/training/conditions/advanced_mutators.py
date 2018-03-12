from . import register

from . import Condition

from .base_mutators import Mutator

from .. import parser

class MGroup(Mutator):
    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset)
        self.mutators = mutators

    def _mutate(self):
        for m in self._mutators:
            m.next()

    def _reset(self):
        for m in self._mutate():
            m._reset()

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, MGroup)


parser.append_register(register, MGroup, "m_group")


class MRoundRobin(Mutator):
    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset)
        self._mutators = mutators
        self._next_mutator = 0

    def _mutate(self):
        self._mutators[self._next_mutator].next()
        self._next_mutator += 1

    def _reset(self):
        self._next_mutator = 0
        for m in self._mutate():
            m._reset()

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, MRoundRobin)


parser.append_register(register, MRoundRobin, "m_roundrobin")


class MLeft2Right(Mutator):
    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset)
        self._mutators = mutators
        self._next_mutator = 0

    def _mutate(self):
        for m in self._mutators:
            sig = m.next()
            if not sig:
                break

    def _reset(self):
        self._next_mutator = 0
        for m in self._mutate():
            m._reset()

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, MLeft2Right)


parser.append_register(register, MLeft2Right, "m_Left2Right")
