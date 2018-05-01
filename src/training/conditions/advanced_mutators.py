from . import cregister, mregister

from .base_conditions import Condition
from .base_mutators import Mutator

from .. import parser
from .. import parser_tools as parset
from .. import main_register
from .. import vregister


class MGroup(Mutator):
    arguments = parset.ClassArguments('MGroup', Condition.arguments,
                                      ('mutators', False, None, mregister),
                                      ('condition_mutate', True, None, cregister),
                                      ('condition_signal', True, None, cregister),
                                      ('condition_reset', True, None, cregister),
                                      order=['mutators', 'condition_mutate',
                                             'condition_signal',
                                             'condition_reset', 'id'])

    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None, id=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset, id)
        self._mutators = mutators
        if not isinstance(self._mutators, list):
            self._mutators = list(self._mutators)

    def _mutate(self):
        for m in self._mutators:
            m.next()

    def _reset(self):
        for m in self._mutators():
            m._reset()

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  MGroup)


main_register.append_register(MGroup, "m_group")


class MRoundRobin(Mutator):
    arguments = parset.ClassArguments('MRoundRobin', Condition.arguments,
                                      ('mutators', False, None, mregister),
                                      ('condition_mutate', True, None, cregister),
                                      ('condition_signal', True, None, cregister),
                                      ('condition_reset', True, None, cregister),
                                      order=['mutators', 'condition_mutate',
                                             'condition_signal',
                                             'condition_reset', 'id'])

    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None, id=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset, id)
        self._mutators = mutators
        if not isinstance(self._mutators, list):
            self._mutators = list(self._mutators)

        self._next_mutator = 0

    def _mutate(self):
        self._mutators[self._next_mutator].next()
        self._next_mutator = (self._next_mutator + 1) % len(self._mutators)

    def _reset(self):
        self._next_mutator = 0
        for m in self._mutators:
            m._reset()

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  MRoundRobin)


main_register.append_register(MRoundRobin, "m_roundrobin")


class MLeft2Right(Mutator):
    arguments = parset.ClassArguments('MLeft2Right', Condition.arguments,
                                      ('mutators', False, None, mregister),
                                      ('condition_mutate', True, None, cregister),
                                      ('condition_signal', True, None, cregister),
                                      ('condition_reset', True, None, cregister),
                                      order=['mutators', 'condition_mutate',
                                             'condition_signal',
                                             'condition_reset', 'id'])

    def __init__(self, mutators, condition_mutate=None,
                 condition_signal=None, condition_reset=None, id=None):
        Mutator.__init__(self, condition_mutate, condition_signal,
                         condition_reset, id)
        self._mutators = mutators
        if not isinstance(self._mutators, list):
            self._mutators = list(self._mutators)

        self._next_mutator = 0

    def _mutate(self):
        for m in self._mutators:
            sig = m.next()
            if not sig:
                break

    def _reset(self):
        self._next_mutator = 0
        for m in self._mutators:
            m._reset()

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  MLeft2Right)


main_register.append_register(MLeft2Right, "m_left2right")
