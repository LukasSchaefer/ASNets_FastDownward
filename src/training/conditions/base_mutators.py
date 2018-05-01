from . import cregister

from .base_conditions import Condition, CTrue, CFalse
from .advanced_conditions import CThreshold, CModulo

from .. import main_register
from .. import parser
from .. import parser_tools as parset
from .. import vregister

import abc


class Mutator(Condition):
    def __init__(self, condition_mutate=None,
                 condition_signal=None, condition_reset=None, id=None):
        Condition.__init__(self, id)
        self.condition_mutate = CTrue() if condition_mutate is None else condition_mutate
        self.condition_signal = CTrue() if condition_signal is None else condition_signal
        self.condition_reset = CFalse() if condition_reset is None else condition_reset

        self._fired_mutate = False
        self._fired_signal = False
        self._fired_reset = False

    def fired_mutate(self):
        return self._fired_mutate

    def fired_signal(self):
        return self._fired_signal

    def fired_reset(self):
        return self._fired_reset

    def next(self):
        self._fired_mutate = self.condition_mutate.satisfied()
        if self._fired_mutate:
            self._mutate()

        self._fired_signal = self.condition_signal.satisfied()

        self._fired_reset = self.condition_reset.satisfied()
        if self._fired_reset:
            self._reset()

        return self._satisfied()

    @abc.abstractmethod
    def _mutate(self):
        pass

    @abc.abstractmethod
    def _reset(self):
        pass

    def _satisfied(self):
        return self.fired_signal()

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Mutator, None)
        if obj is not None:
            return obj
        else:
            raise parset.ArgumentException(
                                    "The definition of the base mutation can "
                                    "only be used for look up of any previously"
                                    " defined mutations via 'Mutator(id=ID)'")


main_register.append_register(Mutator, "mutator")
mregister = main_register.get_register(Mutator)

class MAdd(Mutator):
    arguments = parset.ClassArguments('MAdd', Condition.arguments,
                                      ('variable', False, None, vregister),
                                      ('condition_mutate', True, None, cregister),
                                      ('condition_signal', True, None, cregister),
                                      ('condition_reset', True, None, cregister),
                                      ('step', True, 1, int),
                                      ('reset_value', True, 0, int),
                                      order=['variable', 'condition_mutate',
                                             'condition_signal',
                                             'condition_reset', 'step',
                                             'reset_value', 'id'])

    def __init__(self, variable, condition_mutate=None,
                 condition_signal = None, condition_reset=None,
                 step=1, reset_value=0, id=None):
        Mutator.__init__(self, condition_mutate=condition_mutate,
                         condition_signal=condition_signal,
                         condition_reset=condition_reset, id=id)
        self._variable = variable
        self._step = step
        self._reset_value = reset_value

    def _mutate(self):
        self._variable.value += self._step

    def _reset(self):
        self._variable.value = self._reset_value

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  MAdd)


main_register.append_register(MAdd, "m_add")


class MThreshold(MAdd):
    arguments = parset.ClassArguments('MThreshold', Condition.arguments,
                                      ('variable', False, None, vregister),
                                      ('threshold', False, None, int),
                                      ('reset_value', True, 0, int),
                                      ('step', True, 1, int),
                                      order=['variable', 'threshold',
                                             'reset_value', 'step', 'id'])

    def __init__(self, variable, threshold, reset_value=None, step=1, id=None):
        MAdd.__init__(self, variable, None, CThreshold(variable, threshold),
                      CThreshold(variable, threshold), step, reset_value, id)

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  MThreshold)


main_register.append_register(MThreshold, "m_threshold")


class MModulo(MAdd):
    arguments = parset.ClassArguments('MModulo', Condition.arguments,
                                      ('variable', False, None, vregister),
                                      ('modulo', False, None, int),
                                      ('reset_value', True, 0, int),
                                      ('step', True, 1, int),
                                      order=['variable', 'modulo',
                                             'reset_value', 'step', 'id'])

    def __init__(self, variable, modulo, reset_value=None, step=1, id=None):
        MAdd.__init__(self, variable, None, CModulo(variable, modulo),
                      CModulo(variable, modulo), step, reset_value, id)

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  MModulo)


main_register.append_register(MModulo, "m_modulo")