from .base_conditions import Condition, CTrue, CFalse
from .advanced_conditions import CThreshold, CModulo

from . import register

from .. import parser

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

        return

    @abc.abstractmethod
    def _mutate(self):
        pass

    @abc.abstractmethod
    def _reset(self):
        pass

    def _satisfied(self):
        return self.fired_signal()


class MAdd(Mutator):
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

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, MAdd)


parser.append_register(register, MAdd, "m_add")


class MThreshold(MAdd):
    def __init__(self, variable, threshold, reset_value=None, step=1, id=None):
        MAdd.__init__(self, variable, None, CThreshold(variable, threshold),
                      CThreshold(variable, threshold), step, reset_value, id)

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, MThreshold)


parser.append_register(register, MThreshold, "m_threshold")


class MModulo(MAdd):
    def __init__(self, variable, threshold, reset_value=None, step=1, id=None):
        MAdd.__init__(self, variable, None, CModulo(variable, threshold),
                      CModulo(variable, threshold), step, reset_value, id)

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, MModulo)


parser.append_register(register, MModulo, "m_modulo")