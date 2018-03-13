from . import cregister

from .base_conditions import Condition, CFalse, CTrue

from .. import main_register
from .. import parser
from .. import parser_tools as parset
from .. import vregister



class CFlip(Condition):
    arguments = parset.ClassArguments('CFlip', Condition.arguments,
                                      ('init_value', True, True, parser.convert_bool),
                                      ('flip_points', True, [1], int),
                                      order=['init_value', 'flip_points', 'id'])

    def __init__(self, init_value=True, flip_points=[1], id=None):
        Condition.__init__(self, id)
        self.value = init_value
        self.flip_points = flip_points
        self.next_flip_point = 0
        self.counter = -1

    def _satisfied(self):
        self.counter += 1
        if self.counter >= self.flip_points[self.next_flip_point]:
            self.value = not self.value
            self.next_flip_point = ((self.next_flip_point + 1)
                                    % len(self.flip_points))
            self.counter = 0
        return self.value

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  CFlip)


main_register.append_register(CFlip, "flip")


class CThreshold(Condition):
    arguments = parset.ClassArguments('CThreshold', Condition.arguments,
                                      ('variable', False, None, vregister),
                                      ('threshold', False, None, int),
                                      order=['variable', 'threshold', 'id'])

    def __init__(self, variable, threshold, id=None):
        Condition.__init__(self, id)
        self.variable = variable
        self.threshold = threshold

    def _satisfied(self):
        return self.variable.value >= self.threshold

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  CThreshold)


main_register.append_register(CThreshold, "threshold")


class CModulo(Condition):
    arguments = parset.ClassArguments('CModulo', Condition.arguments,
                                      ('variable', False, None, vregister),
                                      ('modulo', False, None, int),
                                      order=['variable', 'modulo', 'id'])

    def __init__(self, variable, modulo, id=None):
        Condition.__init__(self, id)
        self.variable = variable
        self.modulo = modulo

    def _satisfied(self):
        return self.variable.value % self.modulo == 0

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  CModulo)


main_register.append_register(CModulo, "modulo")


class CHistory(Condition):
    arguments = parset.ClassArguments('CHistory', Condition.arguments,
                                      ('condition_class', False, None,
                                       cregister.get_reference),
                                      ('length', False, None, int),
                                      ('condition', False, None, cregister),
                                      order=['condition_class', 'length',
                                             'condition', 'id'])

    def __init__(self, condition_class, length, condition, id=None):
        Condition.__init__(self, id)
        self.condition_class = condition_class
        self.length = length
        self.condition = condition
        self.history = []

    def _satisfied(self):
        if self.condition.satisfied():
            self.history.append(CTrue())
        else:
            self.history.append(CFalse())

        if len(self.history) > self.length:
            self.history = self.history[-self.length:]

        cnd = self.condition_class(self.history)
        return cnd.satisfied()

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  CHistory)


main_register.append_register(CHistory, "history")


class CHistories(Condition):
    arguments = parset.ClassArguments('CHistories', Condition.arguments,
                                      ('history_condition_class', False, None,
                                       cregister.get_reference),
                                      ('timestep_condition_class', False, None,
                                       cregister.get_reference),
                                      ('length', False, None, int),
                                      ('conditions', False, None, cregister),
                                      order=['history_condition_class',
                                             'timestep_condition_class',
                                             'length', 'conditions', 'id'])

    def __init__(self, history_condition_class, timestep_condition_class,
                 length, conditions, id=None):
        Condition.__init__(self, id)
        self.history_condition_class = history_condition_class
        self.timestep_condition_class = timestep_condition_class
        self.length = length
        self.conditions = conditions
        if not isinstance(self.conditions, list):
            self.conditions = [self.conditions]
        self.history = []

    def _satisfied(self):
        cnd_step = self.timestep_condition_class(self.conditions)
        if cnd_step.satisfied():
            self.history.append(CTrue())
        else:
            self.history.append(CFalse())

        if len(self.history) > self.length:
            self.history = self.history[-self.length:]

        cnd = self.history_condition_class(self.history)
        return cnd.satisfied()

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  CHistories)


main_register.append_register(CHistories, "histories")
