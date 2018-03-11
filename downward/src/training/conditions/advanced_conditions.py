from .base_conditions import Condition, CFalse, CTrue

from ..variable import Variable


class CFlip(Condition):
    def __init__(self, init_value = True, flip_points=[1]):
        Condition.__init__(self)
        self.value = init_value
        self.flip_points = flip_points
        self.next_flip_point = 0
        self.counter = 0

    def _satisfied(self):
        self.counter += 1
        if self.counter > self.flip_points[self.next_flip_point]:
            self.value = not self.value
            self.next_flip_point = ((self.next_flip_point + 1)
                                    % len(self.flip_points))
            self.counter = 0
        return self.value


class CThreshold(Condition):
    def __init__(self, variable, threshold):
        Condition.__init__(self)
        self.variable = variable
        self.threshold = threshold

    def _satisfied(self):
        return self.variable.value >= self.threshold


class CModulo(Condition):
    def __init__(self, variable, modulo):
        Condition.__init__(self)
        self.variable = variable
        self.modulo = modulo

    def _satisfied(self):
        return self.variable.value % self.modulo == 0


class CHistory(Condition):
    def __init__(self, condition_class, length, condition):
        Condition.__init__(self)
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

        cnd = self.condition_class(*self.history)
        return cnd.satisfied()


class CHistories(Condition):
    def __init__(self, history_condition_class, timestep_condition_class,
                 length, conditions):
        Condition.__init__(self)
        self.history_condition_class = history_condition_class
        self.timestep_condition_class = timestep_condition_class
        self.length = length
        self.conditions = conditions
        self.history = []

    def _satisfied(self):
        cnd_step = self.timestep_condition_class(*self.conditions)
        if cnd_step.satisfied():
            self.history.append(CTrue())
        else:
            self.history.append(CFalse())

        if len(self.history) > self.length:
            self.history = self.history[-self.length:]

        cnd = self.condition_class(*self.history)
        return cnd.satisfied()