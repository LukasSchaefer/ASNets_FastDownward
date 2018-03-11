import abc
from future.utils import with_metaclass


class Condition(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        self._last_value = None

    def satisfied(self):
        self._last_value = self._satisfied()
        return self.last_value

    def last(self):
        return self._last_value

    @abc.abstractmethod
    def _satisfied(self):
        pass


class CTrue(Condition):
    def __init__(self):
        Condition.__init__(self)

    def _satisfied(self):
        return True


class CFalse(Condition):
    def __init__(self):
        Condition.__init__(self)

    def _satisfied(self):
        return False


class CNot(Condition):
    def __init__(self):
        Condition.__init__(self)

    def __init__(self, condition):
        self.condition = condition

    def _satisfied(self):
        return not self.condition.satisfied()


class CAnd(Condition):
    def __init__(self):
        Condition.__init__(self)

    def __init__(self, *args):
        self.conditions = args

    def _satisfied(self):
        for c in self.conditions:
            if not c.satisfied():
                return False
        return True


class COr(Condition):
    def __init__(self):
        Condition.__init__(self)

    def __init__(self, *args):
        self.conditions = args

    def _satisfied(self):
        for c in self.conditions:
            if c.satisfied():
                return True
        return False


class CXor(Condition):
    def __init__(self):
        Condition.__init__(self)

    def __init__(self, condition1, condition2):
        self.condition1 = condition1
        self.condition2 = condition2

    def _satisfied(self):
        return self.condition1.satisfied() ^ self.condition2.satisfied()
