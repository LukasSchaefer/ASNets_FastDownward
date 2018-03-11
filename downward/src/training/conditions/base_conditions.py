from . import register
from .. import append_register

import abc
from future.utils import with_metaclass


class Condition(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all conditions.
    Remember to register your concrete subclasses via one (or multiple) names
    at the dictionary 'register' via 'append_register'.
    """
    def __init__(self):
        """
        CONSTRUCTOR.
        Creates the _last_value member variable.
        """
        self._last_value = None

    def satisfied(self):
        """
        Evaluate if this condition is true of false.
        :return: bool value of this condition
        """
        self._last_value = self._satisfied()
        return self.last_value

    def last(self):
        """
        Return the last evaluated value of this condition without evaluating
        it again.
        :return: bool result of the last condition evaluation
        """
        return self._last_value

    @abc.abstractmethod
    def _satisfied(self):
        """
        Abstract method which performes the actual evaluation of the condition.
        :return: bool value of this condition
        """
        pass


class CTrue(Condition):
    def __init__(self):
        Condition.__init__(self)

    def _satisfied(self):
        return True


append_register(register, CTrue, "true", "t")


class CFalse(Condition):
    def __init__(self):
        Condition.__init__(self)

    def _satisfied(self):
        return False


append_register(register, CFalse, "false", "f")


class CNot(Condition):
    def __init__(self):
        Condition.__init__(self)

    def __init__(self, condition):
        self.condition = condition

    def _satisfied(self):
        return not self.condition.satisfied()


append_register(register, CNot, "not", "!", "~")


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


append_register(register, CAnd, "and", "&")


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


append_register(register, COr, "or", "|")

class CXor(Condition):
    def __init__(self):
        Condition.__init__(self)

    def __init__(self, condition1, condition2):
        self.condition1 = condition1
        self.condition2 = condition2

    def _satisfied(self):
        return self.condition1.satisfied() ^ self.condition2.satisfied()


append_register(register, CXor, "xor", "^")
