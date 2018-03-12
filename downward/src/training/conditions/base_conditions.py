from . import register

from .. import parser

from ..parser import ArgumentException

import abc
from future.utils import with_metaclass


class Condition(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all conditions.
    Remember to register your concrete subclasses via one (or multiple) names
    at the dictionary 'register' via 'append_register'.

    Every concrete subclass shall have a class method
    'parse(parse_tree, definitions)' which is used to parse an object of the
    class type (if parse_tree defines solely an id, then an object of the
    subclass with the given id shall be looked for and returned if existing).
    This base class also possesses the 'parse' method and is registered. It
    cannot parse objects of the base class, but it can also look up conditions
    by id.
    To simplify the parsing for every class its parameters should be defined
    in a ClassArgument object. Then the standard methods (check for example
    how it is done in base_conditions.py) handle the whole parsing.
    REMARK: All objects subclass of this base class are stored in the same
    cache, thus, ids may not be reused between objects of different subclasses.
    If looking up objects by id, the abstract base class may look up objects of
    any of its subclasses, ALL subclasses may only look up objects of exactly
    their type.
    """

    arguments = parser.ClassArguments('Condition', None,
        ('id', True, None, str)
    )


    def __init__(self, id=None):
        """
        CONSTRUCTOR.
        Creates the _last_value member variable.
        :param id: id to identify this condition
        """
        self.id = id
        self._last_value = None

    def satisfied(self):
        """
        Evaluate if this condition is true of false.
        :return: bool value of this condition
        """
        self._last_value = self._satisfied()
        return self._last_value

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

    def parse(tree, definitions):
        obj = parser.try_lookup_obj(tree, definitions, Condition, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base condition can "
                                    "only be used for look up of any previously"
                                    " defined condition via 'Condition(id=ID)'")


parser.append_register(register, Condition, "condition", "cnd")


class CTrue(Condition):

    arguments = parser.ClassArguments('CTrue', Condition.arguments)

    def __init__(self, id=None):
        Condition.__init__(self, id)

    def _satisfied(self):
        return True

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, CTrue)


parser.append_register(register, CTrue, "ctrue", "true", "t")


class CFalse(Condition):

    arguments = parser.ClassArguments('CTrue', Condition.arguments)

    def __init__(self, id=None):
        Condition.__init__(self, id)

    def _satisfied(self):
        return False

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, CFalse)


parser.append_register(register, CFalse, "cfalse", "false", "f")


class CNot(Condition):
    arguments = parser.ClassArguments('CNot', Condition.arguments,
                                      ('condition', False, None, register),
                                      order=['condition', 'id'])

    def __init__(self, condition, id=None):
        Condition.__init__(self, id)
        self.condition = condition

    def _satisfied(self):
        return not self.condition.satisfied()

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, CNot)


parser.append_register(register, CNot, "not", "!", "~")


class CAnd(Condition):
    arguments = parser.ClassArguments('CAnd', Condition.arguments,
                                      ('conditions', False, None, register),
                                      order=['conditions', 'id'])

    def __init__(self, conditions, id=None):
        Condition.__init__(self, id)
        if isinstance(conditions, list):
            self.conditions = conditions
        else:
            self.conditions = [conditions]

    def _satisfied(self):
        sat = True
        for c in self.conditions:
            if not c.satisfied():
                sat = False
        return sat

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, CAnd)


parser.append_register(register, CAnd, "and", "&")


class COr(Condition):
    arguments = parser.ClassArguments('COr', Condition.arguments,
                                      ('conditions', False, None, register),
                                      order=['conditions', 'id'])

    def __init__(self, conditions, id=None):
        Condition.__init__(self, id)
        if isinstance(conditions, list):
            self.conditions = conditions
        else:
            self.conditions = [conditions]

    def _satisfied(self):
        sat = False
        for c in self.conditions:
            if c.satisfied():
                sat = True
        return sat

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, COr)


parser.append_register(register, COr, "or", "|")


class CXor(Condition):
    arguments = parser.ClassArguments('CXor', Condition.arguments,
                                      ('condition1', False, None, register),
                                      ('condition2', False, None, register),
                                      order=['condition1', 'condition2', 'id'])

    def __init__(self, condition1, condition2, id=None):
        Condition.__init__(self, id)
        self.condition1 = condition1
        self.condition2 = condition2

    def _satisfied(self):
        return self.condition1.satisfied() ^ self.condition2.satisfied()

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Condition, CXor)


parser.append_register(register, CXor, "xor", "^")

