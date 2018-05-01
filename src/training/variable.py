from . import parser
from . import parser_tools as parset
from . import main_register


register = {}


class VariableTypeException(Exception):
    pass


class Variable(object):

    arguments = parset.ClassArguments('Variable', None,
        ("value", True, None, str),
        ('vtype', True, None, str),
        ('id', True, None, str)
    )

    def __init__(self, value, vtype=None, id=None):
        self.id = id

        if vtype is None:
            self.value = value
            self.vtype = vtype

        elif isinstance(vtype, type):
            self.value = vtype(value)
            self.vtype = vtype

        elif isinstance(vtype, str):
            if vtype == "str":
                self.vtype = str
            elif vtype == "int":
                self.vtype = int
            elif vtype == "float":
                self.vtype = float
            elif vtype == "bool":
                self.vtype = bool
            else:
                raise VariableTypeException("The given type '" + vtype + "'"
                                            " is unkown.")

            if self.vtype == bool:
                self.value = parser.convert_bool(value)
            else:
                self.value = self.vtype(value)

    def __str__(self):
        return ("Variable(" + str(self.value) + ", " + str(self.vtype) + ", "
                + str(self.id) + ")")

    @staticmethod
    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  Variable, True)


main_register.append_register(Variable, "variable", "var", "v")
vregister = main_register.get_register(Variable)