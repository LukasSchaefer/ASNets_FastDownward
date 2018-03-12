from . import parser


register = {}


class VariableTypeException(Exception):
    pass


class Variable(object):

    arguments = parser.ClassArguments('Variable', None,
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

    def parse(tree, definitions):
        return parser.try_whole_obj_parse_process(tree, definitions,
                                                  Variable, None, Variable)


parser.append_register(register, Variable, "variable", "var", "v")