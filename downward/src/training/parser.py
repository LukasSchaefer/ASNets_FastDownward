from . import parse_tree


class ArgumentException(Exception):
    pass

class ClassReferenceConverter(object):
    def __init__(self, register):
        self.register = register

    def convert(self, arg):
        if arg in self.register:
            return self.register[arg]
        else:
            raise ArgumentException("No class reference found for " + str(arg))



def is_class_registered(obj, register):
    for key in register:
        if type(register[key]) == type(obj):
            return True
    return False


def find_object_in_global_definitions(tree, definitions, register=None):
    if tree.data[0] in definitions["global"]:
        obj = definitions["global"][tree.data[0]]
        if not tree.empty():
            raise ArgumentException(str(tree.data[0]) + " is a previously "
                                    "cached variable. This variable cannot be"
                                    " given parameters again.")
        if register is None or is_class_registered(obj, register):
            return obj
        else:
            raise ArgumentException("The usage of " + str(tree.data[0])
                                    + " which is a globally cached object is "
                                    + "invalid here, because its type is not"
                                    + "allowed.")

    return None


class ClassArguments:
    def __init__(self, class_name, base_class_arguments, *args, order=None):
        """
        List of arguments which a class needs to be constructe. Each entry is
        of the form (name, optional, default, register_or_converter) with:
        name = name of the argument
        optional = if not optional, then the user has to define this field else
                    if the field is missing, the default value is used.
        default = default value to use if the field is not specified
        register_or_converter = if a register is provided (dictionary mapping to
                                classes), then the subtree is given to the parse
                                method of the mapped class. Else the value is
                                interpreted as function and the data of the
                                subtrees root is feed into the node.
                                If the subtrees root is a list, then this is
                                done for every child of it and the results are
                                put into a list.

        :param class_name: name of the class associated with this object
        :param args: sequence of (name, optional, default,
                        register_or_converter) tuple
        """
        self.class_name = class_name

        self.order = []
        self.parameters = {}
        if base_class_arguments is not None:
            for arg_name in base_class_arguments.order:
                self.parameters[arg_name] = base_class_arguments.parameters[arg_name]
                self.order.append(arg_name)

        for arg in args:
            # if not redefining previously known parameter, add it to the list
            if arg[0] not in self.parameters:
                self.order.append(arg[0])
            self.parameters[arg[0]] = arg

        if order is not None:
            self.change_order(*order)

    def change_order(self, *args):
        if len(set(args)) != len(args):
            raise ValueError("New order contains some element multiple times.")

        if set(self.order) != set(args):
            raise ValueError("New order does not solely reorders the parameters"
                             ", but adds more and/or skips some.")

        self.order = args

    def parse(self, parameter, tree, definitions):
        # unknown parameter
        if parameter not in self.parameters:
            raise ArgumentException("Tried to parse unknown parameter "
                                    + str(parameter) + " for object of class "
                                    + self.class_name + ".")

        (name, optional, default, reg_or_conv) = self.parameters[parameter]

        # argument not provided by user
        if tree is None:
            if optional:
                return default
            else:
                raise ArgumentException("Obligatory argument " + str(name)
                                        + " missing for object of type "
                                        + str(self.class_name))

        # check if globally defined (if = without register, else with register)
        if reg_or_conv is not None or  callable(reg_or_conv):
            obj = find_object_in_global_definitions(tree, definitions, None)
            if obj is not None:
                return obj
        else:
            obj = find_object_in_global_definitions(tree, definitions, reg_or_conv)
            if obj is not None:
                return obj

        #parse objects from strings
        if tree.data[0] == "list":
            obj = []
            for child in tree.children:
                obj_child = self.parse(parameter, child, definitions)
                obj.append(obj_child)
        else:
            if reg_or_conv is None:
                obj = tree.data[0]

            #is converter function
            elif callable(reg_or_conv):
                print(name)
                obj = reg_or_conv(tree.data[0])

            #is register
            else:
                type_name = tree.data[0].lower()
                if type_name not in reg_or_conv:
                    raise ArgumentException("Unkown object type or id '"
                                            + str(tree.data[0])
                                            + "' for parameter '"
                                            + str(parameter) + "' of type '"
                                            + str(self.class_name) + "'.")
                obj = reg_or_conv[type_name].parse(tree, definitions)

        return obj



def convert_bool(value):
    if value in [True, 1, "t", "T", "1", "true", "True"]:
        return True
    elif value in [False, 0, "f", "F", "0", "false", "False"]:
        return False
    else:
        ValueError("Unable to parse input value to boolean, please use 'True' "
                   "or 'False' as values.")


def append_register(dictionary, item, *args):
    """
    Register in the dictionary for every key given in args the item.

    :param dictionary: dict in which the relation shall be registered
    :param item: item to register (e.g. constructor)
    :param args: names under which the item can be found
    :return: None
    """

    for key in args:
        if not key in dictionary:
            dictionary[key] = item
        else:
            raise KeyError("Internal Error: In a register are multiple times "
                           + "items for the same key (" + key + ") defined.")


def append_definitions(definitions, id, obj):
    if id is None:
        return

    for category in definitions:
        if category == "global":
            continue

        if isinstance(obj, category):
            if id in definitions[category]:
                raise ArgumentException(
                    "Multiple objects defined with id '" + str(id) + "' for "
                    "the category '" + str(category) + "'.")

            definitions[category][id] = obj



def construct(definitions, register, arg):
    """
    Starts the parsing process of an object defined via command line. Creates
    the object and all objects it contains. All objects with a defined id
    parameter will be cached for access in definitions.

    :param definitions: map of objects previously created and registered via id
    :param register: map between cmd line class names and class references
    :param arg: argument to parse
    :return: constructed object
    """
    tree = parse_tree(arg)
    if tree.size() > 1:
        raise ArgumentException("Invalid syntax. A parse tree root has multiple"
                                "children: " + arg)

    item_tree = tree.first_child

    # check if object is already globally defined
    item_obj = find_object_in_global_definitions(item_tree, definitions,
                                                 register)
    if item_obj is not None:
        return item_obj


    #not globally defined
    item_type = item_tree.data[0].lower()
    item_key = item_tree.data[1]

    if item_type not in register:
        raise ArgumentException("The object to construct is of unknown type: "
                                + str(item_type))

    item_obj = register[item_type].parse(item_tree, definitions)

    if item_key != "":
        if item_key in definitions["global"]:
            raise ArgumentException("Multiple definitions for objects of with "
                                    "the name: " + item_key)

        definitions["global"][item_key] = item_obj


def map_object_parameters(tree, class_arguments):
    parameters = class_arguments.order
    args = {}
    named = False

    idx_child = 0
    for child in tree.children:
        if idx_child >= len(parameters):
            raise ArgumentException("More arguments defined for the object than"
                                    " the object possesses arguments.")

        (_, key) = child.data
        if key != "":
            if key in args:
                raise ArgumentException("Defined multiple times the same "
                                        "parameter of a single object" + key)
            if key not in parameters:
                raise ArgumentException("Using unknown key: " + key)

            args[key] = child
            named = True
        else:
            if named:
                raise ArgumentException("After a keyed parameter assignment"
                                        " (key=value), all succeeding "
                                        "parameters of the object have to "
                                        "be keyed, too.")
            if parameters[idx_child] in args:
                raise ArgumentException("Defined multiple times the same "
                                        "parameter of a single object" + key)
            args[parameters[idx_child]] = child
        idx_child += 1

    for parameter in parameters:
        if parameter not in args:
            args[parameter] = None

    return args


def are_all_none_except(dictionary, *args):
    for arg in args:
        if arg not in dictionary:
            return False
        if dictionary[arg] is None:
            return False

    for key in dictionary:
        if dictionary[key] is not None:
            return False

    return True


def try_lookup_obj(tree, definitions, base_clazz, spec_clazz):

    if tree.size() == 1 and tree.first_child.data[1] == "id":
        id = tree.first_child.data[0]

        if id not in definitions[base_clazz]:
            return None
            # It may occure that we define objects with solely an id.
            # raise ArgumentException("The item of id '" + str(id) + "'"
            #                        " to look up does not exist for the "
            #                        "category '" + str(base_clazz) + "'")
        obj = definitions[base_clazz][id]
        if spec_clazz is not None and type(obj) != spec_clazz:
            raise ArgumentException("The item (id=" + str(id) + ") "
                                    "looked up is of the wrong class ("
                                    + str(type(obj)) + " instead of "
                                    + str(spec_clazz) + ").")
        else:
            return obj
    else:
        return None


def try_construct_from_tree_and_class_arguments(clazz, tree, definitions):
    args = map_object_parameters(tree, clazz.arguments)

    call = {}
    for name in clazz.arguments.order:
        call[name] = clazz.arguments.parse(name, args[name], definitions)

    obj = clazz(**call)

    return obj


def try_whole_obj_parse_process(tree, definitions,
                                base_clazz, spec_clazz,
                                instantiation_clazz=None):

    if instantiation_clazz is None:
        instantiation_clazz = spec_clazz

    obj = try_lookup_obj(tree, definitions, base_clazz, spec_clazz)
    if obj is not None:
        return obj

    if instantiation_clazz is None:
        raise ValueError("Cannot parse object, because the concrete class "
                         "reference is missing")

    obj = try_construct_from_tree_and_class_arguments(instantiation_clazz, tree,
                                                      definitions)

    try:
        id = obj.id
        if id is not None:
            append_definitions(definitions, id, obj)
    except AttributeError:
        pass

    return obj
