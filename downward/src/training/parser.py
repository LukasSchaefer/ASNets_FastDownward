from . import parse_tree

from .parser_tools import ArgumentException


def convert_int_or_inf(value):
    try:
        return int(value)
    except ValueError as e:
        if value == "inf":
            return float("inf")
        elif value == "-inf":
            return float("-inf")
        raise e

def convert_bool(value):
    if value in [True, 1, "t", "T", "1", "true", "True"]:
        return True
    elif value in [False, 0, "f", "F", "0", "false", "False"]:
        return False
    else:
        ValueError("Unable to parse input value to boolean, please use 'True' "
                   "or 'False' as values.")


"""
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
"""


def construct(item_cache, register, arg):
    """
    Starts the parsing process of an object defined via command line. Creates
    the object and all objects it contains. All objects with a defined id
    parameter will be cached for access in definitions.

    :param item_cache: cache of objects previously created and registered via id
    :param register: mapper between cmd line class names and class references
    :param arg: argument to parse
    :return: constructed object
    """
    tree = parse_tree(arg)
    if tree.size() > 1:
        raise ArgumentException("Invalid syntax. A parse tree root has multiple"
                                "children: " + arg)

    item_tree = tree.first_child

    # check if object is already globally defined
    item_obj = item_cache.get_from_empty_tree(None, item_tree, register, True)

    if item_obj is not None:
        return item_obj

    # not globally defined
    item_type = item_tree.data[0].lower()
    item_key = item_tree.data[1]

    if not register.has_key(item_type):
        raise ArgumentException("The object to construct is of unknown type: "
                                + str(item_type))

    item_obj = register.get_reference(item_type).parse(item_tree, item_cache)

    if item_key != "":
        if item_cache.has(None, item_key):
            raise ArgumentException("Multiple definitions for objects of with "
                                    "the name: " + item_key)

        else:
            item_cache.add(item_key, item_obj, glob=True)

    return item_obj


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


def try_lookup_obj(tree, item_cache, clazz, register=None):

    if tree.size() == 1 and tree.first_child.data[1] == "id":
        id = tree.first_child.data[0]

        obj = item_cache.get(clazz, id, register, True)
        return obj

    else:
        return None


def try_construct_from_tree_and_class_arguments(clazz, tree, item_cache):
    args = map_object_parameters(tree, clazz.arguments)

    call = {}
    for name in clazz.arguments.order:
        call[name] = clazz.arguments.parse(name, args[name], item_cache)

    obj = clazz(**call)

    return obj


def try_whole_obj_parse_process(tree, item_cache,
                                clazz, instantiate=True):

    obj = try_lookup_obj(tree, item_cache, clazz)
    if obj is not None:
        return obj

    # if clazz is also None, can this happen?
    # if instantiation_clazz is None:
    #    raise ValueError("Cannot parse object, because the concrete class "
    #                     "reference is missing")

    if instantiate:
        obj = try_construct_from_tree_and_class_arguments(clazz, tree,
                                                          item_cache)

        try:
            id = obj.id
            if id is not None:
                item_cache.add(id, obj, False)

        except AttributeError:
            pass

        return obj

    else:
        return None



def load_csv_templates(path, templates):
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or line == "":
                continue

            parts = line.split(";")
            if len(parts) < 3:
                raise ValueError("Invalid template in CSV template " + path
                                 + ". CSV tempaltes have the format:\n"
                                 "KEY;(optional)ARG KEY (e.g. -variable); X; Y;"
                                   " ... . The KEY defines later the template."
                                   "If a template uses an existing key, it"
                                   "overwrites the old template of that key."
                                   "ARG KEY defines which argument the template"
                                   "belongs to. If the template is included with"
                                   "the argument key, then it inserts '-ARG KEY'"
                                   "in the command line. Everything after"
                                   "(X,Y,...) is always inserted into the"
                                   "argument list. Each entry separated by ';'"
                                   "is inserted as an own element.")

            key = parts[0]
            arg = parts[1]
            items = parts[2:]
            for i in range(len(items)):
                items[i] = items[i]
            templates[key] = (arg, items)