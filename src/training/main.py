from .parser_tools import ArgumentException, ItemCache
from .parser import construct

from . import conditions
from . import environments
from . import networks
from . import main_register
from . import parser
from . import problem_sorter
from . import samplers
from . import training_schemas
from . import variable

import sys

# maps the command line arguments for defining objects (e.g. -var) to the
# settings to construct them. {cmd arg : clazz}
predefinition_construction_settings = {}


def add_predefinitions(clazz, *args):
    """
    Adds settings to predefinition_construction_settings
    :param clazz: Clazz of the object which will result from the construction
    :param args: keys under which the clazz construction will be reachable
    :return:
    """
    for key in args:
        if key in predefinition_construction_settings:
            raise KeyError("The key " + str(key) + " used to define an object"
                           " is used multiple times (at least to define "
                           "objects of types "
                           + str(predefinition_construction_settings[key].__name__)
                           + " and " + str(clazz.__name__)) + ")."
        else:
            predefinition_construction_settings[key] = clazz


add_predefinitions(variable.Variable, "-v", "-var", "-variable")
add_predefinitions(conditions.Condition, "-c", "-cnd", "-condition")
add_predefinitions(problem_sorter.ProblemSorter, "-sort", "-sorter",
                   "-problem_sorter")
add_predefinitions(samplers.Sampler, "-sampler", "-samp")
add_predefinitions(networks.Network, "-network", "-net")
add_predefinitions(training_schemas.Schema, "-schema")
add_predefinitions(environments.Environment, "-environment", "-env")


def construct_from_cmd(item_cache, key, value):
    """
    Constructs an object defined on the command line (or another valid string
    representation)
    :param key: defines the object to construct. e.g. -var
    :param value: description used to construct the object
    :return: object constructed
    """

    clazz = predefinition_construction_settings[key]

    if value is None:
        raise ArgumentException("After " + key + " a further argument"
                                "was expected defining a " + str(clazz))

    return construct(item_cache, main_register.get_register(clazz), value)

def process_buffer(item_cache, buffer):
    if len(buffer) > 0:
        raise ValueError("A new item shall be constructed while some data is "
                         "still buffered. Currently now use case is implemented"
                         " which requires the buffer. Buffer: " + str(buffer))
    buffer.clear()


def register_all_default_environment(item_cache, env):
    def register_default_environment(item):
        if hasattr(item, "_environment"):
            if item._environment is None:
                item._environment = env
    item_cache.apply_on_all(register_default_environment)


def main(argv):
    item_cache = ItemCache()
    templates = {}
    main_schema = None
    environment = None

    buffer = []
    idx_arg = 0

    # Parse Arguments
    while idx_arg < len(argv):
        arg = argv[idx_arg]
        print(templates)
        print(argv)
        print(arg)

        if arg in predefinition_construction_settings:
            process_buffer(item_cache, buffer)

            value = None if len(argv) <= idx_arg + 1 else argv[idx_arg + 1]
            construct_from_cmd(item_cache, arg, value)

            idx_arg += 1

        elif arg == "-default-environment":
            if idx_arg + 1 >= len(argv):
                raise ArgumentException ("After " + arg + " has to follow an"
                                         "environment definition.")
            environment = construct_from_cmd(item_cache, "-environment",
                                             argv[idx_arg + 1])
            idx_arg += 1
        elif arg in ["-template", "-tpl"]:
            if len(argv) < idx_arg +2:
                raise ValueError("After " + arg + " has to follow either "
                                 "'+ key' to insert the argument for template"
                                 "key and then the value for template key,"
                                 "'- key' to insert only the value for template"
                                 " key, or a template file type "
                                 "(currently only 'csv') with a file path has"
                                 "to follow.")

            arg2 = argv[idx_arg + 1]
            arg3 = argv[idx_arg + 2]

            if arg2 in ["+", "-"]:
                if arg3 not in templates:
                    raise KeyError("The template " + arg3 + " is not defined.")
                tpl_arg, tpl_value = templates[arg3]
                ary = tpl_value if arg2 == "-" else ([tpl_arg] + tpl_value)
                argv = argv[:idx_arg] + ary + argv[idx_arg + 3:]
                idx_arg -= 1

            #add other file formats when supported
            elif arg2 in ["csv"]:
                parser.load_csv_templates(arg3, templates)
                idx_arg += 2
            else:
                raise ValueError("Unknown template file type " + arg2)

        # is last entry and has no '-'
        elif idx_arg == len(argv) - 1 and not arg.startswith("-"):
            process_buffer(item_cache, buffer)

            main_schema = construct_from_cmd(item_cache, "-schema", argv[idx_arg])
        else:
            buffer.append(arg)

        idx_arg += 1

    # Set up environment
    if environment is None:
        environment = environments.Environment()
    register_all_default_environment(item_cache, environment)

    if main_schema is None:
        print("No schema defined to run. The last argument given shall be a"
              "schema argument (without using '-schema' in front of it).")
    else:
        main_schema.run()



if __name__ == "__main__":
    main(sys.argv[1:])
