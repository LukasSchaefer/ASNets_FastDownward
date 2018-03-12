from .parser import ArgumentException, construct
from . import conditions
from . import networks
from . import problem_sorter
from . import samplers
from . import training_schemas
from . import tree_parser
from . import variable


import sys




def main(argv):
    definitions = {}
    definitions[variable.Variable] = {}
    definitions[conditions.Condition] = {}
    definitions[networks.Network] = {}
    definitions[problem_sorter.ProblemSorter] = {}
    definitions[samplers.Sampler] = {}
    definitions[training_schemas.Schema] = {}
    definitions["global"] = {}


    main_schema = None
    buffer = []
    idx_arg = 0
    while idx_arg < len(argv):
        arg = argv[idx_arg]

        if arg.startswith("-"):
            #process what is still in the buffer
            pass
        if arg in ["-v", "-var", "-variable"]:
            if len(argv) <= idx_arg + 1:
                raise ArgumentException("After " + arg + " a further argument"
                                        "was expected defining a variable.")
            construct(definitions, variable.register,
                      argv[idx_arg + 1])
            idx_arg += 1

        elif arg in ["-c", "-cnd", "-condition"]:
            if len(argv) <= idx_arg + 1:
                raise ArgumentException("After " + arg + " a further argument"
                                        "was expected defining a condition.")
            construct(definitions, conditions.register,
                      argv[idx_arg + 1])
            idx_arg += 1

        idx_arg += 1

    print("Definitions")
    for cat in definitions:
        print("\t", str(cat))
        for id in definitions[cat]:
            obj = definitions[cat][id]
            print("\t\t", id , ' - ', obj)

    print(definitions['global']['tst'].satisfied())
    print(definitions['global']['tst'].satisfied())
    print(definitions['global']['tst'].satisfied())
    print(definitions['global']['tst'].satisfied())
    print(definitions['global']['tst'].satisfied())
    print(definitions['global']['tst'].satisfied())

    print(definitions['global']['tst'].id)



if __name__ == "__main__":
    main(sys.argv[1:])