from .parser_tools import ArgumentException, ItemCache
from .parser import construct

from . import conditions
from . import networks
from . import main_register
from . import problem_sorter
from . import samplers
from . import training_schemas
from . import variable

import sys




def main(argv):
    item_cache = ItemCache()

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
            construct(item_cache, main_register.get_register(variable.Variable),
                      argv[idx_arg + 1])
            idx_arg += 1

        elif arg in ["-c", "-cnd", "-condition"]:
            if len(argv) <= idx_arg + 1:
                raise ArgumentException("After " + arg + " a further argument"
                                        "was expected defining a condition.")
            construct(item_cache, main_register.get_register(conditions.Condition),
                      argv[idx_arg + 1])
            idx_arg += 1

        idx_arg += 1

    print("Definitions")
    print(item_cache.to_string())

    print(item_cache.get(None, 'v0').value)
    print(item_cache.get(None, 'v2').value)

    print(item_cache.get(None, 'tst').next())
    print(item_cache.get(None, 'v0').value)
    print(item_cache.get(None, 'v2').value)

    print(item_cache.get(None, 'tst').next())
    print(item_cache.get(None, 'v0').value)
    print(item_cache.get(None, 'v2').value)

    print(item_cache.get(None, 'tst').next())
    print(item_cache.get(None, 'v0').value)
    print(item_cache.get(None, 'v2').value)

    print(item_cache.get(None, 'tst').next())
    print(item_cache.get(None, 'v0').value)
    print(item_cache.get(None, 'v2').value)

    print(item_cache.get(None, 'tst').next())
    print(item_cache.get(None, 'v0').value)
    print(item_cache.get(None, 'v2').value)

    print(item_cache.get(None, 'tst').next())
    print(item_cache.get(None, 'v0').value)
    print(item_cache.get(None, 'v2').value)

    print(item_cache.get(None, 'tst').id)



if __name__ == "__main__":
    main(sys.argv[1:])