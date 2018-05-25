from .. import dependencies

if dependencies.conditions:
    from .base_conditions import cregister
    from .base_conditions import Condition
    from .base_conditions import CTrue, CFalse
    from .base_conditions import CAnd, COr, CNot, CXor

    from .base_mutators import mregister
    from .base_mutators import MAdd, MThreshold, MModulo
    from .advanced_mutators import MGroup, MLeft2Right, MRoundRobin

