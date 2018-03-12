from .. import parser

register = {}
class_reference_converter = parser.ClassReferenceConverter(register)

from .base_conditions import Condition
from .base_conditions import CTrue, CFalse
from .base_conditions import CAnd, COr, CNot, CXor

from .base_mutators import MAdd, MThreshold, MModulo
from .advanced_mutators import MGroup, MLeft2Right, MRoundRobin

