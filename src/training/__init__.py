# HACK After preloading dependencies via imp.load_source (Py2), the first
# conventional loading of dependencies via from . import fails. Whereas,
# import dependencies works always on python2, on python3 the statement fails,
# This way, it works for both python versions with and without preloading
#
try:
    from . import dependencies
except ImportError:
    import dependencies

# parsing related modules
from .tree import TreeNode
from .tree_parser import parse_tree

from .parser_tools import main_register
from . import parser_tools
from . import parser

# independent modules
from .data import SizeBatchData, SampleBatchData
from .message import Message
from .misc import AbstractBaseClass, InvalidModuleImplementation



# modules defining the objects for the user
from .variable import Variable, vregister

from . import bridges
from . import conditions
from . import environments
from . import networks
from . import problem_sorter
from . import samplers
from . import networks
from . import training_schemas as schemas
from . import variable



from .main import main
