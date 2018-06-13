from .. import main_register
from .. import parser
from .. import parser_tools as parset
from .. import AbstractBaseClass

from ..parser_tools import ArgumentException


class Bridge(AbstractBaseClass):
    """
    Base class for all bridges.
    Remember to register your concrete subclasses via one (or multiple) names
    at the dictionary 'register' via 'append_register'.

    Bridges are intended to provided configurable functionalities for other
    objects. For example, multiple different samplers exists. We can differentiate
    for samplers at least two categories: 1. how do they select the next problem
    to sample and 2. which tool are they using for sampling. To prevent creating
    a bunch of subclasses, the sampler subclasses shall only change the first
    category. How the sampling is then actually done (e.g. through calling
    fast downward) is then set via an argument of the sampler.

    Thus, the direct subclasses of Bridge shall describe their use case
    (e.g. SamplerBridge) and they have subclasses which implement the
    functionality. (A C++ private/protected inheritance would be better here)
    """

    arguments = parset.ClassArguments('Bridge', None,
        ('id', True, None, str)
    )


    def __init__(self, id=None):
        self.id = id

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Bridge, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base bridge can "
                                    "only be used for look up of any previously"
                                    " defined condition via 'Bridge(id=ID)'")


main_register.append_register(Bridge, "bridge")



