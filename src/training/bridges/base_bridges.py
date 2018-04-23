from .. import main_register
from .. import parser
from .. import parser_tools as parset

from ..parser_tools import ArgumentException
from ..environments import Environment

import abc
from future.utils import with_metaclass


class Bridge(with_metaclass(abc.ABCMeta, object)):
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

    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Bridge, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base bridge can "
                                    "only be used for look up of any previously"
                                    " defined condition via 'Bridge(id=ID)'")


main_register.append_register(Bridge, "bridge")


class SamplerBridge(Bridge):
    """
    Superclass for all bridges to define access to sampling techniques for
    the sampler classes.
    """
    arguments = parset.ClassArguments('SamplerBridge', Bridge.arguments,
                                      ("environment", True, None, main_register.get_register(Environment)),
                                      order=["environment", "id"])

    def __init__(self, environment=None, id=None):
        Bridge.__init__(self, id)
        self._environment = environment

    def initialize(self):
        self._initialize()

    @abc.abstractmethod
    def _initialize(self):
        pass

    def sample(self, problem, prefix=""):
        self._sample(problem, prefix)

    @abc.abstractmethod
    def _sample(self, problem, prefix=""):
        pass

    def finalize(self):
        self._finalize()

    @abc.abstractmethod
    def _finalize(self):
        pass

    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Bridge, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base sampler bridge"
                                    " can only be used for look up of any "
                                    "previously defined condition via "
                                    "'SamplerBridge(id=ID)'")


main_register.append_register(SamplerBridge, "samplerbridge")


class FileSamplerBridge(SamplerBridge):
    """
    Superclass for all bridges which expect their problem input to be a file
    in a file system.
    """
    arguments = parset.ClassArguments('FileSamplerBridge',
                                      SamplerBridge.arguments)

    def __init__(self, environment=None, id=None):
        SamplerBridge.__init__(self, environment, id)


    def sample(self, problem, prefix=""):
        self._sample(problem, prefix)

    @abc.abstractmethod
    def _sample(self, problem, prefix=""):
        pass

    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Bridge, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the file sampler bridge"
                                    " can only be used for look up of any "
                                    "previously defined condition via "
                                    "'FileSamplerBridge(id=ID)'")


main_register.append_register(FileSamplerBridge, "filesamplerbridge",
                              "fsampbridge")