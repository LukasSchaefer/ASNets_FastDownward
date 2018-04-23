from .. import parser
from .. import parser_tools as parset

from ..bridges import SamplerBridge
from ..parser_tools import main_register, ArgumentException
from ..variable import Variable

import abc
from future.utils import with_metaclass
import os


class InvalidMethodCallException(Exception):
    pass


class Sampler(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all sampler.
    Do not forget to register your network subclass in this packages 'register'
    dictionary via 'append_register' of the main package.
    """

    arguments = parset.ClassArguments("Sampler", None,
                                      ("sampler_bridge", False, None,
                                       main_register.get_register(SamplerBridge)),
                                      ('variables', True, {},
                                       main_register.get_register(Variable)),
                                      ('id', True, None, str),
                                      variables=[('sample_calls', 0, int)],
                                      )

    def __init__(self, sampler_bridge, variables={}, id=None):
        if not isinstance(variables, dict):
            raise ArgumentException("The provided variables have to be a map. "
                                    "Please define them as {name=VARIABLE,...}.")
        self.sbridge = sampler_bridge
        self.variables = variables
        self.id = id

        self.var_sample_calls, = Sampler.arguments.validate_and_return_variables(variables)

        self.out_log = None  # Message objects for messages from this sampler
        self.in_logs = None  # Dictionary of message objects for ingoing communication
        self.initialized = False
        self.finalized = False

    def initialize(self, out_log=None, in_logs=None):
        if not self.initialized:
            self.in_logs = in_logs
            self.out_log = out_log
            self.sbridge.initialize()
            self._initialize()
            self.initialized = True
        else:
            raise InvalidMethodCallException("Multiple initializations of"
                                             "sampler.")

    def sample(self):
        if not self.initialized:
            raise InvalidMethodCallException("Cannot call sample without "
                                             "initializing the sampler.")
        if self.var_sample_calls is not None:
            self.var_sample_calls.value += 1
        self._sample()

    def finalize(self):
        if not self.initialized:
            raise InvalidMethodCallException("Cannot call finalize the sampler"
                                             " without initializing first.")
        if not self.finalized:
            self.sbridge.finalize()
            self._finalize()
            self.finalized = True
        else:
            raise InvalidMethodCallException("Mutliple finalization calls of"
                                             "sampler.")

    @abc.abstractmethod
    def _initialize(self):
        pass

    @abc.abstractmethod
    def _sample(self):
        pass

    @abc.abstractmethod
    def _finalize(self):
        pass

    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Sampler, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base sampler can "
                                    "only be used for look up of any previously"
                                    " defined schema via 'Sampler(id=ID)'")


main_register.append_register(Sampler, "sampler")
saregister = main_register.get_register(Sampler)
