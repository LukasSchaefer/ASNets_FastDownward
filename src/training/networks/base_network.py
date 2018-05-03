from .. import parser
from .. import parser_tools as parset

from ..parser_tools import main_register, ArgumentException
from ..variable import Variable

import abc
from future.utils import with_metaclass


class InvalidMethodCallException(Exception):
    pass



class NetworkFormat(object):
    name2obj = {}
    suffix2obj = {}

    def __init__(self, name, suffix, description):
        self.name = name
        self.suffix = suffix
        self.description = description
        self._add_to_enum()

    def _add_to_enum(self):
        setattr(NetworkFormat, self.name, self)
        NetworkFormat.name2obj[self.name] = self
        NetworkFormat.suffix[self.suffix] = self

    @staticmethod
    def _get(name, map):
        if name not in map:
            raise ValueError("Unknown key: " + str(name))
        return map[name]

    @staticmethod
    def by_suffix(suffix):
        return NetworkFormat._get(suffix, NetworkFormat.suffix2obj)

    @staticmethod
    def by_name(name):
        return NetworkFormat._get(name, NetworkFormat.name2obj)

    def __str__(self):
        return self.name


NetworkFormat("Protobuf", "pb", "Protobuf Format")
NetworkFormat("hdf5", "h5", "hdf5 format (e.g. used by Keras)")


class Network(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all neural networks.
    Do not forget to register your network subclass in this packages 'register'
    dictionary via 'append_register' of the main package.
    """

    arguments = parset.ClassArguments("Network", None,
                                      ('store', True, None, str),
                                      ('variables', True, {},
                                       main_register.get_register(Variable)),
                                      ('id', True, None, str),
                                      )

    def __init__(self, store=None, variables={}, id=None):
        if not isinstance(variables, dict):
            raise ArgumentException("The provided variables have to be a map. "
                                    "Please define them as {name=VARIABLE,...}.")
        self.path_store = store
        self.variables = variables
        self.id = id

        self.initialized = False
        self.finalized = False

    def initialize(self):
        if not self.initialized:
            self._initialize()
            self.initialized = True
        else:
            raise InvalidMethodCallException("Multiple initializations of"
                                             "network.")


    def finalize(self):
        if not self.finalized:
            self._finalize()
            if self.path_store is not None:
                self.store()
            self.finalized = True
        else:
            raise InvalidMethodCallException("Multiple finalization calls of"
                                             "network.")

    def store(self):
        if self.path_store is not None:
            self._store()

    @abc.abstractmethod
    def _initialize(self):
        pass

    @abc.abstractmethod
    def train(self, msgs, format, data):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def _finalize(self):
        pass

    @abc.abstractmethod
    def _store(self):
        """
        Please notice _store can be used before the network is finalized and
        after. The intension is that if the network is not finalized, then
        _store shall behave as storing intermedate values (or none, if this
        is not desired) and if the network is finalized, then _store shall do
        the final storing of the network.
        :return:
        """
        pass

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, Network, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base network can "
                                    "only be used for look up of any previously"
                                    " defined schema via 'Sampler(id=ID)'")


main_register.append_register(Network, "network")
nregister = main_register.get_register(Network)

