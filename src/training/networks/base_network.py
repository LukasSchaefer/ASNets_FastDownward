from .. import parser
from .. import parser_tools as parset
from .. import AbstractBaseClass

from ..parser_tools import main_register, ArgumentException
from ..variable import Variable

import abc
import os


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
        NetworkFormat.suffix2obj[self.suffix] = self

    def __str__(self):
        s = self.name + "("
        for suffix in self.suffix:
            s += suffix + ", "
        s = s[:-1] + ")"
        return s

    @staticmethod
    def _get(name, map):
        if name not in map:
            raise ValueError("Unknown key for NetworkFormat: " + str(name))
        return map[name]

    @staticmethod
    def by_suffix(suffix):
        return NetworkFormat._get(suffix, NetworkFormat.suffix2obj)

    @staticmethod
    def by_name(name):
        return NetworkFormat._get(name, NetworkFormat.name2obj)

    @staticmethod
    def by_any(key):
        if key in NetworkFormat.name2obj:
            return NetworkFormat.name2obj[key]
        if key in NetworkFormat.suffix2obj:
            return NetworkFormat.suffix2obj[key]
        raise KeyError("Unknown key to identify NetworkFormat: " + key)



NetworkFormat("protobuf", "pb", "Protobuf Format")
NetworkFormat("hdf5", "h5", "hdf5 format (e.g. used by Keras)")


class Network(AbstractBaseClass):
    """
    Base class for all neural networks.
    Do not forget to register your network subclass in this packages 'register'
    dictionary via 'append_register' of the main package.
    """

    arguments = parset.ClassArguments("Network", None,
                                      ('load', True, None, str, "File to load network from"),
                                      ('store', True, None, str, "Path (w/o suffix) where to store network"),
                                      ('formats', True, None, NetworkFormat.by_any, "Single or list of formats in which to save network"),
                                      ('out', True, '.', str, "Path to directory for network outputs"),
                                      ('variables', True, None,
                                       main_register.get_register(Variable)),
                                      ('id', True, None, str),
                                      )

    def __init__(self, load=None, store=None, formats=None, out=".",
                 variables=None, id=None):
        variables = {} if variables is None else variables
        if not isinstance(variables, dict):
            raise ArgumentException("The provided variables have to be a map. "
                                    "Please define them as {name=VARIABLE,...}.")
        self.path_load = load
        self.path_store = store
        self.path_out = out
        self.formats = formats if isinstance(formats, list) else [formats]
        self._check_store_formats()

        self.msgs = None
        self.variables = {} if variables is None else variables
        self.id = id

        self.initialized = False
        self.finalized = False

    @abc.abstractmethod
    def _get_default_network_format(self):
        """
        Return default/prefered format to store networks of this class.
        :return: NetworkFormat
        """
        pass

    @abc.abstractmethod
    def _get_store_formats(self):
        """
        Return iterable of all formats in which networks of this class can be
        stored,
        :return: iterable (with "in" operator) of NetworkFormat objects
        """
        pass

    @abc.abstractmethod
    def _get_load_formats(self):
        """
        Return iterable of all formats from which networks of this class can be
        loaded
        :return: iterable (with "in" operator) of NetworkFormat objecs
        """
        pass

    def get_preferred_state_formats(self):
        """
        Most networks work on sampled PDDL states which can be represented in
        different formats. This method returns the supported StateFormats of
        the network in order of preference.
        If your network does not work on those states, let the method raise an
        exception (like it currently does)
        :return: list of supported StateFormat objects (see SamplingBridges)
        """
        raise InvalidMethodCallException("The network does not support "
                                                "state formats.")

    def initialize(self, msgs, *args, **kwargs):
        """
        Build network object, load model (if requested) and prepare
        :param msgs: Message object for communication between objects (if given)
        :return:
        """
        if self.initialized:
            raise InvalidMethodCallException("Multiple initializations of"
                                             "network.")
        self.msgs = msgs

        self._initialize_general(*args, **kwargs)
        if self.path_load is not None:
            self.load(**kwargs)
        else:
            self._initialize_model(*args, **kwargs)

        self.initialized = True

    @abc.abstractmethod
    def _initialize_general(self, *args, **kwargs):
        """Initialization code except for the model initialization"""
        pass

    @abc.abstractmethod
    def _initialize_model(self, *args, **kwargs):
        """Initialization of the network model (if it is not loaded!)"""
        pass

    def finalize(self, *args, **kwargs):
        if not self.finalized:
            self._finalize(*args, **kwargs)
            if self.path_store is not None:
                self.store()
            self.finalized = True
        else:
            raise InvalidMethodCallException("Multiple finalization calls of"
                                             "network.")

    @abc.abstractmethod
    def _finalize(self):
        pass

    def load(self, path=None, format=None, **kwargs):
        """
        Load network from a file
        :param path: If given, file to load network from, else the path given
                     at construction time is used. If this is also not given,
                     a ValueError is raised.
        :param format: Network format of the file to load. If not given, it is
                       infered from the files suffix if possible.
        :return:
        """
        path = self.path_load if path is None else path
        if path is None:
            raise ValueError("No path defined for loading of a network")
        if not os.path.exists(path):
            raise ValueError("File does not exists to load network form:"
                             + str(path))
        if format is None:
            suffix = os.path.splitext(path)[1][1:]
            format = NetworkFormat.by_suffix(suffix)
        if format not in self._get_load_formats():
            raise ValueError("The network file to load is not of a format"
                             " supported for loading: " + str(format))
        self._load(path, format, **kwargs)

    @abc.abstractmethod
    def _load(self, path, format, *args, **kwargs):
        pass

    def store(self, path=None, formats=None):
        """
        Stores the network in the specified formats.
        :param path: Path without suffix where to store the network. If None,
                     the path given at construction is used. If even this is
                     None, the output directory defined at construction is
                     used and if this is none './network' is used.
        :param formats: Iterable of NetworkFormats in which to store the network.
                        If None is given, then the formats given at construction
                        are used. If None were given there, then the default
                        format is used.
        :return:
        """
        path = self.path_store if path is None else path
        path = os.path.join("." if self.path_out is None else self.path_out, "network") if path is None else path
        formats = self.formats if formats is None else self.formats
        self._check_store_formats(formats)
        if self._model is not None:
            self._store(path, formats)
        else:
            raise ValueError("Uninitialized model cannot be saved!")

    @abc.abstractmethod
    def _store(self, path, formats):
        pass

    def _check_store_formats(self, formats=None):
        formats = self.formats if formats is None else formats
        sf = self._get_store_formats()
        for f in formats:
            if not (f is None or f in sf):
                raise ValueError("The network does not support storing "
                                 "a desired format: " + str(f))

    @abc.abstractmethod
    def train(self, data):
        """
        Train the network on the given data
        :param data:
        :return:
        """
        pass

    @abc.abstractmethod
    def evaluate(self, data):
        """

        :param data:
        :return:
        """
        pass

    def analyse(self, directory=None, prefix=""):
        """
        Analyse the network performance.
        This functionality is optional and not every networks supports it.
        :param directory: Path to directory for storing the analysis results.
                          If None is given, the output directory given at
                          construction time is used. If this is also None,
                          the current working directory is used.
        :param prefix: A prefix which shall be added in front of every file name
                       which the analysis is producing. If None is given, the
                       no prefix is used.
        :return:
        """
        directory = self.path_out if directory is None else directory
        directory = "." if directory is None else directory
        self._analyse(directory, prefix)

    def _analyse(self, directory, prefix):
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
