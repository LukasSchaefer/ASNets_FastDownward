from ..base_bridges import Bridge

from ... import main_register
from ... import parser
from ... import parser_tools as parset
from ... import InvalidModuleImplementation

from ...parser_tools import ArgumentException
from ...environments import Environment
from ...misc import StreamContext

import abc
import os

class SamplerBridge(Bridge):
    """
    Superclass for all bridges to define access to sampling techniques for
    the sampler classes.
    """
    arguments = parset.ClassArguments('SamplerBridge', Bridge.arguments,
        ("tmp_dir", True, None, str,
         "Directory to store temporary data during the sampling"),
        ('streams', True, None, main_register.get_register(StreamContext),
         "StreamContext (usually with StreamDefinitions) for storing the output"),
        ('provide', True, True, parser.convert_bool,
         "Return the sampled data as object"),
        ("forget", True, 0.0, float,
        "Probability to 'forget' to return a sample entry from the data"),
        ('domain', True, None, str),
        ("makedir", True, False, parser.convert_bool),
        ("environment", True, None, main_register.get_register(Environment)),
        order=["tmp_dir", "streams", "provide", "forget", "domain",
               "makedir", "environment", "id"])

    def __init__(self, tmp_dir=None, streams=None, provide=True, forget=0.0,
                 domain=None, makedir=False,
                 environment=None, id=None):
        Bridge.__init__(self, id)
        self._tmp_dir = tmp_dir
        self._streams = StreamContext() if streams is None else streams
        self._provide = provide
        self._forget = forget
        self._domain = domain
        self._makedir = makedir
        self._environment = environment


    def _get_tmp_dir(self, dir, path_problem=None):
        dir = (dir if dir is not None else
               (self._tmp_dir if self._tmp_dir is not None else os.path.dirname(path_problem)))

        if not os.path.isdir(dir):
            if self._makedir:
                os.makedirs(dir)
            else:
                raise parset.ArgumentException("The required temporary directory"
                                               "is missing and permission "
                                               "for its creation was not "
                                               "granted: " + str(dir))
        return dir

    """
    def _get_path_sample(self, path_samples, path_problem=None):
        if path_samples is not None:
            return path_samples

        return (self._target_file if self._target_file is not None
                else ((os.path.splitext(path_problem)[0] + ".data")
                      if self._target_dir is None
                      else os.path.join(self._target_dir,
                                        os.path.splitext(
                                            os.path.basename(path_problem))[0]
                                        + ".data")))
    """
    def initialize(self):
        self._initialize()

    @abc.abstractmethod
    def _initialize(self):
        pass

    def sample(self, path_problem, path_dir_tmp=None,
               path_domain=None, data_container=None):
        """
        Starts a sampling run.
        :param path_problem: Path to the original problem to sample from
        :param path_dir_tmp: Path to temporary storage directory (required if samples
                         shall be appended to sample_file). If neither given
                         here nor for the bridge construction, then the problem
                         directory will be used.
        :param path_domain: path to domain file. If not given automatically
                            looked for.
        :param append: Write sample_file via appending or overwritting
        :param data_container: Container with and add_entry(entry, type) method
                               in which the sampled entries shall be added.
        :return: A container containing the sampled entries. If data_container
                 was given, then data_container shall be returned.
        """

        #path_samples = self._get_path_sample(path_samples, path_problem)
        path_dir_tmp = self._get_tmp_dir(path_dir_tmp, path_problem)
        path_domain = self._domain if path_domain is None else path_domain

        data = self._sample(path_problem, path_dir_tmp,
                            path_domain, data_container)
        if data is None and self._provide:
            raise InvalidModuleImplementation("A SamplingBridge should have"
                                              "provided a data container but"
                                              "returned None.")
        return data

    @abc.abstractmethod
    def _sample(self, path_problem, path_dir_tmp,
                path_domain, data_container):
        pass

    def finalize(self):
        self._finalize()

    @abc.abstractmethod
    def _finalize(self):
        pass

    @staticmethod
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
