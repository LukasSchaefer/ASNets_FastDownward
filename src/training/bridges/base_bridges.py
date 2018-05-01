from .. import main_register
from .. import parser
from .. import parser_tools as parset

from ..parser_tools import ArgumentException
from ..environments import Environment

import abc
from future.utils import with_metaclass
import os

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




class SamplerBridge(Bridge):
    """
    Superclass for all bridges to define access to sampling techniques for
    the sampler classes.
    """
    arguments = parset.ClassArguments('SamplerBridge', Bridge.arguments,
                                      ("tmp_dir", True, None, str),
                                      ('target_file', True, None, str),
                                      ('target_dir', True, None, str),
                                      ('append', True, False, parser.convert_bool),
                                      ('domain', True, None, str),
                                      ("makedir", True, False, parser.convert_bool),
                                      ("environment", True, None, main_register.get_register(Environment)),
                                      order=["tmp_dir", "target_file",
                                             "target_dir", "append", "domain",
                                             "makedir", "environment", "id"])

    def __init__(self, tmp_dir=None, target_file=None, target_dir=None,
                 append=False, domain=None, makedir=False,
                 environment=None, id=None):
        Bridge.__init__(self, id)
        self._tmp_dir = tmp_dir
        self._target_file = target_file
        self._target_dir = target_dir
        self._append = append
        self._domain = domain
        self._makedir = makedir
        self._environment = environment

        if self._target_dir is not None and self._target_file is not None:
            raise ValueError("Either a directory in which the sampled data can"
                             "be stored shall be given OR a file in which all"
                             "samples shall be stored, but NOT BOTH.")



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


    def _get_path_sample(self, path_samples, path_problem=None):
        if path_samples  is not None:
            return path_samples

        return (self._target_file if self._target_file is not None
                else ((os.path.splitext(path_problem)[0] + ".data")
                  if self._target_dir is None
                  else os.path.join(self._target_dir,
                                    os.path.splitext(
                                        os.path.basename(path_problem))[0]
                                    + ".data")))

    def initialize(self):
        self._initialize()

    @abc.abstractmethod
    def _initialize(self):
        pass

    def sample(self, path_problem, path_samples=None, path_dir_tmp=None,
               path_domain=None, append=None):
        """
        Starts a sampling run.
        :param path_problem: Path to the original problem to sample from
        :param path_samples: Path to store the sampled entries
        :param path_dir_tmp: Path to temporary storage directory (required if samples
                         shall be appended to sample_file). If neither given
                         here nor for the bridge construction, then the problem
                         directory will be used.
        :param path_domain: path to domain file. If not given automatically
                            looked for.
        :param append: Write sample_file via appending or overwritting
        :return:
        """


        append = self._append if append is None else append
        path_domain = self._domain if path_domain is None else path_domain
        path_dir_tmp = self._get_tmp_dir(path_dir_tmp, path_problem)
        path_samples = self._get_path_sample(path_samples, path_problem)

        self._sample(path_problem, path_samples, path_dir_tmp, path_domain,
                     append)

    @abc.abstractmethod
    def _sample(self, path_problem, path_samples, path_dir_tmp,
                path_domain, append):
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
