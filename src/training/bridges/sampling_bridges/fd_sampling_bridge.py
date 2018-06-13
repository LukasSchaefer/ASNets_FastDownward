from . import SamplerBridge
from .common import StateFormat, context_load, load_and_convert_data

from ... import main_register
from ... import parser
from ... import parser_tools as parset
from ... import SampleBatchData

from ...environments import Environment, SubprocessTask
from ...misc import hasher

import os
import random



class FastDownwardSamplerBridge(SamplerBridge):
    arguments = parset.ClassArguments('FastDownwardSamplerBridge',
        SamplerBridge.arguments,
        ("search", False, None, str, "Search argument for Fast-Downward"),
        ("format", True, StateFormat.FD, StateFormat.get,
         "Format to represent the sampled state"),
        ("build", True, "debug64dynamic", str, "Build of Fast-Downward to use"),
        ("fd_path", True, "None", str, "Path to the fast-downward.py script"),
        ("prune", True, True, parser.convert_bool, "Prune duplicate samples"),

        order=["search","streams", "format", "build",
             "tmp_dir", "provide", "forget", "domain",
             "makedir", "fd_path", "prune",
             "environment", "id"]
)

    def __init__(self, search, streams=None, format=StateFormat.FD,
                 build="debug64dynamic",
                 tmp_dir=None, provide=True, forget=0.0,
                 domain=None,
                 makedir=False, fd_path=None, prune=True,
                 environment=None, id=None):
        SamplerBridge.__init__(self, tmp_dir, streams, provide, forget,
                               domain, makedir, environment, id)

        self._search = search
        self._format = format
        self._build = build
        self._prune = prune

        if fd_path is None or fd_path == "None":
            fd_path = "."
        if not os.path.isfile(fd_path):
            possible_path = os.path.join(fd_path, "fast-downward.py")
            if not os.path.isfile(possible_path):
                raise parset.ArgumentException("Unable to find fast downward"
                                               "script in/as " + str(fd_path)
                                               + ".")
            else:
                fd_path = possible_path
        self._fd_path = fd_path

    def _initialize(self):
        pass


    def _sample(self, path_problem, path_dir_tmp, path_domain, data_container):
        path_tmp_samples = os.path.join(path_dir_tmp,
                                        os.path.basename(path_problem)
                                        + "." + str(random.randint(0,9999))
                                        + ".tmp")

        data_container = (SampleBatchData(5, [self._format, self._format, str,
                                              self._format, int], 0, 1, 3, 2, 4,
                                          path_problem,
                                          pruning=(hasher.list_hasher if self._prune else None))

                          if data_container is None else data_container)

        if not self._streams.may_reuse(path_problem):

            cmd = [self._fd_path,
                   "--plan-file", "\\real_case\\" + path_tmp_samples + "\\lower_case\\",
                   "--build", self._build,
                   path_problem, "--search", self._search]
            if path_domain is not None:
                cmd.insert(6, path_domain)
            spt = SubprocessTask("Sampling of " + path_problem, cmd)

            # TODO Add environment again
            #self._environment.queue_push(spt)
            #event.wait()
            spt.run()
            load_and_convert_data(
                path_read=path_tmp_samples,
                format=self._format, prune=self._prune,
                path_problem=path_problem, path_domain=path_domain,
                data_container=data_container,
                delete=True, forget=self._forget, write_context=self._streams)

        else:
            if self._provide:
                context_load(self._streams, data_container,
                             format=self._format, prune=self._prune,
                             path_problem=path_problem, path_domain=path_domain,
                             skip=self._skip,
                             forget=self._forget)

        return data_container


    def _finalize(self):
        pass

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  FastDownwardSamplerBridge)


main_register.append_register(FastDownwardSamplerBridge, "fdbridge")