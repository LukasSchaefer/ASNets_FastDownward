from . import SamplerBridge

from .common import StateFormat, context_load

from ... import main_register
from ... import parser
from ... import parser_tools as parset
from ... import SampleBatchData

from ...misc import hasher

import sys


class LoadSampleBridge(SamplerBridge):
    arguments = parset.ClassArguments('LoadSampleBridge',
        SamplerBridge.arguments,
        ("format", True, StateFormat.FD, StateFormat.get,
         "Format to represent the sampled state"),
        ("prune", True, True, parser.convert_bool, "Prune duplicate samples"),
        ("skip", True, True, parser.convert_bool,"Skip problem if no samples exists, else raise error"),
        ("skip_magic", True, False, parser.convert_bool,
         "Skip magic word check (no guarantees on opening the files with the"
         " right tool (DEPRECATED)"),
        ("provide", True, False, parser.convert_bool),
        order=["streams", "format", "prune", "forget", "skip",
             "tmp_dir", "provide", "domain",
             "makedir", "skip_magic",
             "environment", "id"]
)

    def __init__(self, streams=None, format=StateFormat.FD, prune=True, forget=0.0, skip=True,
                 tmp_dir=None, provide=False, domain=None,
                 makedir=False, skip_magic=False, environment=None, id=None):
        SamplerBridge.__init__(self, tmp_dir, streams, provide, forget,
                               domain, makedir, environment, id)

        self._format = format
        self._prune = prune
        self._skip = skip
        self._skip_magic = skip_magic
        if self._provide:
            print("The 'provide' parameter has no effect on the LoadSampleBride.", file=sys.stderr)

    def _initialize(self):
        pass


    def _sample(self, path_problem, path_dir_tmp, path_domain, data_container):

        data_container = (SampleBatchData(5, [self._format, self._format, str,
                                              self._format, int], 0, 1, 3, 2, 4,
                                          path_problem,
                                          pruning=(hasher.list_hasher if self._prune else None))
                          if data_container is None else data_container)

        context_load(self._streams, data_container,
                     format=self._format, prune=self._prune,
                     path_problem=path_problem, path_domain=path_domain,
                     skip=self._skip, skip_magic=self._skip_magic,
                     forget=self._forget)

        return data_container

    def _finalize(self):
        pass

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  LoadSampleBridge)


main_register.append_register(LoadSampleBridge, "loadbridge")