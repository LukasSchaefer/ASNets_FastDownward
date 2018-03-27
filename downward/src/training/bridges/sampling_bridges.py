from . import FileSamplerBridge

from .. import main_register
from .. import parser
from .. import parser_tools as parset

from ..environments import Environment, SubprocessTask

import os
import subprocess
import threading

class FastDownwardSamplerBridge(FileSamplerBridge):
    arguments = parset.ClassArguments('SamplerBridge', FileSamplerBridge.arguments,
                                      ("search", False, None, str)
                                      ("debug", True, False, parser.convert_bool)
                                      ("tmp_dir", True, ".", str),
                                      ("makedir", True, False, parser.convert_bool),
                                      ("fd_path", True, "None", str),
                                      order=["search", "debug", "tmp_dir", "makedir", "fd_path", "environment", "id"]
                                      )

    def __init__(self, search, debug=False, tmp_dir=".", makedir=False, fd_path=None, environment=None, id=None):
        FileSamplerBridge.__init__(self, environment, id)

        self._search = search
        self._debug = debug
        self._tmp_dir = tmp_dir
        self._makedir = makedir
        if not os.path.isdir(self._tmp_dir):
            if self._makedir:
                os.makedirs(self._tmp_dir)
            else:
                raise parset.ArgumentException("The given folder for temporary "
                                               "data is missing and permission "
                                               "for its creation was not "
                                               "granted.")

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

    def _sample(self, problem, prefix):
        cmd = [self._fd_path, "--plan-file",
               os.path.join(self._tmp_dir, prefix + "_plan_file"),
               problem, "--search", self._search]
        if self._debug:
            cmd.insert(1, "--debug")

        event = threading.Event()
        spt = SubprocessTask("Sampling of " + problem + "with prefix " + prefix,
                             cmd, event)

        self._environment.queue_push(spt)
        event.wait()





    def _finalize(self):
        pass

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  FastDownwardSamplerBridge)


main_register.append_register(FastDownwardSamplerBridge, "fdbridge")
