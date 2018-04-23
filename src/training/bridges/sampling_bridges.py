from . import FileSamplerBridge

from .. import main_register
from .. import parser
from .. import parser_tools as parset


from ..environments import Environment, SubprocessTask

import gzip
import os
import subprocess
import threading

class SampleFormat(object):
    name2obj = {}
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self._add_to_enum()

    def _add_to_enum(self):
        setattr(SampleFormat, self.name, self)
        SampleFormat.name2obj[self.name] = self

    def get(name):
        if name not in SampleFormat.name2obj:
            raise ValueError("Unkown key for SampleFormat: " + str(name))
        return SampleFormat.name2obj[name]

SampleFormat("FD", "Only reachable and not static predicates (determined by "
                   "translator module) which are true are"
                   "stored (like FastDownwards PDDL format). Atoms are "
                   "alphabetically sorted.")
SampleFormat("FDAll", "Only reachable and not static predicates (determined by"
                      "translator module) are stored. If an atom is true, then"
                      "its name is suffixed with \"+\" otherwise with \"-\"."
                      " Atoms are alphabetically sorted.")
SampleFormat("FDShort", "Like FDAll, but the atom names are skipped. As their"
                        "order is alphabetical you can retain it.")

SampleFormat("Full", "All atoms are given in alphabetical order annotated with"
                     "\"+\" if true in the state and \"-\" otherwise.")
SampleFormat("FullShort", "Like Full, but the atom names are skipped. As their"
                        "order is alphabetical you can retain it.")



class FastDownwardSamplerBridge(FileSamplerBridge):
    arguments = parset.ClassArguments('FDSamplerBridge', FileSamplerBridge.arguments,
                                      ("search", False, None, str),
                                      ("format", True, SampleFormat.FD, SampleFormat.get),
                                      ("debug", True, False, parser.convert_bool),
                                      ("tmp_dir", True, ".", str),
                                      ("makedir", True, False, parser.convert_bool),
                                      ("fd_path", True, "None", str),
                                      ("compress", True, True, parser.convert_bool),
                                      order=["search", "format", "debug",
                                             "tmp_dir", "makedir", "fd_path",
                                             "compress", "environment", "id"]
                                      )

    def __init__(self, search, format=SampleFormat.FD, debug=False, tmp_dir=".",
                 makedir=False, fd_path=None, compress=True,
                 environment=None, id=None):
        FileSamplerBridge.__init__(self, environment, id)

        self._search = search
        self._format = format
        self._debug = debug
        self._tmp_dir = tmp_dir
        self._makedir = makedir
        self._compress = compress
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

    def _convert(self, path_problem, path_sample):
        pass
    def _sample(self, path_problem, prefix):
        path_samples = os.path.join(self._tmp_dir, prefix + "_samples.data")
        cmd = [self._fd_path, "--plan-file",
               path_samples,
               path_problem, "--search", self._search]
        if self._debug:
            cmd.insert(1, "--debug")


        spt = SubprocessTask("Sampling of " + path_problem + "with prefix " + prefix,
                             cmd)

        # TODO Add environment again
        #self._environment.queue_push(spt)
        #event.wait()
        spt.run()
        self._convert(path_problem, path_samples)








    def _finalize(self):
        pass

    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  FastDownwardSamplerBridge)


main_register.append_register(FastDownwardSamplerBridge, "fdbridge")
