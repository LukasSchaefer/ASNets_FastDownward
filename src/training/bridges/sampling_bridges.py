from builtins import ValueError
from locale import format

from . import SamplerBridge

from .. import main_register
from .. import parser
from .. import parser_tools as parset


from ..environments import Environment, SubprocessTask

from ... import translate
from ...translate import pddl

import gzip
import os
import random
import sys

gzip_input_converter = lambda x: x.encode()

if sys.version_info[0] == 2:
    gzip_input_converter = lambda x: x


CACHE_STR_GROUNDINGS = {}
def get_cached_groundings(groundable):
    if groundable not in CACHE_STR_GROUNDINGS:
        gnd = groundable.str_grounded_predicates(sort=True)
        gnd = [x for x in gnd if not x.startswith("Negated")]
        CACHE_STR_GROUNDINGS[groundable] = gnd
    return CACHE_STR_GROUNDINGS[groundable]


CACHE_STR_NO_SAS_INITS = {}
def get_cached_no_sas_inits(pddl, sas):
    t = (pddl, sas)
    if t not in CACHE_STR_NO_SAS_INITS:
        init = set([str(x) for x in pddl.init if x.predicate != "="])
        for var_names in sas.variables.value_names:
            for name in var_names:
                if name in init:
                    init.remove(name)

        CACHE_STR_NO_SAS_INITS[t] = init
    return CACHE_STR_NO_SAS_INITS[t]

CACHE_STR_TYPE_OBJECTS_PDDL = {}
def get_cached_type_obj_pddl(pddl):
    if pddl not in CACHE_STR_TYPE_OBJECTS_PDDL:
        s = ""
        objs = {}
        for obj in pddl.objects:
            if not obj.type_name in objs:
                objs[obj.type_name] = set()
            objs[obj.type_name].add(obj.name)
        for type_name in objs:
            s += type_name + "("
            for obj in objs[type_name]:
                s += obj + " "
            s = s[:-1] + ")\t"
        s = s[:-1]
        CACHE_STR_TYPE_OBJECTS_PDDL[pddl] = s
    return CACHE_STR_TYPE_OBJECTS_PDDL[pddl]


class DataCorruptedError(Exception):
    pass


class SampleFormat(object):
    name2obj = {}

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self._add_to_enum()

    def _add_to_enum(self):
        setattr(SampleFormat, self.name, self)
        SampleFormat.name2obj[self.name] = self

    @staticmethod
    def get(name):
        if name not in SampleFormat.name2obj:
            raise ValueError("Unknown key for SampleFormat: " + str(name))
        return SampleFormat.name2obj[name]

    @staticmethod
    def extract_from_meta(meta, default, new_format=None):
        if not meta[0] == "<":
            raise ValueError("Given meta is no meta tag: " + meta)
        key_format = "format="
        idx = meta.find(key_format)
        if idx == -1:
            if default is None:
                raise ValueError(
                    "Unable to determine format for entry: " + meta)
            else:
                return default
        else:
            start_idx = idx + len(key_format)
            end_idx = None
            if meta[start_idx] == "\"":
                start_idx += 1
                end_idx = meta.find("\"", start_idx)
                if end_idx == -1:
                    raise ValueError("Unable to determine end of entry format "
                                     "description: " + meta)
            else:
                end_idx = meta.find(" ", start_idx)
                if end_idx == -1:
                    if meta[-1] != ">":
                        raise ValueError(
                            "Unable to determine end of entry format "
                            "description: " + meta)

            format_name = meta[start_idx:end_idx]

            if format_name not in SampleFormat.name2obj:
                raise ValueError("Extracted format is not existing: ",
                                 format_name)
            else:
                if new_format is not None:
                    meta = meta[:start_idx] + new_format.name + meta[end_idx :]
                    return (SampleFormat.name2obj[format_name], meta)
                return SampleFormat.name2obj[format_name]

    def __str__(self):
        return self.name


SampleFormat("FD", "Only reachable and not static predicates (determined by "
                   "translator module) which are true are"
                   "stored (like FastDownwards PDDL format). Atoms are "
                   "alphabetically sorted.")
SampleFormat("FDFull", "Only reachable and not static predicates (determined by"
                      "translator module) are stored. If an atom is true, then"
                      "its name is suffixed with \"+\" otherwise with \"-\"."
                      " Atoms are alphabetically sorted.")
SampleFormat("FDFullShort", "Like FDAll, but the atom names are skipped. As their"
                        "order is alphabetical you can retain it.")

SampleFormat("Full", "All atoms are given in alphabetical order annotated with"
                     "\"+\" if true in the state and \"-\" otherwise.")
SampleFormat("FullShort", "Like Full, but the atom names are skipped. As their"
                          "order is alphabetical you can retain it.")
SampleFormat("Objects", "Adds of the problem to the meta tag and then provides"
                        "only the present grounded predicates.")


def parse_state_present_atoms(state, as_string=False):
    """
    Parse state formats which name the present atoms only (e.g. FD)
    :param state: Atom a(...)\tAtom b(...)...
    :param as_string: True -> set contains str of atom. False -> contains atom obj
    :return: set of atoms or string of atoms
    """
    return set([(atom.strip() if as_string else pddl.Atom.from_string(atom.strip()))
                for atom in state.split("\t") if not atom.startswith("Negated")])


def parse_state_positive_negative_atoms(state, atoms=None, make_atom=False):
    """
    Parse state formats which name atoms and tell if they are positive or
    negative (e.g. Full).
    If atoms is given, then the atom names shall be missing in the state and
    only + or - is given. The i-th sign is associated with the i-th entry in
    the atoms list. if make_atom is False, then this entry is directly added
    to the needed set (if atoms contains string, then the string is added else
    the atom object or whatever is in there). If make_atom is True, then its value
    is converted to an Atom object
    :param state: Atom a(...)+\tAtom b(...)-... resp. +\t-\t...
    :param atoms: list of atoms
    :param make_atom: True -> convert atom string to atom object. False -> nothing
    :return: set of present atoms and set of absent atoms
    """
    items = state.split("\t")
    pos = set()
    neg = set()
    if atoms is not None and len(atoms) != len(items):
        raise ValueError("State parsing error. Unequal amount of atoms in the"
                         "state to parse and the given atom list.")
    for i in range(len(atoms)):
        atom = atoms[i].strip()
        if atom.startswith("Negated"):
            continue
        if atoms is None:
            right_set = pos if atom[-1] == "+" else neg
            right_set.add(pddl.Atom.from_string(atom[:-1]) if make_atom else atoms[:-1])
        else:
            right_set = pos if atom == "+" else neg
            right_set.add(pddl.Atom.from_string(atoms[i]) if make_atom else atoms[i])

    return pos, neg



def convert_from_fd(state, format, pddl_task, sas_task):
    if format == SampleFormat.FD:
        return state
    state = parse_state_present_atoms(state, as_string=True)
    new_state = ""
    if format in [SampleFormat.Full, SampleFormat.FullShort]:
        for atom in get_cached_groundings(pddl_task):
            new_state += atom if format == SampleFormat.Full else ""
            new_state += ("+" if atom in state else "-") + "\t"
        new_state = new_state[:-1]
    elif format in [SampleFormat.FDFull, SampleFormat.FDFullShort]:
        for atom in get_cached_groundings(sas_task.variables):
            new_state += atom if format == SampleFormat.FDFull else ""
            new_state += ("+" if atom in state else "-") + "\t"
    elif format == SampleFormat.Objects:
        new_state += get_cached_type_obj_pddl(pddl_task) + "\t"
        init = get_cached_no_sas_inits(pddl_task, sas_task) | state
        for atom in init:
            new_state += atom + "\t"
        new_state = new_state[:-1]
    else:
        raise NotImplementedError("The conversion from FD is not implemented "
                                  "to: " + str(format))
    return new_state




def convert_from_X_to_Y(state, in_format, out_format,
                        pddl_task=None, sas_task=None):
    if in_format == SampleFormat.FD:
        return convert_from_fd(state, out_format, pddl_task, sas_task)

    raise NotImplementedError("Conversions from other formats as \"FD\""
                                  "are not supported.")


def convert_data_entry(format, compress,
                       path_final_samples, path_tmp_samples,
                       append=False,
                       path_problem=None, path_domain=None,
                       pddl_task=None, sas_task=None,
                       input_formats=None, default_format=None):

    """

    :param format:
    :param compress:
    :param path_final_samples: Path where to store the samples
    :param path_tmp_samples: Path where to intermediate samples (before conversion)
    :param append: True => Append samples to final file,
                   False => Override final file
    :param path_problem:
    :param path_domain:
    :param pddl_task:
    :param sas_task:
    :param input_format: not binding hin list on which formats the data entries
                         are. This hint is used to check if all necessary data
                         is there. If an format is encountered which was not
                         hinted, it tries to convert them, but might fail if
                         the necessary additional information (e.g. pddl_task
                         object) is not obtainable). TODO IMPLEMENT CHECKS
    :param default_format: if the format field is missing for an entry, this
                           value will be used.
    :return:
    """

    if pddl_task is None or sas_task is None:
        if path_problem is None:
            raise ValueError("Missing needed arguments. Either provide pddl"
                             " and sas task objects or a path to the problem"
                             "file (and if not trivially findable also a path"
                             "to the domain file).")
        if path_domain is None:
            path_domain = parser.find_domain(path_problem)

        (pddl_task, sas_task) = translate.translator.main([path_domain, path_problem, "--no-sas-file", "--log-verbosity", "ERROR"])

    outopen = gzip.open if compress else open
    write_mode = ("a" if append else "w") + ("b" if compress else "")

    with open(path_tmp_samples, 'r') as source:
        with outopen(path_final_samples, write_mode) as target:
            # Process line after line
            for line in source:
                line = line.strip()
                if line.startswith("#"):
                    continue

                meta = None
                new_meta = None
                data = None
                entry_format = default_format
                if line.startswith("<"):
                    meta_end = line.find(">")
                    if meta_end == -1:
                        raise DataCorruptedError("Entry has an opening, but not"
                                                 "closing tag: " + line)
                    meta = line[: meta_end + 1]
                    entry_format, new_meta = SampleFormat.extract_from_meta(meta, default_format, format)
                    data = line[meta_end + 1:]
                else:
                    new_meta = "<Meta format=\"" + format.name + "\">"

                if entry_format is None:
                    raise ValueError("Unable to determine input format of data "
                                     "set entry: " + line)
                if data is None or data == "":
                    raise DataCorruptedError("Invalid data set entry: " + line)
                data = [x.strip() for x in data.split(";")]
                if len(data) < 5:
                    raise DataCorruptedError("Too few fields in data entry: "
                                             + str(len(data)))


                # Main state, goal state, other state
                idxs_states = [0, 1, 3]

                for idx_state in idxs_states:
                    data[idx_state] = convert_from_X_to_Y(
                        data[idx_state], entry_format, format,
                        pddl_task, sas_task)

                # Actually, we do not need to parse the stuff below and can
                # simply copy it.
                # Operator between the main and other state
                #operator = data[2]
                # heuristic value for main state if given
                #heuristic = None if data[4] == "" else  int(data[4])
                # other heeuristic values given as NAME=VALUE
                #other_heuristics = {name.strip(): int(value.strip())
                #                    for (name, value)
                #                    in [other.split("=") for other in data[5:]]}

                entry = new_meta + ";".join(data) + "\n"
                if compress:
                    entry = gzip_input_converter(entry)
                target.write(entry)
    os.remove(path_tmp_samples)



class FastDownwardSamplerBridge(SamplerBridge):
    arguments = parset.ClassArguments('FastDownwardSamplerBridge',
                                      SamplerBridge.arguments,
                                      ("search", False, None, str),
                                      ("format", True, SampleFormat.FD, SampleFormat.get),
                                      ("build", True, "debug64dynamic", str),
                                      ("fd_path", True, "None", str),
                                      ("compress", True, True, parser.convert_bool),
                                      order=["search", "format", "build",
                                             "tmp_dir", "target_file",
                                             "target_dir", "append", "domain",
                                             "makedir", "fd_path",
                                             "compress", "environment", "id"]
                                      )

    def __init__(self, search, format=SampleFormat.FD, build="debug64dynamic",
                 tmp_dir=None, target_file=None, target_dir=None,
                 append=False, domain=None,
                 makedir=False, fd_path=None, compress=True,
                 environment=None, id=None):
        SamplerBridge.__init__(self,tmp_dir, target_file, target_dir,
                               append, domain, makedir, environment, id)

        self._search = search
        self._format = format
        self._build = build
        self._compress = compress

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


    def _sample(self, path_problem, path_samples, path_dir_tmp, path_domain,
                append):
        path_tmp_samples = os.path.join(path_dir_tmp,
                                        os.path.basename(path_problem)
                                        + "." + str(random.randint(0,9999))
                                        + ".tmp")

        cmd = [self._fd_path,
               "--plan-file", "\\real_case\\" + path_tmp_samples + "\\lower_case\\",
               "--build", self._build,
               path_problem, "--search", self._search]
        if path_domain is not None:
            cmd.insert(6, path_domain)


        spt = SubprocessTask("Sampling of " + path_problem + "with in "
                             + path_samples, cmd)

        # TODO Add environment again
        #self._environment.queue_push(spt)
        #event.wait()
        spt.run()

        if self._format != SampleFormat.FD or self._compress:
            convert_data_entry(self._format, self._compress,
                               path_samples, path_tmp_samples,
                               append, path_problem)
        else:
            with open(path_tmp_samples, "r") as src:
                with open(path_samples, "a" if append else "w") as trg:
                    for line in src:
                        trg.write(line)
            os.remove(path_tmp_samples)



    def _finalize(self):
        pass

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  FastDownwardSamplerBridge)


main_register.append_register(FastDownwardSamplerBridge, "fdbridge")
