from __future__ import print_function

from ... import parser

from ...misc import StreamContext

from .... import translate
from ....translate import pddl

import gzip
import os
import random
import sys

# Python 2/3 compatibility
gzip_write_converter = lambda x: x.encode()
gzip_read_converter = lambda x: x.decode()
if sys.version_info[0] == 2:
    gzip_write_converter = lambda x: x
    gzip_read_converter = lambda x: x

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

class InvalidSampleEntryError(Exception):
    pass


# Globals
MAGIC_WORD = "# MAGIC FIRST LINE\n"

# Caching
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


class StateFormat(object):
    name2obj = {}

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self._add_to_enum()

    def _add_to_enum(self):
        setattr(StateFormat, self.name, self)
        StateFormat.name2obj[self.name] = self

    @staticmethod
    def get(name):
        if name not in StateFormat.name2obj:
            raise ValueError("Unknown key for StateFormat: " + str(name))
        return StateFormat.name2obj[name]

    def __str__(self):
        return self.name


StateFormat("FD", "Only reachable and not static predicates (determined by "
                   "translator module) which are true are"
                   "stored (like FastDownwards PDDL format). Atoms are "
                   "alphabetically sorted.")
StateFormat("FDFull", "Only reachable and not static predicates (determined by"
                      "translator module) are stored. If an atom is true, then"
                      "its name is suffixed with \"+\" otherwise with \"-\"."
                      " Atoms are alphabetically sorted.")
StateFormat("FDFullShort", "Like FDAll, but the atom names are skipped. As their"
                        "order is alphabetical you can retain it.")

StateFormat("Full", "All atoms are given in alphabetical order annotated with"
                     "\"+\" if true in the state and \"-\" otherwise.")
StateFormat("FullShort", "Like Full, but the atom names are skipped. As their"
                          "order is alphabetical you can retain it.")
StateFormat("Objects", "Adds of the problem to the meta tag and then provides"
                        "only the present grounded predicates.")


def split_meta(line):
    line = line.strip()
    if not line.startswith("<"):
        return None, line
    else:
        meta_end = line.find(">")
        if meta_end == -1:
            raise DataCorruptedError("Entry has an opening, but not"
                                     "closing tag: " + line)
        return line[: meta_end + 1], line[meta_end + 1:]


def extract_from_meta(meta, key, default=None, converter=None):
    if meta is None:
        return default, None, None

    if not meta[0] == "<":
        raise ValueError("Given meta is no meta tag: " + meta)
    idx = meta.find(key + "=")
    if idx == -1:
        return default, None, None
    else:
        start_idx = idx + len(key) + 1
        end_idx = None
        if meta[start_idx] == "\"":
            start_idx += 1
            end_idx = meta.find("\"", start_idx)
            if end_idx == -1:
                raise ValueError("Unable to determine end of meta entry "
                                 "description: " + meta)
        else:
            end_idx = meta.find(" ", start_idx)
            if end_idx == -1:
                if meta[-1] != ">":
                    raise ValueError(
                        "Unable to determine end of meta entry "
                        "description: " + meta)

        value = meta[start_idx:end_idx]
        return ((value if converter is None else converter(value)),
                start_idx, end_idx)


def extract_and_replace_from_meta(meta, key, default=None, new_value=None,
                                  str2val=None):
    if meta is None:
        meta = "<Meta>"
    value, start_idx, end_idx = extract_from_meta(meta, key, converter=str2val)
    if value is None:
        if default is None:
            raise ValueError("Unable to determine value for key " + str(key))
        else:
            value = default
            start_idx = meta.find(">")
            if start_idx == -1:
                raise ValueError("No valid meta tag")
            end_idx = start_idx
            if new_value is not None:
                new_value = " " + str(key) + "=\"" + str(new_value) + "\""

    if new_value is not None:
        meta = meta[:start_idx] + str(new_value) + meta[end_idx :]
    return value, meta


def load_sample_line(line, data_container, format, pddl_task, sas_task, pruning_set=None, default_format=None):
    line = line.strip()
    if line.startswith("#") or line == "":
        return None, None, None

    meta, data = split_meta(line)
    if meta is None and default_format is None:
        raise InvalidSampleEntryError("Missing meta tag. Cannot infer state format.")

    entry_type, _, _ = extract_from_meta(meta, "type")
    entry_format, meta = extract_and_replace_from_meta(
        meta, "format", default=default_format, new_value=format.name,
        str2val=StateFormat.get)

    if pruning_set is not None:
        h_data = hash((entry_type, data))
        if h_data in pruning_set:
            return None, None, None
        else:
            pruning_set.add(h_data)

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

    entry = meta + ";".join(data) + "\n"
    if data_container is not None:
        data[4] = int(data[4])
        data_container.add(data, type=entry_type)
    return entry, meta, data


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
    if format == StateFormat.FD:
        return state
    state = parse_state_present_atoms(state, as_string=True)
    new_state = ""
    if format in [StateFormat.Full, StateFormat.FullShort]:
        for atom in get_cached_groundings(pddl_task):
            new_state += atom if format == StateFormat.Full else ""
            new_state += ("+" if atom in state else "-") + "\t"
        new_state = new_state[:-1]
    elif format in [StateFormat.FDFull, StateFormat.FDFullShort]:
        for atom in get_cached_groundings(sas_task.variables):
            new_state += atom if format == StateFormat.FDFull else ""
            new_state += ("+" if atom in state else "-") + "\t"
    elif format == StateFormat.Objects:
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
    if in_format == StateFormat.FD:
        return convert_from_fd(state, out_format, pddl_task, sas_task)

    if in_format == out_format:
        return state
    raise NotImplementedError("Conversions from other formats as \"FD\""
                                  "are not supported.")



def load_and_convert_data(path_read, format, default_format=None, prune=True,
                          path_problem=None, path_domain=None,
                          pddl_task=None, sas_task=None,
                          data_container=None,
                          delete=False,
                          skip_magic_word_check=False, forget=0.0, write_context=None):
    """

    :param path_read: Path to the file containing the samples to load
    :param format: StateFormat into which to convert the samples
    :param default_format: Input format of the loading samples
                           (if not specified within sample entry)
    :param prune: If true prunes duplicate entries (the meta information is
                  except for the type attribute ignored)
    :param path_problem: Path to the problem description to which the samples
                         belong. For converting the samples, the problem has to
                         be known. Provide at least one of the following
                         parameter settings:
                         - pddl_task and sas_task
                         - path_problem and path_domain
                         - path_problem and path_domain can be automatically
                           detected
                        If multiple settings are given, the first available one
                        of this list is used.
    :param path_domain: Path to the domain description of the problem
                        (for interactions of this parameter see path_problem)
    :param pddl_task: PDDL Task object of the problem
                     (for interactions of this parameter see path_problem)
    :param sas_task: SAS Task object of the problem
                     (for interactions of this parameter see path_problem)
    :param data_container: Data gathering object for the loaded entries. Object
                           requires an add(entry, type) method
                           (e.g. SizeBatchData). If None is given, the adding
                           is skipped.
    :param delete: Deletes path_input at the end of this method
    :param skip_magic_word_check: Deprecated. Use this to read old sample files.
                                  The outcome when reading a file with a wrong
                                  reading technique (e.g. compressed file via
                                  open) is undetermined.
    :param write_context: If given, this context manager will be used to write
                          the output into. It is required to have the following
                          methods: open(file, mode) which informs about the file
                          we would like to write and returns the context manager,
                          write(*args, **kwargs), and close()
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

        (pddl_task, sas_task) = translate.translator.main(
            [path_domain, path_problem,
             "--no-sas-file", "--log-verbosity", "ERROR"])

    write_context = StreamContext() if write_context is None else write_context

    # Reading techniques and settings
    old_hashs = set() if prune else None
    read_format_found=False
    read_techniques = [(open, lambda x: x), (gzip.open, gzip_read_converter)]

    for (read_open, read_conv) in read_techniques:
        try:
            with read_open(path_read, "r") as src:
                with write_context.next(path_problem) as trg:

                    first = False if skip_magic_word_check else True
                    for line in src:
                        line = read_conv(line)

                        if first:
                            first = False
                            read_format_found = line == MAGIC_WORD
                            if read_format_found:
                                trg.write(MAGIC_WORD)
                            else:
                                break

                        else:
                            if forget != 0.0 and random.random() < forget:
                                continue
                            entry, _, _ = load_sample_line(
                                line, data_container, format,
                                pddl_task, sas_task, old_hashs, default_format)
                            if entry is not None:
                                trg.write(entry)

                    # The first reading now throwing an exception is accepted
                    if skip_magic_word_check:
                        read_format_found = True
                    if delete:
                        os.remove(path_read)
                    if read_format_found:
                        break

        except UnicodeDecodeError:
            pass
        except InvalidSampleEntryError as e:
            if sys.version_info < (3,):
                pass
            else:
                raise e

    if not read_format_found:
        raise ValueError("the given file could not be correctly opened with one "
                         "of the known techniques.")


def context_load(stream_context, data_container, format, prune=True,
         path_problem=None, path_domain=None,
         pddl_task=None, sas_task=None,
         default_format=None, skip=True, skip_magic=False, forget=0.0):

    paths_samples = set()
    for stream in stream_context._streams:
        paths_samples.add(stream.get_next_path(path_problem))

    for path_samples in paths_samples:
        if not os.path.exists(path_samples):
            if skip:
                continue
            else:
                raise FileNotFoundError("A sample file to load does not exist:"
                                        + str(path_samples))
        load_and_convert_data(
            path_read=path_samples,
            format=format, default_format=default_format, prune=prune,
            path_problem=path_problem, path_domain=path_domain,
            pddl_task=pddl_task, sas_task=sas_task,
            data_container=data_container,
            skip_magic_word_check=skip_magic, forget=forget,
            write_context=None)




