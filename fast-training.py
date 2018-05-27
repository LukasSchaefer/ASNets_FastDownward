#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from src.training import parser, parser_tools

from src.training.bridges import StateFormat, LoadSampleBridge
from src.training.misc import DomainProperties
from src.training.networks import Network, NetworkFormat
from src.training.samplers import DirectorySampler

import argparse
import os
import re
import shlex
import subprocess
import sys
import time


CHOICE_STATE_FORMATS = []
for name in StateFormat.name2obj:
    CHOICE_STATE_FORMATS.append(name)

CHOICE_NETWORK_FORMATS = []
for name in NetworkFormat.name2obj:
    CHOICE_NETWORK_FORMATS.append(name)


DESCRIPTIONS = """Train network on previously sampled data."""

ptrain = argparse.ArgumentParser(description=DESCRIPTIONS)

ptrain.add_argument("network", type=str,
                     action="store", default=None,
                     help="Definition of the network.")
ptrain.add_argument("-a", "--args", type=str,
                     action="store", default=None,
                     help="Single string describing a set of arguments to add "
                          "in front of all arguments if calling another script "
                          "for training execution (see '--execute').")
ptrain.add_argument("-d", "--directory", type=str, nargs="+", action="append",
                    default=[],
                    help="Path to a list of directories from which to load the"
                         " training data. This argument can be given multiple"
                         " times. The execution of this scrips equals then"
                         " calling this script with the same arguments for each "
                         "(if not --sub-directory-training, then the domain file"
                         "is required in the first given directory")
ptrain.add_argument("-sdt", "--sub-directory-training",
                     action="store_true",
                     help="Changes training from one network on the data within "
                          "all given directories to training a single network"
                          "per directory (and subdirectory) which contains a"
                          "domain.pddl file and at least one *.data file (for "
                          "those directories selected, the data is loaded from "
                          "them and from subdirectores like before)")
                    #TODO domain file required top dir?
ptrain.add_argument("-df", "--directory-filter", type=str,
                    action="append", default=[],
                    help="A subdirectory name has to match the regex otherwise"
                         "it is not traversed. By default no regex matches are"
                         "required. This argument can be given any number of"
                         "time to add additional filters (the directory name has"
                         "to match ALL regexes)")
ptrain.add_argument("-dp", "--domain-properties", action="store_true",
                    help="If set and the networks supports it, then the network"
                         " is provided with an analysis of "
                         "properties of the problems domain")
ptrain.add_argument("-e", "--execute", type=str,
                     action="store", default=None,
                     help="Path to script to execute for the training runs. If "
                          "none is given, then this script is used, otherwise, "
                          "it calls an external script in a subprocess and "
                          "passes its parameters")
ptrain.add_argument("-f", "--format", choices=CHOICE_STATE_FORMATS,
                     action="store", default=None,
                     help=("State format name into which the loaded data shall"
                           "be converted (if not given, the preferred of the"
                           "network is chosen)"))
ptrain.add_argument("-l", "--load", type=str,
                     action="store", default=None,
                     help="Overrides the network load location defined in the "
                          "network definition by 'output directory/name'")
ptrain.add_argument("-m", "--max-depth", type=int,
                     action="store", default=None,
                     help="Maximum depth from the root which is traversed ("
                          "default has no maximum, 0 means traversing no"
                          "subfolders, only the content of the root)")
ptrain.add_argument("-n", "--name", type=str,
                     action="store", default=None,
                     help="Overrides the network store location defined in the "
                          "network definition by 'output directory/name'")
ptrain.add_argument("-o", "--output",
                     action="store_true",
                     help="Overrides the output directory specified in the"
                          "network definition with the first root directory.")
ptrain.add_argument("-pf", "--problem-filter", type=str,
                     action="append", default=[],
                     help="A problem file name has to match the regex otherwise"
                          "it is not registered. By default no regex matches are"
                          "required. This argument can be given any number of"
                          "time to add additional filters (the file name has"
                          "to match ALL regexes)")
ptrain.add_argument("-p", "--prefix", type=str,
                     action="store", default=[],
                     help="Prefix to add in front of analysis outputs")
ptrain.add_argument("-s", "--selection-depth", type=int,
                     action="store", default=None,
                     help="Minimum depth from the root which has to be traversed"
                          " before problem files are registered (default has "
                          "no minimum)")
ptrain.add_argument("--skip",
                     action="store_true",
                     help=("If set, then missing sample files are skipped, "
                           "otherwise every problem file is expected to have "
                           "sample file."))
ptrain.add_argument("--skip-magic",
                     action="store_true",
                     help=("Tries to load the sample without performing a check"
                           "that it uses the right reader for the sample file"
                           "format (use case old sample files without magic "
                           "word. USE ONLY IF YOU KNOW WHAT YOU ARE DOING)"))
ptrain.add_argument("-v", "--verification", type=str,
                     action="store", default=None,
                     help="Regex identifying which data sets are used for"
                          "verification of the network performance.")

arguments = set()
for action in ptrain._actions:
    for key in action.option_strings:
        arguments.add(key)


def prepare_training_before_loading(options, directories):

    network = parser.construct(
        parser_tools.ItemCache(),
        parser_tools.main_register.get_register(Network),
        options.network)

    if options.output:
        network.path_out = directories[0]
    if options.name is not None:
        network.path_store = os.path.join(network.path_out, options.name)
    if options.load is not None:
        network.path_load = os.path.join(network.path_out, options.load)

    if options.format is not None:
        format = StateFormat.get(options.format)
    else:
        format = network.get_preferred_state_formats()[0]

    return network, format


def prepare_training_after_loading(options, network, path_domain, paths_problems):
    if (options.domain_properties
            and hasattr(network, "_set_domain_properties")):
        domprob = DomainProperties.get_property_for(path_domain=path_domain,
                                                    paths_problems=paths_problems)
        network._set_domain_properties(domprob)


def get_directory_groups(options):
    """
    Create a nested list. Every entry in the outer list corresponds to the
    directories associated to an independent training run.
    It is checked that the first directory in every list contains a domain.pddl
    file.
    :param options: parsed options
    :return: [[directory, ...]]
    """
    directory_groups = []  # List of list of directories. Each execution gets on outer list
    if options.sub_directory_training:
        todo = []
        for directory_group in options.directory:
            todo.extend(directory_group)
        while len(todo) > 0:
            next_dir = todo.pop()
            if os.path.isfile(os.path.join(next_dir, "domain.pddl")):
                directory_groups.append([next_dir])
            todo.extend([os.path.join(next_dir, sub)
                         for sub in os.listdir(next_dir)
                         if os.path.isdir(os.path.join(next_dir, sub))])
    else:
        for directory_group in options.directory:
            if os.path.isfile(os.path.join(directory_group[0], "domain.pddl")):
                directory_groups.append(directory_group)
    return directory_groups



def load_data(options, directories, format):
    bridge = LoadSampleBridge(format=format, prune=True,
                              skip=options.skip, skip_magic=options.skip_magic)

    dtrain, dtest = None, None
    ignore = []
    # We create a test set
    if options.verification is not None:
        test_problem_filter = list(options.problem_filter)
        test_problem_filter.append(options.verification)

        dir_samp = DirectorySampler(bridge, directories,
                                    options.directory_filter, test_problem_filter,
                                    None,
                                    options.max_depth, options.selection_depth,
                                    merge=True)
        dir_samp.initialize()
        dtest = dir_samp.sample()
        dir_samp.finalize()
        ignore = dir_samp._iterable

    # We create the training set
    dir_samp = DirectorySampler(bridge, directories,
                                options.directory_filter, options.problem_filter,
                                ignore,
                                options.max_depth, options.selection_depth,
                                merge=True)
    dir_samp.initialize()
    dtrain = dir_samp.sample()
    dir_samp.finalize()
    if dtest is not None:
        dtrain[0].remove_duplicates_from_iter(dtest)
    ignore.extend(dir_samp._iterable)
    return dtrain, dtest, ignore


def extract_and_remove(argv, *keys):
    occ = []
    buffer = None
    idx = 0
    while idx < len(argv):
        if argv[idx] in keys:
            if buffer is not None:
                occ.append(buffer)
            del argv[idx]
            idx -= 1
            buffer = []

        elif argv[idx] in arguments:
            if buffer is not None:
                occ.append(buffer)
            buffer = None
        elif buffer is not None:
            buffer.append(argv[idx])
            del argv[idx]
            idx -= 1
        idx += 1

    if buffer is not None:
        occ.append(buffer)
    return occ


def get_execute_call_arguments(options, argv):
    new_command = list(argv)
    new_command.insert(0, options.execute)
    extract_and_remove(new_command, "-e", "--execute")

    idx_directory_group = 2
    if options.args is not None:
        _ = extract_and_remove(new_command, "-a", "--args")
        execute_pre_args = shlex.split(options.args)
        new_command[1:1] = execute_pre_args
        idx_directory_group += len(execute_pre_args)

    if options.sub_directory_training:
        extract_and_remove(new_command, "-sdt", "--sub-directory-training")

    extract_and_remove(new_command, "-d", "--directory")
    return new_command, idx_directory_group


def timing(old_time, msg):
    new_time = time.time()
    print(msg % (new_time-old_time))
    return new_time


def train(argv):
    start_time = time.time()
    options = ptrain.parse_args(argv)
    if options.verification is not None:
        options.verification = re.compile(options.verification)
    directory_groups = get_directory_groups(options)
    if len(directory_groups) == 0:
        raise argparse.ArgumentError("No valid list of directories found.")
    start_time = timing(start_time, "Parsing time: %ss")

    if options.execute is None:
        for idx_dg in range(len(directory_groups)):
            dg = directory_groups[idx_dg]
            print("Processing Directory Group " + str(idx_dg) + ": "
                  + str(dg))
            network, format = prepare_training_before_loading(options, dg)
            dtrain, dtest, problems = load_data(options, dg, format)
            prepare_training_after_loading(options, network,
                                           os.path.join(dg[0], "domain.pddl"),
                                           problems)

            #state_size = len(dtrain[0].data["O"][0][0][dtrain[0].field_current_state].split("\t"))
            start_time = timing(start_time, "Loading data time: %ss")

            network.initialize(None)
            start_time = timing(start_time, "Network initialization time: %ss")

            network.train(dtrain, dtest)
            start_time = timing(start_time, "Network training time: %ss")

            network.evaluate(dtest)
            start_time = timing(start_time, "Network evaluation time: %ss")

            network.analyse()
            start_time = timing(start_time, "Network analysis time: %ss")

            network.finalize()
            _ = timing(start_time, "Network finalization time: %ss")

    else:
        new_command, idx_group = get_execute_call_arguments(options, list(argv))
        for idx_dg in range(len(directory_groups)):
            dg = directory_groups[idx_dg]
            call_command = list(new_command)
            call_command[idx_group: idx_group] = ["--directory"] + dg
            print("Call executable: ", call_command)
            subprocess.call(call_command)
            sys.exit()
if __name__ == "__main__":
    train(sys.argv[1:])

