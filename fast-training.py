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
ptrain.add_argument("root", type=str, nargs="+", action="store",
                     help="Path to the root directories with the sampled data. "
                          "The data can be in subfolders, the domain file has"
                          "to be directly in the main folder as domain.pddl)")
ptrain.add_argument("-d", "--directory-filter", type=str,
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
ptrain.add_argument("-f", "--format", choices=CHOICE_STATE_FORMATS,
                     action="store", default=None,
                     help=("State format name into which the loaded data shall"
                           "be converted (if not given, the preferred of the"
                           "network is chosen)"))
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
ptrain.add_argument("-p", "--problem-filter", type=str,
                     action="append", default=[],
                     help="A problem file name has to match the regex otherwise"
                          "it is not registered. By default no regex matches are"
                          "required. This argument can be given any number of"
                          "time to add additional filters (the file name has"
                          "to match ALL regexes)")
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


def parse_train(argv):
    options = ptrain.parse_args(argv)

    options.network = parser.construct(
        parser_tools.ItemCache(),
        parser_tools.main_register.get_register(Network),
        options.network)

    if options.output:
        options.network.path_out = options.root[0]
    if options.name is not None:
        options.network.path_store = os.path.join(options.network.path_out,
                                                  options.name)

    if options.format is not None:
        options.format = StateFormat.get(options.format)
    else:
        options.format = options.network.get_preferred_state_formats()[0]

    if options.verification is not None:
        options.verification = re.compile(options.verification)

    if (options.domain_properties
            and hasattr(options.network, "_set_domain_properties")):
        domprob = DomainProperties.get_property_for(*options.root)
        options.network._set_domain_properties(domprob)

    return options


def load_data(options):
    bridge = LoadSampleBridge(format=options.format, prune=True,
                              skip=options.skip, skip_magic=options.skip_magic)

    dtrain, dtest = None, None
    ignore = set()
    # We create a test set
    if options.verification is not None:
        test_problem_filter = list(options.problem_filter)
        test_problem_filter.append(options.verification)

        dir_samp = DirectorySampler(bridge, options.root,
                                    options.directory_filter, test_problem_filter,
                                    None,
                                    options.max_depth, options.selection_depth,
                                    merge=True)
        dir_samp.initialize()
        dtest = dir_samp.sample()
        dir_samp.finalize()
        ignore = dir_samp._iterable

    # We create the training set
    dir_samp = DirectorySampler(bridge, options.root,
                                options.directory_filter, options.problem_filter,
                                ignore,
                                options.max_depth, options.selection_depth,
                                merge=True)
    dir_samp.initialize()
    dtrain = dir_samp.sample()
    dir_samp.finalize()
    if dtest is not None:
        dtrain[0].remove_duplicates_from_iter(dtest)

    return dtrain, dtest


def timing(old_time, msg):
    new_time = time.time()
    print(msg % (new_time-old_time))
    return new_time

def train(argv):
    start_time = time.time()
    options = parse_train(argv)
    start_time = timing(start_time, "Parsing time: %ss")

    dtrain, dtest = load_data(options)
    state_size = len(dtrain[0].data["O"][0][0][dtrain[0].field_current_state].split("\t"))
    start_time = timing(start_time, "Loading data time: %ss")

    options.network.initialize(None)
    start_time = timing(start_time, "Network initialization time: %ss")

    options.network.train(dtrain, dtest)
    start_time = timing(start_time, "Network training time: %ss")

    options.network.evaluate(dtest)
    start_time = timing(start_time, "Network evaluation time: %ss")

    options.network.analyse()
    start_time = timing(start_time, "Network analysis time: %ss")

    options.network.finalize()
    _ = timing(start_time, "Network finalization time: %ss")



if __name__ == "__main__":
    train(sys.argv[1:])

"""
    def _convert_state(state):
        parts = state.split("\t")
        for idx in range(len(parts)):
            parts[idx] = 0 if parts[idx][-1] == "-" else 1
        return np.array(parts)

    def _convert_data( data):
        data = data if isinstance(data, list) else [data]
        for data_set in data:
            for type in data_set.data:
                for batch in data_set.data[type]:
                    for entry in batch:
                        entry[data_set.field_current_state] = (
                            _convert_state(entry[data_set.field_current_state]))
                        entry[data_set.field_goal_state] = (
                            _convert_state(
                                entry[data_set.field_goal_state]))
                        entry[data_set.field_other_state] = (
                            _convert_state(
                                entry[data_set.field_other_state]))
            data_set.finalize()
        return data
        
    
    #dtrain = _convert_data(dtrain)
    #dtest = _convert_data(dtest)
    #kdg_train = KerasDataGenerator(dtrain,
    #                               x_fields=[dtrain[0].field_current_state,
    #                                         dtrain[0].field_goal_state],
    #                               x_converter=lambda x: [
    #                                   np.stack(x[:, 0], axis=0),
    #                                   np.stack(x[:, 1], axis=0)],
    #                               precaching=True)
    #D = kdg_train[0][0][0]
    #G = kdg_train


        """
