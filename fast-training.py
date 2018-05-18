#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from src.training.bridges import StateFormat, LoadSampleBridge
from src.training.networks import MLPDynamicKeras, NetworkFormat
from src.training.misc import DomainProperties
from src.training.samplers import DirectorySampler

tr = None
ts = None
D = None
import argparse
import os
import re
import sys
import numpy as np

import time
CHOICE_STATE_FORMATS = []
for name in StateFormat.name2obj:
    CHOICE_STATE_FORMATS.append(name)

CHOICE_NETWORK_FORMATS = []
for name in NetworkFormat.name2obj:
    CHOICE_NETWORK_FORMATS.append(name)


DESCRIPTIONS = """Train network on previously sampled data."""

ptrain = argparse.ArgumentParser(description=DESCRIPTIONS)

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
ptrain.add_argument("-n", "--network", type=str,
                     action="store", default=None,
                     help="Filename under which the trained network shall be "
                          "stored (the network will be stored in the output "
                          "directory). If not given, the network will not be"
                          "stored.")
ptrain.add_argument("-nf", "--network-format", choices=CHOICE_NETWORK_FORMATS,
                     action="append", default=None,
                     help=("List of network formats in which the trained "
                           "network shall be trained. If not given, the networks"
                           "default format is used"))
ptrain.add_argument("-o", "--output", type=str,
                     action="store", default=None,
                     help="Path to the output directory. If non is given, the"
                          "first root directory is used.")
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

    if options.format is None:
        # TODO Take preferred format of network
        options.format = StateFormat.Full
    else:
        options.format = StateFormat.get(options.format)

    print(options.network_format)
    if options.network_format is not None:
        for idx in range(len(options.network_format)):
            options.network_format[idx] = NetworkFormat.by_name(
                options.network_format[idx])

    options.output = options.root[0] if options.output is None else options.output

    if options.verification is not None:
        options.verification = re.compile(options.verification)

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


D = None
G = None
K=None


def train(argv):
    options = parse_train(argv)
    global D, G, K
    domprob = DomainProperties.get_property_for(*options.root)

    dtrain, dtest = load_data(options)
    state_size = len(dtrain[0].data["O"][0][0][dtrain[0].field_current_state].split("\t"))


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

    start_time = time.time()
    network = MLPDynamicKeras(
        3, -1, out=options.root[0],
        store=None if options.network is None else os.path.join(options.output, options.network),
        formats=options.network_format,
        test_similarity="hamming")

    network.initialize(None, domain_properties=domprob)

    D = network.train(dtrain, dtest)

    G = network.evaluate(dtest)
    print("RuNTiME", time.time()-start_time)
    network.analyse(options.output)
    network.finalize()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy
    #ary = np.arange(18).reshape(3,6)
    #plt.matshow(ary, aspect=)

    #for i in np.linspace(0,1,20):
    #    plt.plot(np.arange(10), [i]*10, c=mpl.cm.jet(i))
    #plt.savefig("tst.png")

    #sys.exit(32)
    #import sys
    #from src.training.main import main
    #main(sys.argc[1:])
    print("HELLO")
    train(sys.argv[1:])

