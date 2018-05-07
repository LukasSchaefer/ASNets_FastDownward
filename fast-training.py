#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from src.training.bridges import StateFormat, LoadSampleBridge
from src.training.networks import MLPDynamicKeras, KerasDataGenerator
from src.training.samplers import DirectorySampler

from src.translate.addons.domain_properties import DomainProperties
tr = None
ts = None
D = None
import argparse
import re
import sys
import numpy as np

CHOICES_FORMAT = []
for name in StateFormat.name2obj:
    CHOICES_FORMAT.append(name)

DESCRIPTIONS = """Train network on previously sampled data."""

ptrain = argparse.ArgumentParser(description=DESCRIPTIONS)

ptrain.add_argument("root", type=str, nargs="+", action="store",
                     help="Path to the root directory with the sampled data. "
                          "The data can be in subfolders, the domain file has"
                          "to be directly in the main folder as domain.pddl)")
ptrain.add_argument("-d", "--directory-filter", type=str,
                     action="append", default=[],
                     help="A subdirectory name has to match the regex otherwise"
                          "it is not traversed. By default no regex matches are"
                          "required. This argument can be given any number of"
                          "time to add additional filters (the directory name has"
                          "to match ALL regexes)")
ptrain.add_argument("-f", "--format", choices=CHOICES_FORMAT,
                     action="store", default=None,
                     help=("State format name into which the loaded data shall"
                           "be converted (if not given, the preferred of the"
                           "network is chosen)"))
ptrain.add_argument("-m", "--max-depth", type=int,
                     action="store", default=None,
                     help="Maximum depth from the root which is traversed ("
                          "default has no maximum, 0 means traversing no"
                          "subfolders, only the content of the root)")
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

    if options.verification is not None:
        options.verification = re.compile(options.verification)

    return options

def load_data(options):
    bridge = LoadSampleBridge(format=options.format, prune=True,
                              skip=options.skip)
    ds = DirectorySampler(bridge, options.root,
                          options.directory_filter, options.problem_filter,
                          options.max_depth, options.selection_depth)

    ds.initialize()
    datas = ds.sample()
    dtrain = []
    dtest = []
    global D
    D = datas
    for data in datas:
        if (data.get_file() is None
                or options.verification is None
                or options.verification.match(data.get_file()) is None):
            dtrain.append(data)
        else:
            dtest.append(data)
    return dtrain, dtest
D = None
G = None
K=None
def train(argv):
    options = parse_train(argv)

    #domprob = DomainProperties.get_property_for(options.root)
    #sys.exit(67)

    dtrain, dtest = load_data(options)
    state_size = len(dtrain[0].data["O"][0][0][dtrain[0].field_current_state].split("\t"))
    global D, G, K

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


    network = MLPDynamicKeras(3, -1, out = options.root[0])
    network.initialize(None, state_size=state_size)
    D = network.train(dtrain, dtest)
    G = network.evaluate(dtest)
    network.analyse()

if __name__ == "__main__":
    #import sys
    #from src.training.main import main
    #main(sys.argc[1:])
    print("HELLO")
    train(sys.argv[1:])

