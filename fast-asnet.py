#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time
import sys
import ast
from keras import optimizers
import numpy as np

from src.translate.translator import main as translate
from src.translate.normalize import normalize
from src.translate.instantiate import instantiate, get_fluent_predicates
from src.translate.build_model import compute_model
from src.translate.pddl_to_prolog import translate as pddl_to_prolog

sys.path.append("network_models/asnets")
from problem_meta import ProblemMeta
from asnet_keras_model import ASNet_Model_Builder
from utils import custom_binary_crossentropy

from src.training import parser, parser_tools
from src.training.bridges.sampling_bridges.asnet_sampling_bridge import ASNetSampleBridge
from src.training.environments.task import SubprocessTask
from src.training.misc import StreamContext, StreamDefinition, DomainProperties
from src.training.networks import Network, NetworkFormat
from src.training.networks.keras_networks.keras_asnet import KerasASNet
from src.training.problem_sorter.base_sorters import DifficultySorter
from src.training.samplers import IterableFileSampler

DESCRIPTIONS = """Train, sample and evaluate ASNets."""

pasnet = argparse.ArgumentParser(description=DESCRIPTIONS)

# arguments generally from fast-training (some were removed/ not necessary or currently used for ASNets):
pasnet.add_argument("-d", "--directory", type=str,
                    action="store", default=None,
                    help="Path to directory from which to load the "
                         "training data.")
pasnet.add_argument("-w", "--weights", type=str,
                    action="store", default=None,
                    help="Path to weight file to load weights from.")
pasnet.add_argument("--dry",
                     action="store_true",
                     help="Tells only which trainings it would perform, but does "
                          "not perform the training step.")
pasnet.add_argument("-fin", "--finalize", type=str, nargs="+",
                    action="store", default=[],
                    help="List some key=value pairs which are passed "
                         "as key=value to the networks finalize method.")
pasnet.add_argument("--forget", type=float,
                     action="store", default=0.0,
                     help=("Probability of skipping to load entries of the "
                           "verification data"))
pasnet.add_argument("-init", "--initialize", type=str, nargs="+",
                    action="store", default=[],
                    help="List some key=value pairs which are passed "
                         "as key=value to the networks initialize method.")
pasnet.add_argument("-p", "--prefix", type=str,
                     action="store", default="",
                     help="Prefix to add in front of analysis outputs")
pasnet.add_argument("--skip",
                     action="store_true",
                     help=("If set, then missing sample files are skipped, "
                           "otherwise every problem file is expected to have "
                           "sample file."))
pasnet.add_argument("-v", "--verification", type=str,
                     action="store", default=None,
                     help="Regex for grouping the data set files into 'use for "
                          "verification' and 'use for training'.")
pasnet.add_argument("-vs", "--verification-split", type=float,
                     action="store", default=0.0,
                     help="Fraction of the trainings data to split off for the "
                          "verification data.")
                          
pasnet.add_argument("-t", "--train", action="store_true",
                     help="Flag indicating that training should be executed.")
pasnet.add_argument("-e", "--evaluate", action="store_true",
                     help="Flag indicating that evaluation should be executed.")
pasnet.add_argument("--sort_problems", type=bool,
                     action="store", default=False,
                     help="Boolean value indicating whether the problems should "
                          "be sorted by difficulty.")
pasnet.add_argument("--batch", type=int,
                     action="store", default=300,
                     help="Batch size for training.")
pasnet.add_argument("--epochs", type=int,
                     action="store", default=300,
                     help="Number of epochs during training.")
pasnet.add_argument("--delete", type=bool,
                     action="store", default=True,
                     help="Flag indicating whether the sample file will be deleted "
                     "after data extraction.")

# ASNet sampling search arguments
pasnet.add_argument("--build", type=str,
                    action="store", default="debug64dynamic",
                    help="Build to use for ASNet Sampling Search.")
pasnet.add_argument("--teacher_search", type=str,
                    action="store", default="astar(lmcut(), transform=asnet_sampling_transform())",
                    help="Teacher search configuration to use during ASNet "
                         "Sampling Search.")
pasnet.add_argument("--trajectory_limit", type=int,
                    action="store", default=300,
                    help="Limit for explorated sample trajectories.")
pasnet.add_argument("--use_non_goal_teacher_paths", type=bool,
                    action="store", default=True,
                    help="Bool value indicating whether paths/ trajectories of "
                         "the teacher search not reaching a goal state should be "
                         "sampled in the ASNet Sampling.")
pasnet.add_argument("--use_teacher_search", type=bool,
                    action="store", default=True,
                    help="Bool value indicating whether the teacher search should "
                         "be used for sampling.")

# ASNet model arguments
pasnet.add_argument("-layers", "--number_of_layers", type=int,
                    action="store", default=2,
                    help="Number of layers of the ASNet (number_of_layers proposition "
                         "layers and number_of_layers + 1 action layers)")
pasnet.add_argument("-hsize", "--hidden_rep_size", type=int,
                    action="store", default=16,
                    help="Hidden representation size used for every module (= size of "
                         "module outputs)")
pasnet.add_argument("-act", "--activation", type=str,
                    action="store", default='relu',
                    help="Name of activation function to be used in all modules of all "
                         "layers but the last output layer")
pasnet.add_argument("-drop", "--dropout", type=float,
                    action="store", default=0.0,
                    help="Dropout rate used in every intermediate node (= probability to "
                         "deactivate intermediate nodes)")
pasnet.add_argument("-ki", "--kernel_initializer", type=str,
                    action="store", default='glorot_normal',
                    help="Initializer to be used for all weight matrices/ kernels of all "
                         "modules")
pasnet.add_argument("-bi", "--bias_initializer", type=str,
                    action="store", default='zeros',
                    help="Initializer to be used for all bias vectors of all modules")
pasnet.add_argument("--extra_input_features", type=str,
                    action="store", default=None,
                    help="Additional input features per action. This involves additional "
                         "heuristic input features landmarks or None.")
pasnet.add_argument("-opt", "--optimizer", type=str,
                    action="store", default='adam',
                    help="Optimizer to be used during training")

arguments = set()
for action in pasnet._actions:
    for key in action.option_strings:
        arguments.add(key)


def parse_key_value_pairs_to_kwargs(pairs):
    kwargs = {}
    for pair in pairs:
        idx = pair.find("=")
        if idx == -1:
            raise argparse.ArgumentError("The key=value pairs need an '=' sign "
                                         "to separate keys from values.")
        kwargs[pair[:idx]] = pair[idx + 1:]
    return kwargs


def get_problems_from_directory(directory, sort_problems):
    """
    Create a list with all problem file paths of the given directory group
    :param directory_group: directory group to extract problems from
    :param sort_problems: boolean value indicating whether the problems should
                          be sorted by difficulty
    :return: [problem1, problem2, ...]
    """
    # collect problems
    problem_list = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            # f is a file
            if f != "domain.pddl":
                if f.endswith(".pddl"):
                    # f is pddl file and not the domain -> problem file
                    problem_list.append(os.path.join(directory, f))
    if sort_problems:
        # sort problems by difficulty
        difficulty_sorter = DifficultySorter()
        # dict in format: {difficulty: [problem1, problem2], difficulty2: [...], ...}
        # with difficulties being numbers as keys and list of corresponding problem files as values
        problem_diff_dict = difficulty_sorter.sort(problem_list)
        # sort difficulty numbers
        difficulties = list(problem_diff_dict.keys())
        difficulties.sort()
        # extract problems
        problem_list = []
        for diff in difficulties:
            problem_list.extend(problem_diff_dict[diff])

    return problem_list


def create_pddl_task(domain_path, problem_path):
    """
    Reads, translates the pddl_task and simplifies it to remove
    unused/ unreachable predicates etc.
    Also creates the corresponding task_meta file
    :return: pddl_task, task_meta
    """
    print("Processing PDDL Translation.")
    # deactivate prints for translation
    sys.stdout = open(os.devnull, 'w')
    pddl_task, _ = translate([domain_path, problem_path])
    normalize(pddl_task)
    prog = pddl_to_prolog(pddl_task)
    model = compute_model(prog)
    _, grounded_predicates, propositional_actions, _, _ = instantiate(pddl_task, model)
    fluent_predicates = get_fluent_predicates(pddl_task, model)
    pddl_task.simplify(fluent_predicates)

    # reactivate prints
    sys.stdout = sys.__stdout__

    print("Computing task meta information.")
    task_meta = ProblemMeta(pddl_task, propositional_actions, grounded_predicates)
    return pddl_task, task_meta


def create_asnet_model(task_meta, options, extra_input_size, weights_path=None):
    """
    Builds and compiles the Keras ASNet Model

    :param task_meta: task meta-information necessary to build the model
    :param options: parser options containing values:
        - number_of_layers: number of layers of the ASNet
                            (number_of_layers proposition layers and number_of_layers+ 1 action layers)
        - hidden_representation_size: hidden representation size used for every
                                      module (= size of module outputs)
        - activation: name of activation function to be used in all modules
                      of all layers but the last output layer
        - dropout: dropout rate used in every intermediate node (= probability to
                   deactivate intermediate nodes)
        - kernel_initializer: initializer to be used for all weight matrices/ kernels
                              of all modules
        - bias_initializer: initializer to be used for all bias vectors
                            of all modules
        - optimizer: optimizer to be used during training
        - loss: loss function to be used during training
    :param extra_input_size: size of additional input features per action
                             This usually involves additional heuristic input features
                             like values indicating landmark values for actions (as used
                             in the paper of Toyer et al. about ASNets)
    :param weights_path: path to a .h5 file with weights of previous ASNet to be reused 
                         (or None if no weights should be reused)
    :return: Keras ASNet Model
    """
    print("Building the ASNet keras model...")
    asnet_builder = ASNet_Model_Builder(task_meta)
    asnet_model = asnet_builder.build_asnet_keras_model(options.number_of_layers,
                                                        options.hidden_rep_size,
                                                        options.activation,
                                                        options.dropout,
                                                        options.kernel_initializer,
                                                        options.bias_initializer,
                                                        extra_input_size)
    asnet_model.compile(loss=custom_binary_crossentropy, optimizer=options.optimizer)
    if weights_path:
        print("Loading previous weights")
        asnet_model.load_weights(weights_path, by_name=True)
    print("Done building the model")
    return asnet_model


def prepare_and_construct_network_before_loading(options, extra_input_size, model):
    """
    Constructs KerasASNet Network for training etc. with the given model

    :param options: parser options
    :param extra_input_size: size for additional input features per action
    :param model: created ASNet Keras model
    :return: KerasASNet Network-object
    """
    network = KerasASNet(extra_input_size=extra_input_size,
                         model=model,
                         epochs=options.epochs)

    return network


def sample(options, directory, domain_path, problem_path, extra_input_size):
    """
    Execute sampling search for asnets with given configuration

    :param options: parser options for sampling search configuration
    :param directory: currently processed directory
    :param domain_path: path to domain file
    :param problem_path: path to problem file to execute sampling on
    :param extra_input_size: size for additional input features per action
    :return:
    """
    cmd = ['python', 'fast-downward.py',
                "--build", options.build, domain_path, problem_path,
                '--search', 'asnet_sampling_search(search=' + options.teacher_search +
                            ', trajectory_limit=' + str(options.trajectory_limit) +
                            ', use_non_goal_teacher_paths=' + str(options.use_non_goal_teacher_paths) +
                            ', use_teacher_search=' + str(options.use_teacher_search) +
                            ', network_search=policy_search(p=np(network=asnet(path=' + str(os.path.join(directory, 'asnet.pb')) + ', extra_input_size=' + str(extra_input_size) + ')))' +
                            ', target=' + os.path.join(directory, "sample.data") + ')']
    spt = SubprocessTask('Sampling of ' + problem_path, cmd)
    spt.run()


def load_data(options, directory, domain_path, problem_path, extra_input_size):
    """
    Load sampling information via ASNetBridge from the sample file

    :param options: parser options
    :param directory: directory containing all problem files etc.
    :param domain_path: path to domain file
    :param problem_path: path to problem file the sampling information is from
    :param extra_input_size: size for additional input features per action
    :return: dtrain, dtest
        with both being lists of SizeBatchData
    """
    bridge = ASNetSampleBridge(sample_path=os.path.join(directory, "sample.data"),
                               domain=domain_path, prune=True, forget=options.forget,
                               skip=options.skip, extra_input_size=extra_input_size,
                               delete=options.delete)

    dtrain, dtest = None, None
    # We create a test set
    if options.verification is not None:
        test_problem_filter = list(options.problem_filter)
        test_problem_filter.append(options.verification)

        dir_samp = IterableFileSampler(bridge, [problem_path], options.batch,
                                       merge=True)
        dir_samp.initialize()
        dtest = dir_samp.sample()
        dir_samp.finalize()

    # We create the training set
    bridge._forget = 0.0
    dir_samp = IterableFileSampler(bridge, [problem_path], options.batch,
                                   merge=True)
    dir_samp.initialize()
    dtrain = dir_samp.sample()
    dir_samp.finalize()
    if options.verification_split > 0:
        splitted = dtrain[0].splitoff(options.verification_split)[0]
        dtest[0].add_data(splitted)
    if dtest is not None:
        dtrain[0].remove_duplicates_from_iter(dtest)
    return dtrain, dtest


def timing(old_time, msg):
    new_time = time.time()
    print(msg % (new_time-old_time))
    return new_time


def train(options, directory, domain_path, problem_list):
    """
    Training ASNet with options corresponding configuration for all problems in problem_list
    Saves the network in a protobuf file 'asnet.pb' in given directory

    :param options: parser options
    :param directory: directory containing all relevant files
    :param domain_path: path to domain file for all problems
    :param problem_list: list of paths to problem files of the domain
    :return: 
    """
    start_time = time.time()
    if options.verification is not None:
        options.verification = re.compile(options.verification)
    
    options.initialize = parse_key_value_pairs_to_kwargs(options.initialize)
    options.finalize = parse_key_value_pairs_to_kwargs(options.finalize)

    if options.extra_input_features is None:
        extra_input_size = 0
    else:
        if options.extra_input_features == "landmarks":
            extra_input_size = 1
        else:
            raise ValueError("Invalid extra input feature value: %s" % options.extra_input_features)

    start_time = timing(start_time, "Parsing time: %ss")

    # path to previous weight file (or None if not existing)
    previous_asnet_weights = options.weights

    asnet = None
    for problem_index, problem_path in enumerate(problem_list):
        print("Processing problem file " + str(problem_path) + " ("
                + str(problem_index + 1) + "/ " + str(len(problem_list)) + ")")
        if options.dry:
            continue

        _, task_meta = create_pddl_task(domain_path, problem_path)
        start_time = timing(start_time, "PDDL translation time: %ss")

        asnet_model = create_asnet_model(task_meta, options, extra_input_size, previous_asnet_weights)
        start_time = timing(start_time, "Keras model creation time: %ss")

        asnet = prepare_and_construct_network_before_loading(options, extra_input_size, asnet_model)
        # store protobuf network file for fast-downward sampling
        if os.path.isfile(os.path.join(directory, "asnet.pb")):
            os.remove(os.path.join(directory, "asnet.pb"))
        asnet._store(os.path.join(directory, "asnet"), [NetworkFormat.protobuf])
        start_time = timing(start_time, "Preparing and storing of the network time: %ss")

        # build ASNetSamplingSearch command and execute for sampling -> saves samples in sample.data
        sample(options, directory, domain_path, problem_path, extra_input_size)
        start_time = timing(start_time, "Sampling search time: %ss")

        dtrain, dtest = load_data(options, directory, domain_path, problem_path, extra_input_size)

        start_time = timing(start_time, "Loading data time: %ss")

        asnet.initialize(None, **options.initialize)
        start_time = timing(start_time, "Network initialization time: %ss")

        asnet.train(dtrain, dtest)
        start_time = timing(start_time, "Network training time: %ss")

        asnet.evaluate(dtest)
        start_time = timing(start_time, "Network evaluation time: %ss")

        asnet.analyse(prefix=options.prefix)
        start_time = timing(start_time, "Network analysis time: %ss")

        asnet.finalize(**options.finalize)
        _ = timing(start_time, "Network finalization time: %ss")

        # saving weights for next problem instance
        if previous_asnet_weights is not None:
            os.remove(previous_asnet_weights)
        asnet._store_weights(os.path.join(directory, "asnet_weights.h5"))
        previous_asnet_weights = os.path.join(directory, "asnet_weights.h5")

    # save model in protobuf for fast-downward
    if asnet is not None:
        asnet._store(os.path.join(directory, "asnet"), NetworkFormat.protobuf)


def evaluate(options, directory, domain_path, problem_list):
    """
    Evaluate ASNet with options corresponding configuration for all problems in problem_list
    Saves the network results in "<problem_name>_result"

    :param options: parser options
    :param directory: directory containing all relevant files
    :param domain_path: path to domain file for all problems
    :param problem_list: list of paths to problem files of the domain
    :return: 
    """
    # TODO


def main(argv):
    options = pasnet.parse_args(argv[1:])

    # check that domain file is present
    directory = options.directory
    if directory is None:
        raise ValueError("No directory given")
    domain_path = os.path.join(directory, "domain.pddl")
    if not os.path.isfile(domain_path):
        raise ValueError("No 'domain.pddl' file found in the given directory.")

    # problem_list contains problem paths
    problem_list = get_problems_from_directory(directory, options.sort_problems)
    if len(problem_list) == 0:
        raise ValueError("No valid problem files found.")

    if options.train:
        train(options, directory, domain_path, problem_list)
    elif options.evaluate:
        evaluate(options, directory, domain_path, problem_list)
    else:
        raise ValueError("Neither evaluate nor train flag set.")
                

if __name__ == "__main__":
    main(sys.argv)
