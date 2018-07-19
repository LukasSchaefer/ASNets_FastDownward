#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import os
import re
import sys
import subprocess
import time

from keras.optimizers import Adam, SGD
from keras import backend as K

sys.path.append("network_models/asnets")
from problem_meta import ProblemMeta
from asnet_keras_model import ASNet_Model_Builder
from losses import custom_binary_crossentropy, just_opt_custom_binary_crossentropy

from src.translate.translator import main as translate
from src.translate.instantiate import instantiate, get_fluent_predicates
from src.translate.build_model import compute_model
from src.translate.pddl_to_prolog import translate as pddl_to_prolog

from src.training import parser, parser_tools
from src.training.bridges.sampling_bridges.asnet_sampling_bridge import ASNetSampleBridge
from src.training.misc import StreamContext, StreamDefinition, DomainProperties
from src.training.networks import Network, NetworkFormat
from src.training.networks.keras_networks.keras_asnet import KerasASNet
from src.training.networks.keras_networks.keras_tools import store_keras_model_as_protobuf
from src.training.problem_sorter.base_sorters import DifficultySorter
from src.training.samplers import IterableFileSampler

DESCRIPTION = """Action Schema Networks (ASNets) Training, Sampling and Evaluation\n

The script supports two modes of execution:\n
\t-t/--train: Trains ASNets on given problems with the given configuration\n
\t-e/--evaluate: Evaluates ASNets on given problems with the given configuration

Use -h within a block to show the help menu for the block (this ends the processing
of the block and continues with the next block)

The scripts always need a given directory path after -d/--directory which contains
a PDDL domain-file 'domain.pddl' with this exact name as well as an arbitrary amount
of PDDL problem instances of this domain. The training and/ or evaluation will be executed
on these files.

TRAINING:\n
Training performs EPOCHS (--epochs EPOCHS; default 300) Epochs in which training is
performed for each given problem. Training first builds the network model and then
executes ASNet sampling via a fast-downward sampling search (using a teacher search;
default A* with LM-Cut heuristic). This process samples a set of states explored
which are afterwards used to train the network for the current problem in TRAIN_EPOCHS
(--train_epochs TRAIN_EPOCHS; default 300) training-epochs. After this process the
learned weights are saved and used in the next iteration (epoch or problem).
This iterative sampling and training mechanism stops after all EPOCHS are performed or
the early stopping criteria are met.

EVALUATION:\n
Evaluation executes a naive network policy search for each problem in the given directory.
The evaluation process should always be given previously trained weights of ASNets of the
exact same domain and with the same configuration by the -w/--weights <path_to_weight_file.h5>
argument. These weights have to be saved in a HDF5 file. This can be achieved in keras by
using 'model.save_weights(weight_file_path)'. Otherwise new weights have to be generated for
evaluation which would not be trained and are therefore expected to perform poorly.\n

Example:\n
./SCRIPT -t -d DIRECTORY OPTIONS - train ASNets on all problems in DIRECTORY with OPTIONS\n
./SCRIPT -e -d DIRECTORY -w WEIGHT_FILE OPTIONS - loads the weights from WEIGHT_FILE and
    executes the network policy search for all problems in DIRECTORY with OPTIONS\n
./SCRIPT -t -e -d DIRECTORY - first train ASNets on all problems in DIRECTORY with OPTIONS
    and afterwards executes the network policy search for all these problems using the given
    OPTIONS and the weights obtained from the training"""

pasnet = argparse.ArgumentParser(description=DESCRIPTION)

# arguments generally from fast-training (some were removed/ not necessary or currently used for ASNets):
pasnet.add_argument("-d", "--directory", type=str,
                    action="store", default=None,
                    help="Path to directory from which to load the "
                         "training data. This directory must contain "
                         "a 'domain.pddl' file with this exact name and "
                         "an arbitrary amount of problem instances of this "
                         "exact domain.")
pasnet.add_argument("-w", "--weights", type=str,
                    action="store", default=None,
                    help="Path to ASNet HDF5 weight file to load weights from. "
                         "Note that these weights have to be generated from a "
                         "ASNet for the same domain as they are intended to be "
                         "used for (set by -d).")
pasnet.add_argument("--dry",
                     action="store_true",
                     help="Tells only which trainings/ evaluations it would perform, but does "
                          "not perform these step.")
pasnet.add_argument("--skip",
                     action="store_true",
                     help=("If set, then missing sample files are skipped, "
                           "otherwise every problem file is expected to have "
                           "sample file."))
pasnet.add_argument("-init", "--initialize", type=str, nargs="+",
                    action="store", default=[],
                    help="List some key=value pairs which are passed "
                         "as key=value to the networks initialize method.")
pasnet.add_argument("-fin", "--finalize", type=str, nargs="+",
                    action="store", default=[],
                    help="List some key=value pairs which are passed "
                         "as key=value to the networks finalize method.")
pasnet.add_argument("-v", "--verification", type=str,
                     action="store", default=None,
                     help="Regex for grouping the data set files into 'use for "
                          "verification' and 'use for training'.")
pasnet.add_argument("-vs", "--verification-split", type=float,
                     action="store", default=0.0,
                     help="Fraction of the trainings data to split off for the "
                          "verification data.")
pasnet.add_argument("--forget", type=float,
                     action="store", default=0.0,
                     help="Probability of skipping to load entries of the "
                           "verification data")
pasnet.add_argument("-p", "--prefix", type=str,
                     action="store", default="",
                     help="Prefix to add in front of analysis outputs")
                          
pasnet.add_argument("-t", "--train", action="store_true",
                     help="Flag indicating that training should be executed.")
pasnet.add_argument("-e", "--evaluate", action="store_true",
                     help="Flag indicating that evaluation should be executed.")
pasnet.add_argument("--sort_problems", action="store_true",
                     help="Flag showing whether the problems should be sorted by difficulty. "
                          "Expects certain problem file naming.")
pasnet.add_argument("--epochs", type=int,
                     action="store", default=10,
                     help="Number of epochs of sampling and training on each problem during training.")
pasnet.add_argument("--problem_epochs", type=int,
                     action="store", default=3,
                     help="Number of consecutive epochs of sampling and training on each problem during "
                          "the entire training process before going to the next problem (This is important "
                          "because whenever a new problem is used trained the keras ASNet model has "
                          "to be built and cannot be stored -> inefficient to always iterate over problems).")
pasnet.add_argument("--train_epochs", type=int,
                     action="store", default=100,
                     help="Number of epochs of training after sampling for one problem.")
pasnet.add_argument("--time_limit", type=int,
                     action="store", default=7200,
                     help="Time limitation for training only in seconds (default 2h).")
pasnet.add_argument("--no_sample_accumulating", action="store_true",
                     help="Flag deactivating accumulation of samples for one problem (activating "
                          "deletion in bridge after sample data extraction.)")
pasnet.add_argument("--delete", action="store_true",
                     help="If flag ist set then all sample and network data is deleted "
                          "after the entire training/ evaluation process.")
pasnet.add_argument("--print_all", action="store_true",
                     help="Show all intermediate prints.")

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
pasnet.add_argument("--use_only_goal_teacher_paths", action="store_true",
                    help="Flag indicating whether only paths/ trajectories of "
                         "the teacher search reaching a goal state should be "
                         "sampled in the ASNet Sampling.")
pasnet.add_argument("--use_no_teacher_search", action="store_true",
                    help="Flag indicating whether the teacher search should "
                         "be deactivated during sampling.")

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
                    action="store", default=0.25,
                    help="Dropout rate used in every intermediate node (= probability to "
                         "deactivate intermediate nodes)")
pasnet.add_argument("-ki", "--kernel_initializer", type=str,
                    action="store", default='glorot_normal',
                    help="Initializer to be used for all weight matrices/ kernels of all "
                         "modules")
pasnet.add_argument("-bi", "--bias_initializer", type=str,
                    action="store", default='zeros',
                    help="Initializer to be used for all bias vectors of all modules")
pasnet.add_argument("--regularizer_value", type=float,
                    action="store", default=0.001,
                    help="Regularization value used for L2 regularization applied to all "
                         "weights (including bias vectors)")
pasnet.add_argument("--extra_input_features", type=str,
                    action="store", default=None,
                    help="Additional input features per action. This involves additional "
                         "heuristic input features from landmarks or None. The options for "
                         "these landmark inputs are 'landmarks' or 'binary_landmarks'.")
pasnet.add_argument("-opt", "--optimizer", type=str,
                    action="store", default='adam',
                    help="Optimizer to be used during training (usually Adam with "
                         "potentially adapted learning rate).")
pasnet.add_argument("-lr", "--learning_rate", type=float,
                    action="store", default=0.001,
                    help="Learning rate used for (Adam or SGD) Optimizer.")
pasnet.add_argument("--loss_just_opt", action="store_true",
                    help="Flag stating that the custom loss only considering optimal actions "
                         "should be used")

arguments = set()
for action in pasnet._actions:
    for key in action.option_strings:
        arguments.add(key)


def deactivate_prints():
    """
    Preventing all print outputs until reactivate_prints() call
    """
    sys.stdout = open(os.devnull, 'w')

def reactivate_prints():
    """
    Reactivate print outputs after deactivate_prints() call
    """
    sys.stdout = sys.__stdout__

def execute_cmd_without_output(cmd):
    """
    Execute cmd subprocess without outputs
    :param cmd: command to execute
    :return: retcode of subprocess call
    """
    FNULL = open(os.devnull, 'w')
    retcode = subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    return retcode

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


def create_pddl_task(options, domain_path, problem_path):
    """
    Reads, translates the pddl_task and simplifies it to remove
    unused/ unreachable predicates etc.
    Also creates the corresponding task_meta file
    :return: pddl_task, task_meta
    """
    print("Processing PDDL Translation.")
    if not options.print_all:
        deactivate_prints()
    pddl_task, sas_task = translate([domain_path, problem_path])
    prog = pddl_to_prolog(pddl_task)
    model = compute_model(prog)
    _, grounded_predicates, propositional_actions, _, _ = instantiate(pddl_task, model)
    fluent_predicates = get_fluent_predicates(pddl_task, model)
    pddl_task.simplify(fluent_predicates)

    if not options.print_all:
        reactivate_prints()

    print("Computing task meta information.")
    task_meta = ProblemMeta(pddl_task, sas_task, propositional_actions, grounded_predicates)
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
        - regularizer_value: regularization value used for L2 regularization applied
                             to all weights (included bias vectors)
        - optimizer: optimizer to be used during training
        - learning_rate: learning rate to use for (Adam) Optimizer
        - loss_just_opt: use loss function only considering opt actions
    :param extra_input_size: size of additional input features per action
                             This usually involves additional heuristic input features
                             like values indicating landmark values for actions (as used
                             in the paper of Toyer et al. about ASNets)
    :param weights_path: path to a .h5 file with weights of previous ASNet to be reused 
                         (or None if no weights should be reused)
    :return: Keras ASNet Model
    """
    print("Building the ASNet keras model...")
    asnet_builder = ASNet_Model_Builder(task_meta, options.print_all)
    asnet_model = asnet_builder.build_asnet_keras_model(options.number_of_layers,
                                                        options.hidden_rep_size,
                                                        options.activation,
                                                        options.dropout,
                                                        options.kernel_initializer,
                                                        options.bias_initializer,
                                                        options.regularizer_value,
                                                        extra_input_size)
    if options.optimizer == 'adam':
        optimizer = Adam(lr=options.learning_rate)
    elif options.optimizer == 'sgd':
        optimizer = SGD(lr=options.learning_rate, momentum=0.7)
    else:
        optimizer = options.optimizer
    if options.loss_just_opt:
        asnet_model.compile(loss=custom_binary_crossentropy, optimizer=optimizer)
    else:
        asnet_model.compile(loss=just_opt_custom_binary_crossentropy, optimizer=optimizer)
    if weights_path:
        print("Loading previous weights")
        asnet_model.load_weights(weights_path, by_name=True)
    return asnet_model


def prepare_and_construct_network_before_loading(options, extra_input_size, model):
    """
    Constructs KerasASNet Network for training etc. with the given model

    :param options: parser options for problem epochs
    :param extra_input_size: size for additional input features per action
    :param model: built keras asnet model
    :return: KerasASNet Network-object
    """
    return KerasASNet(extra_input_size=extra_input_size,
                      model=model,
                      epochs=options.train_epochs)


def sample(options, directory, domain_path, problem_path, extra_input_size):
    """
    Execute sampling search for asnets with given configuration

    :param options: parser options for sampling search configuration
    :param directory: currently processed directory
    :param domain_path: path to domain file
    :param problem_path: path to problem file to execute sampling on
    :param extra_input_size: size for additional input features per action
    :return: retcode of call
    """
    cmd = ['python', 'fast-downward.py',
                "--build", options.build, domain_path, problem_path,
                '--search', 'asnet_sampling_search(search=' + options.teacher_search +
                            ', use_non_goal_teacher_paths=' + str(not options.use_only_goal_teacher_paths) +
                            ', use_teacher_search=' + str(not options.use_no_teacher_search) +
                            ', network_search=policysearch(p=np(network=asnet(path=' + str(os.path.join(directory, 'asnet.pb')) +
                            ', extra_input_size=' + str(extra_input_size) + ')),trajectory_limit=' + str(options.trajectory_limit) + ')' +
                            ', target=' + os.path.join(directory, "sample.data") + ')']
    print('Running sampling search for ' + problem_path + '...')
    try:
        if options.print_all:
            retcode = subprocess.call(cmd)
        else:
            retcode = execute_cmd_without_output(cmd)
        if retcode < 0:
            print("Sampling search was terminated by signal", -retcode, file=sys.stderr)
    except (OSError, SystemExit) as e:
        print("Sampling search terminated with OSError or Systemexit", e, file=sys.stderr)
        return 0
    return retcode


def fd_evaluate(options, domain_path, problem_path, network_path, extra_input_size):
    """
    Execute policysearch with network policy of asnet.pb with given configuration

    :param options: parser options for search configuration
    :param domain_path: path to domain file
    :param problem_path: path to problem file to execute search on
    :param network_path: path to asnet.pb network file to load for network policy
    :param extra_input_size: size for additional input features per action
    :return:
    """
    cmd = ['python', 'fast-downward.py',
                "--build", options.build, domain_path, problem_path,
                '--search', 'policysearch(p=np(network=asnet(path=' + str(network_path) +
                ', extra_input_size=' + str(extra_input_size) + ')))']
    print('Running network policy search for ' + problem_path + '...')
    # run process
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # runs command
    sys.stdout.flush()
    solution_found = None
    for line in iter(p.stdout.readline, b''):
        sys.stdout.flush()
        if options.print_all:
            print(line.decode("utf-8").rstrip())
        if b"Solution found!" in line.rstrip():
            solution_found = True
        elif b"No solution - FAILED" in line.rstrip():
            solution_found = False
    if solution_found is None:
        raise ValueError("Neither solution found nor failed solution result detected after fast-downward search.")
    return solution_found


def exploration_explored_goal(sample_path):
    """
    Check whether the network exploration of the sampling creating sample_file
    reached a goal state

    :param sample_path: path to sample.data file including all sampling information
    :return: true if network exploration reached a goal state and false otherwise
    """
    sample_file = open(sample_path, 'r')
    sample_lines = [l.strip() for l in sample_file.readlines()]
    # reversed order so that last sampling result is met first
    # (otherwise if sampling results are accumulated the old results would be met first)
    for line in reversed(sample_lines):
        if line == "GOAL_EXPLORATION":
            print("EXPLORATION SOLVED THE PROBLEM")
            sample_file.close()
            return True
        elif line == "NO_GOAL_EXPLORATION":
            print("EXPLORATION DID NOT SOLVE THE PROBLEM")
            sample_file.close()
            return False
    raise ValueError("Sample File %s did not contain information on network exploration!" % sample_path)


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
    print("Loading sampling data ...")
    bridge = ASNetSampleBridge(sample_path=os.path.join(directory, "sample.data"),
                               domain=domain_path, prune=True, forget=options.forget,
                               skip=options.skip, extra_input_size=extra_input_size,
                               delete=options.no_sample_accumulating)

    dtrain, dtest = None, None
    # We create a test set
    if options.verification is not None:
        test_problem_filter = list(options.problem_filter)
        test_problem_filter.append(options.verification)

        dir_samp = IterableFileSampler(bridge, [problem_path], merge=True)
        dir_samp.initialize()
        dtest = dir_samp.sample()
        dir_samp.finalize()

    # We create the training set
    bridge._forget = 0.0
    dir_samp = IterableFileSampler(bridge, [problem_path], merge=True)
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

def duration_format(duration):
    """
    :param duration: time duration in seconds
    :return: string resembling duration with hours, minutes and seconds
    """
    seconds = int(duration)
    minutes = int(duration / 60)
    hours = int(minutes / 60)
    if hours > 0:
        minutes = minutes - hours * 60
        seconds = seconds - hours * 3600 - minutes * 60
        return "%dh %dmin %ds" % (hours, minutes, seconds)
    elif minutes > 0:
        seconds = seconds - minutes * 60
        return "%dmin %ds" % (minutes, seconds)
    else:
        return "%ds" % seconds



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
    begin_train_time = time.time()
    start_time = begin_train_time
    if options.verification is not None:
        options.verification = re.compile(options.verification)
    
    options.initialize = parse_key_value_pairs_to_kwargs(options.initialize)
    options.finalize = parse_key_value_pairs_to_kwargs(options.finalize)

    # set extra_input_size per action depending on extra_input_features
    if options.extra_input_features is None:
        extra_input_size = 0
    elif options.extra_input_features == "landmarks":
        extra_input_size = 2
    elif options.extra_input_features == "binary_landmarks":
        extra_input_size = 3
    else:
        raise ValueError("Invalid extra input feature value: %s" % options.extra_input_features)

    start_time = timing(start_time, "Parsing time: %ss")

    # remove old sample data
    if os.path.isfile(os.path.join(directory, "sample.data")):
        os.remove(os.path.join(directory, "sample.data"))

    # path to previous weight file (or None if not existing)
    previous_asnet_weights = options.weights
    if previous_asnet_weights is not None:
        assert os.path.isfile(previous_asnet_weights), "The given weight file was not found"

    # asnet model to save weights after final epoch for evaluation
    asnet_model = None

    # bool flag to interrupt further epochs
    stop_training = False

    # success_rate = percentage of network exploration runs during sampling which reach a
    #                goal state
    # best success rate over all epochs
    best_success_rate = -1
    # success rate of current epoch
    current_success_rate = 0.0

    # incremented if current_success_rate did not improve over best_success_rate by more
    # than 0.01%
    epochs_since_improvement = 0

    epoch_counter = 0
    
    end_training = False
    while epoch_counter < options.epochs and (current_success_rate < 95 or epochs_since_improvement < 3):
        print()
        print("Starting epoch %d" % (epoch_counter + 1))

        current_time = time.time()
        training_time = current_time - begin_train_time
        duration_string = duration_format(training_time)
        print("Training already takes %s" % duration_string)

        # number of explorations reaching a goal state
        solved_explorations = 0
        # number of total explorations executed
        executed_explorations = 0
        for problem_index, problem_path in enumerate(problem_list):
            print()
            print("Processing problem file " + str(problem_path) + " ("
                    + str(problem_index + 1) + "/ " + str(len(problem_list)) + ")")
            if options.dry:
                continue

            # number of explorations reaching a goal state for this problem
            solved_explorations_problem = 0
            # number of total explorations executed for this problem
            executed_explorations_problem = 0

            # clear keras session (for new problem)
            K.clear_session()

            # build the model for the problem
            print("Building keras ASNet model")
            _, task_meta = create_pddl_task(options, domain_path, problem_path)
            start_time = timing(start_time, "PDDL translation time: %ss")
            asnet_model = create_asnet_model(task_meta, options, extra_input_size, previous_asnet_weights)
            start_time = timing(start_time, "Keras model creation time: %ss")

            # construct and prepare training network class
            asnet = prepare_and_construct_network_before_loading(options, extra_input_size, asnet_model)
            start_time = timing(start_time, "Building and preparing the network time: %ss")

            # initialize training network class
            asnet.initialize(None, **options.initialize)
            start_time = timing(start_time, "Network initialization time: %ss")

            print("Starting problem epochs in epoch %d" % (epoch_counter + 1))
            problem_epoch = 0
            while problem_epoch < options.problem_epochs:
                print()

                # check if time limit for training is reached
                training_time = current_time - begin_train_time
                duration_string = duration_format(options.time_limit)
                if training_time > options.time_limit:
                    print("Training is taking over " + duration_string + " -> training is stopped now!")
                    end_training = True
                    break

                print("Starting problem epoch %d / %d" % (problem_epoch + 1, options.problem_epochs))
                # if previous asnet.pb file exists -> remove
                if os.path.isfile(os.path.join(directory, "asnet.pb")):
                    os.remove(os.path.join(directory, "asnet.pb"))

                # store protobuf network file for fast-downward sampling
                asnet._store(os.path.join(directory, "asnet"), [NetworkFormat.protobuf])
            
                start_time = timing(start_time, "Preparing and storing of the network time: %ss")

                # build ASNetSamplingSearch command and execute for sampling -> saves samples in sample.data
                if not os.path.isfile(os.path.join(directory, "sample.data")):
                    open(os.path.join(directory, "sample.data"), 'a')
                retcode = sample(options, directory, domain_path, problem_path, extra_input_size)
                print("Sampling Search retcode:", retcode)
                if retcode < 0:
                    raise ValueError("Training process is stopped due to error signal of sampling search call with retcode %d" % retcode)
                start_time = timing(start_time, "Sampling search time: %ss")

                # extract exploration information for the given problem sampling (out of sample.data)
                if exploration_explored_goal(os.path.join(directory, "sample.data")):
                    solved_explorations += 1
                    solved_explorations_problem += 1
                executed_explorations += 1
                executed_explorations_problem += 1

                dtrain, dtest = load_data(options, directory, domain_path, problem_path, extra_input_size)

                start_time = timing(start_time, "Loading data time: %ss")

                asnet.train(dtrain, dtest)
                start_time = timing(start_time, "Network training time: %ss")

                if dtest:
                    asnet.evaluate(dtest)
                    start_time = timing(start_time, "Network evaluation time: %ss")

                    asnet.analyse(prefix=options.prefix)
                    start_time = timing(start_time, "Network analysis time: %ss")

                problem_epoch += 1

            # training time limit reached -> end early
            if end_training:
                break

            # after last problem epoch -> finalize network
            asnet.finalize(**options.finalize)
            _ = timing(start_time, "Network finalization time: %ss")

            # saving weights for next problem instance in last problem epoch
            print("Storing weights in %s" % os.path.join(directory, "asnet_weights.h5"))
            if previous_asnet_weights is not None:
                os.remove(previous_asnet_weights)
            asnet._store_weights(os.path.join(directory, "asnet_weights.h5"))
            previous_asnet_weights = os.path.join(directory, "asnet_weights.h5")

            # remove existing sampling data from last problem
            if os.path.isfile(os.path.join(directory, "sample.data")):
                os.remove(os.path.join(directory, "sample.data"))

            print("%d / %d network explorations were successfull for this problem" % (solved_explorations_problem, executed_explorations_problem))

        # time limit reached -> end early
        if end_training:
            break

        # compute percentage of successfull explorations in epoch
        current_success_rate = (solved_explorations / executed_explorations) * 100
        print("Epochs success rate: %d" % current_success_rate)

        # update best_success_rate & epochs_since_improvement
        if current_success_rate < best_success_rate + 0.01:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        if current_success_rate > best_success_rate:
            best_success_rate = current_success_rate

        epoch_counter+= 1
        
    if (current_success_rate >= 95 and epochs_since_improvement >= 3):
        print("EARLY TRAINING TERMINATION:")
        print("The success rate is with %d%% >= 95%% and there were no significant improvements of the success rate for %d epochs." % (current_success_rate, epochs_since_improvement))

    # delete remaining weights and model and sample data
    if options.delete:
        print("Deleting network files and sample data")
        if os.path.isfile(os.path.join(directory, "asnet.pb")):
            os.remove(os.path.join(directory, "asnet.pb"))
        if os.path.isfile(os.path.join(directory, "sample.data")):
            os.remove(os.path.join(directory, "sample.data"))
        if os.path.isfile(os.path.join(directory, "asnet_weights.h5")):
            os.remove(os.path.join(directory, "asnet_weights.h5"))

    # save final model weights
    print("Saving final weights in %s" % os.path.join(directory, "asnet_final_weights.h5"))
    if asnet_model is not None:
        asnet_model.save_weights(os.path.join(directory, "asnet_final_weights.h5"))

    _ = timing(begin_train_time, "Entire training time: %ss")


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
    begin_eval_time = time.time()
    start_time = begin_eval_time

    options.initialize = parse_key_value_pairs_to_kwargs(options.initialize)
    options.finalize = parse_key_value_pairs_to_kwargs(options.finalize)

    # set extra_input_size per action depending on extra_input_features
    if options.extra_input_features is None:
        extra_input_size = 0
    elif options.extra_input_features == "landmarks":
        extra_input_size = 2
    elif options.extra_input_features == "binary_landmarks":
        extra_input_size = 3
    else:
        raise ValueError("Invalid extra input feature value: %s" % options.extra_input_features)

    start_time = timing(start_time, "Parsing time: %ss")

    # path to previous weight file (or None if not existing)
    asnet_weights = options.weights
    if asnet_weights is None:
        print("WARNING: Evaluation is started without given weights! Will use randomly initialized weights.")

    # solved evaluation counter
    solved_problems = 0

    for problem_index, problem_path in enumerate(problem_list):
        print("Evaluating problem file " + str(problem_path) + " ("
                + str(problem_index + 1) + "/ " + str(len(problem_list)) + ")")
        if options.dry:
            continue
        problem_eval_time = time.time()
        # clear tensorflow session to remove computational graph of last model
        K.clear_session()

        _, task_meta = create_pddl_task(options, domain_path, problem_path)
        start_time = timing(start_time, "PDDL translation time: %ss")
        asnet_model = create_asnet_model(task_meta, options, extra_input_size, asnet_weights)
        start_time = timing(start_time, "Keras model creation time: %ss")

        # store protobuf network file for fast-downward search
        network_file = os.path.join(directory, "asnet.pb")
        if os.path.isfile(network_file):
            os.remove(network_file)
        store_keras_model_as_protobuf(asnet_model, file=network_file)
        start_time = timing(start_time, "Network saving time: %ss")

        # execute fast-downward network search
        solved_problem = fd_evaluate(options, domain_path, problem_path, network_file, extra_input_size)
        start_time = timing(start_time, "Network search evaluation time: %ss")
        if solved_problem:
            solved_problems += 1
            print("Problem was SOLVED")
        else:
            print("Problem was NOT SOLVED")
        _ = timing(problem_eval_time, "Problem evaluation time for problem " +problem_path + ": %ss")

    # delete remaining model file
    if options.delete:
        print("Deleting network file")
        if os.path.isfile(os.path.join(directory, "asnet.pb")):
            os.remove(os.path.join(directory, "asnet.pb"))

    print("Solved %d / %d problems" % (solved_problems, len(problem_list)))
    _ = timing(begin_eval_time, "Entire evaluation time: %ss")


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
        if options.evaluate:
            options.weights = os.path.join(directory, "asnet_final_weights.h5")
            evaluate(options, directory, domain_path, problem_list)
    elif options.evaluate:
        evaluate(options, directory, domain_path, problem_list)
    else:
        raise ValueError("Neither evaluate nor train flag set.")
                

if __name__ == "__main__":
    main(sys.argv)
