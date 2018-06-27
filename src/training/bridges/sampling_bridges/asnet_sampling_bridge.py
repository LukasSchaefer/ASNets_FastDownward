from . import SamplerBridge

from ...data import SizeBatchData 

from ... import main_register
from ... import parser
from ... import parser_tools as parset
from ... import SampleBatchData

from ...misc import hasher
from ...misc import StreamContext

import os
import random
import ast
import sys
import numpy as np

# Exception for corrupted samples
class DataCorruptedError(Exception):
    pass


def load_samples(sample_path, data_container, prune=True,
                 skip=True, forget=0.0, extra_input_size=0):
    """
    Loads all samples from stream_context into data_container

    :param sample_file: path to file containing the samples
    :param data_container: data_container to save the samples information into
    :param prune: boolean value indicating whether duplicate entries are pruned
    :param skip: boolean value indicating whether non-existing sample files should
                 simply be skipped/ ignored (otherwise an exception will be thrown)
    :param forget: probability (value should be in range [0.0, 1.0]) to forget/ skip
                   sample states
    :param extra_input_size: input size of additional input features per action
                             (0 -> no additional input features)
    """

    if not os.path.exists(sample_path):
        if skip:
            return
        else:
            raise FileNotFoundError("A sample file to load does not exist:"
                                    + str(sample_path))
    load_asnet_sample_data(
        path_read=sample_path, prune=prune,
        data_container=data_container,
        forget=forget,
        extra_input_size=extra_input_size)


def load_asnet_sample_data(path_read, prune=True,
                           data_container=None,
                           delete=False, forget=0.0,
                           extra_input_size=0):
    """
    loads the samples in path_read file and saves these in the given container

    :param path_read: Path to the file containing the samples to load
    :param prune: If true prunes duplicate entries (the meta information is
                  except for the type attribute ignored)
    :param data_container: Data gathering object for the loaded entries. Object
                           requires an add(entry, type) method
                           (e.g. SizeBatchData). If None is given, the adding
                           is skipped.
    :param delete: Deletes path_input at the end of this method
    :param forget: probability (value should be in range [0.0, 1.0]) to forget/ skip
                   sample states
    :param extra_input_size: input size of additional input features per action
                             (0 -> no additional input features)
    :return:
    """
    # hash set for duplicate pruning
    old_hashs = set() if prune else None

    # read sample file
    with open(path_read, "r") as src:
        # each line (potentially) corresponds to a sample
        for line in src:
            if forget != 0.0 and random.random() < forget:
                continue
            # load sample entry into container
            load_sample_line(line, data_container, old_hashs, extra_input_size)

        if delete:
            os.remove(path_read)


def load_sample_line(line, data_container, old_hashs, extra_input_size):
    """
    loading one (potential) sample line out of sample file into data_container

    sample entries have the following format:
    <PROBLEM_HASH>; <GOAL_VALUES>; <STATE_VALUES>; <APPLICABLE_VALUES>; <OPT_VALUES>(; <ADDITIONAL_INPUT_FEATURES>)

        - <PROBLEM_HASH>: hash-value indicating the problem instance
        - <GOAL_VALUES>: binary value for every fact indicating whether the fact is part of the goal. Values
                         are ordered lexicographically by fact-names in a "," separated list form
        - <STATE_VALUES>: binary values (0,1) indicating whether a fact is true. Values are ordered lexicographically
                          by fact-names and are all "," separated in a list and are given for every fact (e.g. [0,1,1,0]) 
        - <APPLICABLE_VALUES>: binary values indicating whether an action is applicable in the current state.
                               Ordering again is lexicographically by action-names and the values are in a ","
                               separated list form for all actions.
        - <OPT_VALUES>: binary value for each action indicating whether the action starts a found plan according to
                        the teacher-search. Again ordered lexicographically by action-names in a "," separated list.
        - <ADDITIONAL_INPUT_FEATURES>: optional additional input features for each action

    these are then saved in data_container as lists of following format:
    [<PROBLEM_HASH>, <GOAL_VALUES>, <STATE_VALUES>, <APPLICABLE_VALUES>, <OPT_VALUES>(, <ADDITIONAL_INPUT_FEATURES>)]

    :param line: line corresponding to comment (starts with #) or sample
    :param data_container: Data gathering object for the loaded entries. Object
                           requires an add(entry, type) method
                           (e.g. SizeBatchData). If None is given, the adding
                           is skipped.
    :param old_hashs: set of hash values for already found samples to prune
                      duplicates or None if no pruning is applied
    :param extra_input_size: input size of additional input features per action
                             (0 -> no additional input features)
    :return: 
    """
    line = line.strip()
    if line.startswith("#") or line == "":
        # comment or empty line -> nothing to read
        return

    if old_hashs is not None:
        line_hash = hash(line)
        if line_hash in old_hashs:
            # line is a duplicate
            return
        else:
            old_hashs.add(line_hash)

    data = [x.strip() for x in line.split(";")]

    if extra_input_size > 0:
        expected_fields = 6
    else:
        expected_fields = 5

    if len(data) != expected_fields:
        raise DataCorruptedError("Sample entry must have " + str(expected_fields) + " fields seperated by ';' but had: "
                                 + str(len(data)))

    # extract all data fields
    problem_hash = data[0]
    goal_values = [ast.literal_eval(data[1])]
    state_values = [ast.literal_eval(data[2])]
    applicable_values = [ast.literal_eval(data[3])]
    opt_values = [ast.literal_eval(data[4])]

    sample_list = [problem_hash,
                   np.array(goal_values, dtype=np.int32),
                   np.array(state_values, dtype=np.int32),
                   np.array(applicable_values, dtype=np.int32),
                   np.array(opt_values, dtype=np.int32)]

    if extra_input_size > 0:
        additional_input_values = [ast.literal_eval(data[5])]
        sample_list.append(np.array(additional_input_values, dtype=np.int32))

    if data_container is not None:
        data_container.add(sample_list)


class ASNetSampleBridge(SamplerBridge):
    arguments = parset.ClassArguments('ASNetSampleBridge',
        SamplerBridge.arguments,
        ("prune", True, True, parser.convert_bool, "Prune duplicate samples"),
        ("skip", True, True, parser.convert_bool, "Skip problem if no samples exists, else raise error"),
        ("provide", True, False, parser.convert_bool),
        ("extra_input_size", True, 0, parser.convert_int_or_inf, "Input size of additional input features"),
        ("sample_path", True, "sample.data", str),
        ("delete", True, True, parser.convert_bool, "Delete Sampling file after extraction"),
        order=["streams", "prune", "forget", "skip",
             "tmp_dir", "provide", "extra_input_size",
             "sample_path", "delete", "domain",
             "makedir", "environment", "id"]
)

    def __init__(self, streams=None, prune=True, forget=0.0, skip=True,
                 tmp_dir=None, provide=False, extra_input_size=0, sample_path="sample.data",
                 delete=True, domain=None, makedir=False, environment=None, id=None):
        SamplerBridge.__init__(self, tmp_dir, streams, provide, forget,
                               domain, makedir, environment, id)

        self._prune = prune
        self._skip = skip
        self._extra_input_size = extra_input_size
        self._sample_path = sample_path
        self._delete = delete
        if self._provide:
            error_message = "The 'provide' parameter has no effect on the ASNetSampleBridge."
            if sys.version_info[0] < 3:
                print >> sys.stderr, error_message
            else:
                print(error_message, file=sys.stderr)

    def _initialize(self):
        pass


    def _sample(self, path_problem, path_dir_tmp, path_domain, data_container):
        data_container = (SizeBatchData(nb_fields=6, meta=path_problem,
                                        pruning=(hasher.list_hasher if self._prune else None))
                          if data_container is None else data_container)

        load_samples(self._sample_path,
                     data_container=data_container,
                     prune=self._prune,
                     skip=self._skip,
                     forget=self._forget,
                     extra_input_size=self._extra_input_size)

        if self._delete:
            os.remove(self._sample_path)

        return data_container

    def _finalize(self):
        pass

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  ASNetSampleBridge)

main_register.append_register(ASNetSampleBridge, "asnetbridge")