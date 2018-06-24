#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import ast
import h5py
import tensorflow as tf
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


def create_pddl_task(argv):
    """
    Reads, translates the pddl_task and simplifies it to remove
    unused/ unreachable predicates etc.
    Also creates the corresponding task_meta file
    :return: pddl_task, task_meta
    """
    if argv is None or len(argv) != 3:
        print("Usage: print_task <domain.pddl> <problem.pddl>")
        raise ValueError("You need to give 2 arguments!")
    else:
        print("Processing PDDL Translation.")
        pddl_task, _ = translate([argv[1],argv[2]])
        normalize(pddl_task)
        prog = pddl_to_prolog(pddl_task)
        model = compute_model(prog)
        _, grounded_predicates, propositional_actions, _, _ = instantiate(pddl_task, model)
        fluent_predicates = get_fluent_predicates(pddl_task, model)
        pddl_task.simplify(fluent_predicates)

        print("Computing task meta information.")
        task_meta = ProblemMeta(pddl_task, propositional_actions, grounded_predicates)
        print("Number of actions: " + str(len(pddl_task.actions)))
        print("Number of predicates: " + str(len(pddl_task.predicates)))
        print("Number of grounded predicates: " + str(len(task_meta.grounded_predicates)))
        print("Number of propositional actions: " + str(len(task_meta.propositional_actions)))
        return pddl_task, task_meta


def read_sample(line):
    """
    Extract single sample out of the corresponding line-string and returns
    dict of values corresponding to the sample informations in the format as explained below
    """
    fields = line.split(';')
    assert(len(fields) == 6, "Sample does not include all necessary information!")
    problem_hash = fields[0]
    goal_values = ast.literal_eval(fields[1])
    state_values = ast.literal_eval(fields[2])
    applicable_values = ast.literal_eval(fields[3])
    network_probs = ast.literal_eval(fields[4])
    opt_values = ast.literal_eval(fields[5])

    sample_dict = {}
    sample_dict['hash'] = problem_hash
    sample_dict['goals'] = np.asarray(goal_values)
    sample_dict['facts'] = np.asarray(state_values)
    sample_dict['applicable_actions'] = np.asarray(applicable_values)
    sample_dict['network_probs'] = np.asarray(network_probs)
    sample_dict['opt_values'] = np.asarray(opt_values)
    return sample_dict


def extract_samples():
    """
    Reading samples.txt file and build up list of samples with each
    sample entry corresponding to a dict of values in the following format
    ["hash": <PROBLEM_HASH>, "goals": <GOAL_VALUES>, "facts": <STATE_VALUES>, "applicable_actions": <APPLICABLE_VALUES>,
     "network_probs": <NETWORK_PROBS>, "opt_values": <OPT_VALUES>]
    with
        - <PROBLEM_HASH>: hash-value indicating the problem instance
        - <GOAL_VALUES>: binary value for every fact indicating whether the fact is part of the goal. Values
                         are ordered lexicographically by fact-names in a "," separated list form
        - <STATE_VALUES>: binary values (0,1) indicating whether a fact is true. Values are ordered lexicographically
                          by fact-names and are all "," separated in a list and are given for every fact (e.g. [0,1,1,0]) 
        - <APPLICABLE_VALUES>: binary values indicating whether an action is applicable in the current state.
                               Ordering again is lexicographically by action-names and the values are in a ","
                               separated list form for all actions.
        - <NETWORK_PROBS>: float value for every action representing the probability to choose the action in the
                           state according to the network policy. Values are again ordered lexicographically by
                           action-names and values are "," separated, e.g. [0.0,0.1,0.3,0.6]
        - <OPT_VALUES>: binary value for each action indicating whether the action starts a found plan according to
                        the teacher-search. Again ordered lexicographically by action-names in a "," separated list.
    """
    samples = []
    with open('samples.txt') as sample_file:
        for line in sample_file.readlines():
            if line.startswith("#"):
                # comment -> skip
                continue
            else:
                samples.append(read_sample(line))
    return samples


def prediction(network, sample):
    """
    Computes and return prediction for the network for the given sample-dict
    as a numpy-array
    """
    inputs = [sample['facts'], sample['goals'], sample['applicable_values']]
    return network.predict(inputs)


def main(argv):
        _, task_meta = create_pddl_task(argv)
        print("Building the ASNet keras model...")
        asnet_builder = ASNet_Model_Builder(task_meta)
        asnet_model = asnet_builder.build_asnet_keras_model(1, dropout=0.25)
        asnet_model.compile(loss='mean_squared_error', optimizer='adam')
        print("Done building the model")
        asnet_model.save('asnet_model.h5')

        samples = extract_samples()


if __name__ == "__main__":
    main(sys.argv)