#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
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

def main(argv):
    """
    function call:
    python fast-asnet.py <domain.pddl> <problem.pddl>
    """
    _, task_meta = create_pddl_task(argv)
    print("Building the ASNet keras model...")
    asnet_builder = ASNet_Model_Builder(task_meta)
    asnet_model = asnet_builder.build_asnet_keras_model(1, dropout=0.25)
    asnet_model.compile(loss='mean_squared_error', optimizer='adam')
    print("Done building the model")
    header_size = np.asarray([layer.name.encode('utf8') for layer in asnet_model.layers]).nbytes
    print("Header size for h5 model file in bytes: " + str(header_size))
    asnet_model.save('asnet_model.h5')


if __name__ == "__main__":
    main(sys.argv)