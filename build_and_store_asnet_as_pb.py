#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from keras.optimizers import Adam

sys.path.append("network_models/asnets")
from problem_meta import ProblemMeta
from asnet_keras_model import ASNet_Model_Builder
from losses import custom_binary_crossentropy, just_opt_custom_binary_crossentropy

from src.translate.translator import main as translate
from src.translate.normalize import normalize
from src.translate.instantiate import instantiate, get_fluent_predicates
from src.translate.build_model import compute_model
from src.translate.pddl_to_prolog import translate as pddl_to_prolog

from src.training.networks.keras_networks.keras_tools import store_keras_model_as_protobuf


def create_pddl_task(domain_path, problem_path):
    """
    Reads, translates the pddl_task and simplifies it to remove
    unused/ unreachable predicates etc.
    Also creates the corresponding task_meta file
    :return: pddl_task, task_meta
    """
    pddl_task, sas_task = translate([domain_path, problem_path])
    normalize(pddl_task)
    prog = pddl_to_prolog(pddl_task)
    model = compute_model(prog)
    _, grounded_predicates, propositional_actions, _, _ = instantiate(pddl_task, model)
    fluent_predicates = get_fluent_predicates(pddl_task, model)
    pddl_task.simplify(fluent_predicates)

    print("Computing task meta information.")
    task_meta = ProblemMeta(pddl_task, sas_task, propositional_actions, grounded_predicates)
    return task_meta


def build_asnet_model(task_meta, number_of_layers, loss_just_opt, weights_path):
    asnet_builder = ASNet_Model_Builder(task_meta, False)
    asnet_model = asnet_builder.build_asnet_keras_model(num_layers=number_of_layers)
    optimizer = Adam(lr=0.001)
    if loss_just_opt:
        asnet_model.compile(loss=custom_binary_crossentropy, optimizer=optimizer)
    else:
        asnet_model.compile(loss=just_opt_custom_binary_crossentropy, optimizer=optimizer)
    asnet_model.load_weights(weights_path, by_name=True)
    return asnet_model


def main(argv):
    """
    Usage: ./build_and_store_asnet_as_pb.py <domain_path> <problem_path> <number_of_layers> <loss_just_opt> <weights_path> <network_path>
    """
    if not len(argv) == 7:
        print("Usage: ./build_and_store_asnet_as_pb.py <domain_path> <problem_path> <number_of_layers> <loss_just_opt> <weights_path> <network_path>")
        sys.exit(1)
    domain_path = argv[1]
    problem_path = argv[2]
    number_of_layers = int(argv[3])
    loss_just_opt = bool(argv[4])
    weights_path = argv[5]
    network_path = argv[6]

    task_meta = create_pddl_task(domain_path, problem_path)
    asnet_model = build_asnet_model(task_meta, number_of_layers, loss_just_opt, weights_path)
    store_keras_model_as_protobuf(asnet_model, file=network_path)


if __name__ == "__main__":
    main(sys.argv)
