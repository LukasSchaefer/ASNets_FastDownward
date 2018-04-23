#!/bin/bash


../../fast-downward.py --build debug64dynamic p02.pddl --search "eager_greedy([nh(network=probnet(type=classification,path=best_model.pb, input_layer=dense_1_input, \real_case\output_layer=output_layer/Softmax\default\))], cost_type=ONE)"
