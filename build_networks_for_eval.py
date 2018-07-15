#! /usr/bin/env python

import os
import sys
from build_and_store_asnet_as_pb import main as build_pb

asnetsfastdownward_dir = os.path.dirname(os.path.realpath(__file__))

domains = ['blocksworld', 'elevator', 'floortile', 'hanoi', 'parcprinter', 'sokoban', 'turnandopen', 'tyreworld']

configurations = {}
conf1 = ('False', '2', '"astar(lmcut(),transform=asnet_sampling_transform())"')
configurations['conf1'] = conf1
conf2 = ('False', '2', '"astar(add(),transform=asnet_sampling_transform())"')
configurations['conf2'] = conf2
conf3 = ('False', '2', '"ehc(ff(),transform=asnet_sampling_transform())"')
configurations['conf3'] = conf3
conf4 = ('False', '4', '"astar(lmcut(),transform=asnet_sampling_transform())"')
configurations['conf4'] = conf4
conf5 = ('False', '4', '"astar(add(),transform=asnet_sampling_transform())"')
configurations['conf5'] = conf5
conf6 = ('False', '4', '"ehc(ff(),transform=asnet_sampling_transform())"')
configurations['conf6'] = conf6


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 build_networks_for_eval.py confx")
        print("Possible configurations:")
        for conf, values in configurations.items():
            print("%s: just_opt_loss = %s, layers = %s, teacher_search = %s" % (conf, values[0], values[1], values[2]))
        sys.exit(1)

    conf = argv[1]
    if not conf in configurations.keys():
        print("Given configuration %s does not exist!" % conf)
    just_opt_loss, layers, teacher_search = configurations[conf]
    print("Using %s: just_opt_loss = %s, layers = %s, teacher_search = %s" % (conf, just_opt_loss, layers, teacher_search))

    for domain in domains:
        domain_benchmark_dir = os.path.join(asnetsfastdownward_dir, 'benchmarks/' + domain)
        domain_file_path = os.path.join(domain_benchmark_dir, 'domain.pddl')
        files = [f for f in os.listdir(domain_benchmark_dir) if os.path.isfile(os.path.join(domain_benchmark_dir, f))]
        for problem_name in files:
            if problem_name == 'domain.pddl':
                continue
            # file should be a problem of the given domain
            problem_path = os.path.join(domain_benchmark_dir, problem_name)
            problem_name = problem_name[:-5]

            weights_path = os.path.join(asnetsfastdownward_dir, 'evaluation/network_runs/training/' + domain + '/' + conf + '/asnet_final_weights.h5')
            network_path = os.path.join(asnetsfastdownward_dir, 'evaluation/network_runs/evaluation/protobuf_networks/' + domain + '/' + conf + '/' + problem_name + '.pb')

            build_pb(['blub', domain_file_path, problem_path, layers, just_opt_loss, weights_path, network_path])


if __name__ == "__main__":
    main(sys.argv)
