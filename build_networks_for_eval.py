#! /usr/bin/env python

import os
import sys
import time
from build_and_store_asnet_as_pb import main as build_pb
import keras.backend as K

asnetsfastdownward_dir = os.path.dirname(os.path.realpath(__file__))

# domains = ['blocksworld', 'floortile', 'tyreworld', 'sokoban', 'parcprinter', 'hanoi', 'elevator', 'turnandopen']
# domains = ['elevator', 'turnandopen']
domains = ['hanoi']
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
        if domain == 'parcprinter':
            domain_benchmark_dir = os.path.join(asnetsfastdownward_dir, 'benchmarks/' + domain + '/diff_1')
        else:
            domain_benchmark_dir = os.path.join(asnetsfastdownward_dir, 'benchmarks/' + domain)
        domain_file_path = os.path.join(domain_benchmark_dir, 'domain.pddl')
        files = [f for f in os.listdir(domain_benchmark_dir) if os.path.isfile(os.path.join(domain_benchmark_dir, f))]
        for problem_name in files:
            if not problem_name.endswith('.pddl'):
                continue

            problem_path = os.path.join(domain_benchmark_dir, problem_name)
            problem_name = problem_name[:-5]

            if problem_name == 'domain':
                continue

            # file should be a problem of the given domain
            network_path = os.path.join(asnetsfastdownward_dir, 'evaluation/network_runs/evaluation/protobuf_networks/' + domain + '/' + conf + '/' + problem_name + '.pb')
            if os.path.isfile(network_path):
                continue

            too_large_problems = ['d-24', 'd-25', 'd-26', 'd-27', 'd-28', 'd-29', 'd-30']
            if domain == 'elevator' and problem_name in too_large_problems:
                continue

            weights_path = os.path.join(asnetsfastdownward_dir, 'evaluation/network_runs/training/' + domain + '/' + conf + '/asnet_final_weights.h5')
            current_time = time.time()
            build_pb(['blub', domain_file_path, problem_path, layers, just_opt_loss, weights_path, network_path])
            network_build_time = time.time() - current_time

            log_path = os.path.join(asnetsfastdownward_dir, 'evaluation/network_runs/evaluation/protobuf_networks/' + domain + '/' + conf + '/' + problem_name + '.log')
            f = open(log_path, 'w')
            f.write("Model building & saving time: %f" % network_build_time)
            f.close()


if __name__ == "__main__":
    main(sys.argv)
