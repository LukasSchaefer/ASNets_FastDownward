#! /usr/bin/env python

# adds number of groundings to protobuf network creation logs (with time)
# + size of protobuf file (if present)

import os
import sys

from src.translate.translator import main as translate

asnetsfastdownward_dir = os.path.dirname(os.path.realpath(__file__))


def compute_number_of_groundings(pddl_domain_path, pddl_prob_path):
    _, sas_task = translate([pddl_domain_path, pddl_prob_path])
    number_of_groundings = len(sas_task.operators)
    number_of_groundings += sum(sas_task.variables.ranges)
    return '\nNumber of groundings: %d' % number_of_groundings


def compute_pb_network_size(log_path):
    protobuf_path = log_path[:-3] + 'pb'
    if os.path.isfile(protobuf_path):
        # if protobuf file is present add size of protobuf file
        # size in Byte
        size = os.stat(protobuf_path).st_size
        # size in MB
        size = (size / 1024) / 1024
        size_line = '\nProtobuf network size: %dMB' % size
    else:
        size_line = ''
    return size_line


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 build_networks_for_eval.py <path/to/protobuf/networks_dir>")
        sys.exit(1)

    network_dir = argv[1]
    assert os.path.isdir(network_dir)

    for d in os.listdir(network_dir):
        dom_dir = os.path.join(network_dir, d)
        assert os.path.isdir(dom_dir)
        for c in os.listdir(dom_dir):
            conf_dir = os.path.join(dom_dir, c)
            assert os.path.isdir(conf_dir)
            for f in os.listdir(conf_dir):
                if not f.endswith('.log'):
                    continue
                prob_name = f[:-4]
                log_path = os.path.join(conf_dir, f)
                assert os.path.isfile(log_path)
                
                pddl_domain_path = os.path.join(asnetsfastdownward_dir, 'benchmarks/evaluation_domains/' + d + '/' + 'domain.pddl')
                pddl_prob_path = os.path.join(asnetsfastdownward_dir, 'benchmarks/evaluation_domains/' + d + '/' + prob_name + '.pddl')

                log_file = open(log_path, 'r')
                log_lines = log_file.readlines()
                if len(log_lines) == 1:
                    # only time given yet
                    log_lines.append(compute_number_of_groundings(pddl_domain_path, pddl_prob_path))
                    log_lines.append(compute_pb_network_size(log_path))
                elif len(log_lines) == 2:
                    # time and groundings given
                    log_lines.append(compute_pb_network_size(log_path))
                log_file.close()

                f = open(log_path, 'w')
                for line in log_lines:
                    f.write(line)
                f.close()


if __name__ == "__main__":
    main(sys.argv)
