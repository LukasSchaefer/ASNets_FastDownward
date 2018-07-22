#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

def extract_success_rates(log_lines):
    success_rates = []
    problem_name_regex = r'Success rate: (\d+)'
    for line in log_lines:
        match = re.match(problem_name_regex, line)
        if match:
            success_rate = match.group(1)
            success_rates.append(success_rate)
    return success_rates


def write_success_rate_graph(success_rates, tex_path):
    with open(tex_path, 'w') as f:
        f.write('\\begin{tikzpicture}[trim axis left]\n')
        f.write('\t\\begin{axis}[\n')
        f.write('\t\tscale only axis,\n')
        f.write('\t\theight=5cm,\n')
        f.write('\t\twidth=\\textwidth,\n')
        f.write('\t\tmax space between ticks=50,\n')
        f.write('\t\tminor x tick num=4,\n')
        f.write('\t\tminor y tick num=4,\n')
        f.write('\t\ttick style={semithick,color=black},\n')
        f.write('\t\txlabel=epoch,\n')
        f.write('\t\tylabel=success rate (in percent)]\n')
        f.write('\t\\addplot[smooth,mark=*] plot coordinates {\n')
        for epoch, success_rate in enumerate(success_rates):
            f.write('\t\t(%d,%d)\n' % (epoch + 1, int(success_rate)))
        f.write('\t};\n')
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}')


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python3 loss_graph.py <path/to/training_sum.log> (<path/to/save_dir>)")
        sys.exit(1)
    training_log_path = argv[1]
    if len(argv) == 3:
        save_dir = argv[2]
    else:
        save_dir = './'

    log_file = open(training_log_path, 'r')
    log_lines = [l.strip() for l in log_file.readlines()]

    # first line ends with <domain>: -> split to get "<domain:" and drop last character
    domain_name = log_lines[0].split()[-1][:-1]
    tex_path = os.path.join(save_dir, 'success_rate_graph_' + domain_name + '.tex')
    success_rates = extract_success_rates(log_lines)
    write_success_rate_graph(success_rates, tex_path)


if __name__ == "__main__":
    main(sys.argv)
