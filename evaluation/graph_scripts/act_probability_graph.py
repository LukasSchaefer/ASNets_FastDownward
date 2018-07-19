#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

def extract_probabilities(log_lines, skip_x_probs):
    probabilities = []
    current_problem = ""
    prob_regex = r'was chosen with probability ([0-9]+\.[0-9]*)'
    prob_counter = -1
    for line in log_lines:
        match = re.search(prob_regex, line)
        if match:
            prob_counter += 1
            if prob_counter % skip_x_probs != 0:
                continue
            prob = float(match.group(1))
            probabilities.append(prob)
    return probabilities


def write_probability_graph(probabilities, tex_path):
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
        f.write('\t\txlabel=action $a$,\n')
        f.write('\t\txticklabels={,,},\n')
        f.write('\t\tylabel=$\pi^\\theta(a \\mid s)$]\n')
        f.write('\t\\addplot[smooth,mark=*] plot coordinates {\n')
        for prob_index, prob in enumerate(probabilities):
            f.write('\t\t(%d,%f)\n' % (prob_index, prob))
        f.write('\t};\n')
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}')


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python3 loss_graph.py <path/to/training_sum.log> (<skip_every_x_probs>)")
        sys.exit(1)
    training_log_path = argv[1]
    if len(argv) == 4:
        skip_x_losses = int(argv[3])
    else:
        # default skip 10 probabilities before taking one
        skip_x_probs = 10

    log_file = open(training_log_path, 'r')
    log_lines = [l.strip() for l in log_file.readlines()]

    # first line ends with <domain>: -> split to get "<domain:" and drop last character
    domain_name = log_lines[0].split()[-1][:-1]
    tex_path = 'action_probability_graph_' + domain_name + '.tex'
    probabilities = extract_probabilities(log_lines, skip_x_probs)
    write_probability_graph(probabilities, tex_path)


if __name__ == "__main__":
    main(sys.argv)
