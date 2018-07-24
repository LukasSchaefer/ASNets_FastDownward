#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

def extract_probabilities(log_lines, skip_x_probs):
    probabilities = []
    epoch_indeces = []
    current_problem = ""
    prob_regex = r'was chosen with probability ([0-9]+\.[0-9]*)'
    epoch_regex = r'Epoch (\d+):'
    prob_counter = -1
    prob_index = 0
    for line in log_lines:
        match = re.search(prob_regex, line)
        if match:
            prob_counter += 1
            if prob_counter % skip_x_probs != 0:
                continue
            prob = float(match.group(1))
            probabilities.append((prob_index * skip_x_probs, prob))
            prob_index += 1
            continue

        match = re.match(epoch_regex, line)
        if match:
            epoch_indeces.append(prob_index)
            prob_index += 1
    return probabilities, epoch_indeces


def write_probability_graph(probabilities, epoch_indeces, tex_path):
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
        for prob_index, prob in probabilities:
            f.write('\t\t(%d,%f)\n' % (prob_index, prob))
        f.write('\t};\n')
        for epoch_index in epoch_indeces:
            f.write('\t\\draw[dashed] ({axis cs:%d,0}|-{rel axis cs:0,1}) -- ({axis cs:%d,0}|-{rel axis cs:0,0});\n' % (epoch_index, epoch_index))
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}')


def main(argv):
    if len(argv) < 2 or len(argv) > 4:
        print("Usage: python3 loss_graph.py <path/to/training_sum.log> (<path/to/save_dir>) (<skip_every_x_probs>)")
        sys.exit(1)
    training_log_path = argv[1]
    if len(argv) == 3:
        save_dir = argv[2]
    else:
        save_dir = './'
    if len(argv) == 4:
        skip_x_probs = int(argv[3])
    else:
        # default skip 10 probabilities before taking one
        skip_x_probs = 10
    if 'elevator' in training_log_path:
        skip_x_probs = 1

    log_file = open(training_log_path, 'r')
    log_lines = [l.strip() for l in log_file.readlines()]

    # first line ends with <domain>: -> split to get "<domain:" and drop last character
    domain_name = log_lines[0].split()[-1][:-1]
    tex_path = os.path.join(save_dir, 'action_probability_graph_' + domain_name + '.tex')
    probabilities, epoch_indeces = extract_probabilities(log_lines, skip_x_probs)
    write_probability_graph(probabilities, epoch_indeces[1:], tex_path)


if __name__ == "__main__":
    main(sys.argv)
