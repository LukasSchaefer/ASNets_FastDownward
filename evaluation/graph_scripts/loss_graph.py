#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

def extract_loss_values(log_lines, skip_x_losses):
    problem_losses = {}
    loss_index = 0
    current_problem = ""
    problem_name_regex = r'Training data for problem ([a-zA-Z0-9-_]*\.pddl) in epoch \d+:'
    loss_regex = r'loss: ([0-9]+\.[0-9]*)'
    loss_counter = -1
    for line in log_lines:
        match = re.match(problem_name_regex, line)
        if match:
            current_problem = match.group(1)
            loss_counter = -1
            if current_problem not in problem_losses.keys():
                problem_losses[current_problem] = [[]]
            else:
                problem_losses[current_problem].append([])
        
        match_loss = re.match(loss_regex, line)
        if match_loss:
            loss_counter += 1
            if loss_counter % skip_x_losses != 0:
                continue
            loss = float(match_loss.group(1))
            problem_losses[current_problem][-1].append((loss_index, loss))
            loss_index += 1
    return problem_losses


def write_loss_graph(problem_losses, tex_path):
    colors = ['blue', 'green', 'magenta', 'olive', 'orange', 'red', 'yellow']
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
        f.write('\t\tlegend style={\n')
        f.write('\t\t\tat={(0.5,1.2)},\n')
        f.write('\t\t\tanchor=north,\n')
        f.write('\t\t\tnodes={anchor=mid},\n')
        f.write('\t\t\tlegend columns=-1,\n')
        f.write('\t\t\t/tikz/every even column/.append style={column sep=4pt}\n')
        f.write('\t\t},\n')
        f.write('\t\tother/.style={\n')
        f.write('\t\t\tforget plot\n')
        f.write('\t\t},\n')
        f.write('\t\txlabel=training epoch,\n')
        f.write('\t\tylabel=loss value]\n')
        for problem_index, problem_file in enumerate(problem_losses.keys()):
            for epoch_index, epoch_losses in enumerate(problem_losses[problem_file]):
                f.write('\n')
                if epoch_index == 0:
                    f.write('\t\\addplot[smooth,mark=*,color=%s] plot coordinates {\n' % colors[problem_index])
                else:
                    f.write('\t\\addplot[other,smooth,mark=*,color=%s] plot coordinates {\n' % colors[problem_index])
                for index, loss in epoch_losses:
                    f.write('\t\t(%d,%f)\n' % (index, loss))
                f.write('\t};\n')
                if epoch_index == 0:
                    f.write('\t\\addlegendentry{%s}\n' % problem_file)
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}')


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python3 loss_graph.py <path/to/training_sum.log> (<skip_every_x_losses>)")
        sys.exit(1)
    training_log_path = argv[1]
    if len(argv) == 4:
        skip_x_losses = int(argv[3])
    else:
        # default skip 30 losses before taking one
        skip_x_losses = 30

    log_file = open(training_log_path, 'r')
    log_lines = [l.strip() for l in log_file.readlines()]

    # first line ends with <domain>: -> split to get "<domain:" and drop last character
    domain_name = log_lines[0].split()[-1][:-1]
    tex_path = 'loss_graph_' + domain_name + '.tex'
    problem_losses = extract_loss_values(log_lines, skip_x_losses)
    write_loss_graph(problem_losses, tex_path)


if __name__ == "__main__":
    main(sys.argv)
