#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys


def extract_time_distribution(log_lines):
    problem_times = {}
    current_problem = ""
    problem_name_regex = r'Training data for problem ([a-zA-Z0-9-_]*\.pddl) in epoch \d+:'
    model_build_time_regex = r'model creation time: ([0-9]+\.[0-9]*)s'
    sampling_search_time_regex = r'sampling search time: ([0-9]+\.[0-9]*)s'
    training_time_regex = r'training time: ([0-9]+\.[0-9]*)s'
    model_build_time = 0.0
    sampling_search_time = 0.0
    training_time = 0.0
    for line in log_lines:
        match = re.match(problem_name_regex, line)
        if match:
            if current_problem != "":
                if current_problem in problem_times.keys():
                    model_time, sampling_time, train_time = problem_times[current_problem]
                    model_build_time += model_time
                    sampling_search_time += sampling_time
                    training_time += train_time
                problem_times[current_problem] = (model_build_time, sampling_search_time, training_time)
                model_build_time = 0.0
                sampling_search_time = 0.0
                training_time = 0.0
            current_problem = match.group(1)
        
        model_build_match = re.match(model_build_time_regex, line)
        if model_build_match:
            model_build_time = float(model_build_match.group(1))

        sampling_search_match = re.match(sampling_search_time_regex, line)
        if sampling_search_match:
            sampling_search_time += float(sampling_search_match.group(1))

        training_match = re.match(training_time_regex, line)
        if training_match:
            training_time += float(training_match.group(1))
    return problem_times


def write_time_distribution_bars(time_distribution, tex_path):
    network_build_time = 0.0
    sampling_time = 0.0
    training_time = 0.0
    for problem_file in time_distribution.keys():
        b_t, s_t, t_t = time_distribution[problem_file]
        network_build_time += b_t
        sampling_time += s_t
        training_time += t_t

    with open(tex_path, 'w') as f:
        f.write('\\begin{tikzpicture}\n')
        f.write('\t\\begin{axis}[\n')
        f.write('\t\tscale only axis,\n')
        f.write('\t\tybar,\n')
        f.write('\t\tenlargelimits=0.15,\n')
        f.write('\t\tbar width=1.0cm,\n')
        f.write('\t\twidth=\\textwidth,\n')
        f.write('\t\theight=6cm,\n')
        f.write('\t\tylabel={time in seconds},\n')
        f.write('\t\ty label style={at={(axis description cs:-0.0,.5)},anchor=south},\n')
        f.write('\t\tsymbolic x coords={network building, sampling search, training},\n')
        f.write('\t\txtick=data,\n')
        f.write('\t\tnodes near coords,\n')
        f.write('\t\tnodes near coords align={vertical}\n')
        f.write('\t]\n')
        f.write('\t\\addplot coordinates {(network building, %f) (sampling search, %f) (training, %f)};\n' % (network_build_time, sampling_time, training_time))
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}\n')


def write_problem_time_distribution_bars(time_distribution, tex_path):
    with open(tex_path, 'w') as f:
        f.write('\\begin{tikzpicture}\n')
        f.write('\t\\begin{axis}[\n')
        f.write('\t\tscale only axis,\n')
        f.write('\t\tybar,\n')
        f.write('\t\tenlargelimits=0.15,\n')
        f.write('\t\tx=6cm,\n')
        f.write('\t\tylabel={time in seconds},\n')
        f.write('\t\tsymbolic x coords={network building, sampling search, training},\n')
        f.write('\t\txtick=data,\n')
        f.write('\t\tbar width=0.8cm,\n')
        f.write('\t\tlegend style={\n')
        f.write('\t\t\tat={(0.5,-0.15)},\n')
        f.write('\t\t\tanchor=north,\n')
        f.write('\t\t\tlegend columns=-1,\n')
        f.write('\t\t\t/tikz/every even column/.append style={column sep=4pt}\n')
        f.write('\t\t}\n')
        f.write('\t]\n')
        for problem_file in time_distribution.keys():
            network_build_time, sampling_time, training_time = time_distribution[problem_file]
            f.write('\t\\addplot coordinates {(network building, %f) (sampling search, %f) (training, %f)};\n' % (network_build_time, sampling_time, training_time))
            f.write('\t\\addlegendentry{%s}\n' % problem_file)
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}\n')


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python3 time_distribution_bars.py <path/to/training_sum.log> (<path/to/save_dir>)")
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
    tex_path = os.path.join(save_dir, 'time_distribution_' + domain_name + '.tex')
    prob_tex_path = os.path.join(save_dir, 'prob_time_distribution_' + domain_name + '.tex')
    time_distribution = extract_time_distribution(log_lines)
    write_time_distribution_bars(time_distribution, tex_path)
    write_problem_time_distribution_bars(time_distribution, prob_tex_path)


if __name__ == "__main__":
    main(sys.argv)
