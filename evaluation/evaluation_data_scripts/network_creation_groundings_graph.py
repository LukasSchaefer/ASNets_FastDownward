#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

def extract_points(network_dir):
    # store coordinates for points of graph representing #groundings -> network creation time/ size
    network_time_regex = r'Model building & saving time: ([0-9]+\.[0-9]+)'
    groundings_regex = r'Number of groundings: (\d+)'
    network_size_regex = r'Protobuf network size: (\d+)MB'
    network_size_float_regex = r'Protobuf network size: (\d+\.\d+)MB'
    network_creation_time_points = []
    network_size_points = []
    for d in os.listdir(network_dir):
        dom_dir = os.path.join(network_dir, d)
        assert os.path.isdir(dom_dir)
        c = 'conf2'
        # only do it for one conf (information is the same just different weights)
        conf_dir = os.path.join(dom_dir, c)
        assert os.path.isdir(conf_dir)
        for f in os.listdir(conf_dir):
            if not f.endswith('.log'):
                continue
            log_path = os.path.join(conf_dir, f)
            assert os.path.isfile(log_path)

            log_file = open(log_path, 'r')
            log_lines = [l.strip() for l in log_file.readlines()]
            if not len(log_lines) == 3:
                print("Log file %s is skipped because not all information is included" % log_path)
                continue

            time_match = re.match(network_time_regex, log_lines[0])
            assert time_match
            time = float(time_match.group(1))
            groundings_match = re.match(groundings_regex, log_lines[1])
            assert groundings_match
            groundings = int(groundings_match.group(1))
            size_match = re.match(network_size_regex, log_lines[2])
            if not size_match:
                size_match = re.match(network_size_float_regex, log_lines[2])
                assert size_match
                size = int(float(size_match.group(1)))
            else:
                size = int(size_match.group(1))

            network_creation_time_points.append((groundings, time, d))
            network_size_points.append((groundings, size, d))

    return network_creation_time_points, network_size_points


def write_groundings_graph(network_creation_time_points, network_size_points, save_dir):
    with open(os.path.join(save_dir, 'groundings_time_graph.tex'), 'w') as f:
        f.write('\\begin{tikzpicture}\n')
        f.write('\t\\begin{axis}[\n')
        f.write('\t\tscale only axis,\n')
        f.write('\t\tscaled ticks=false,\n')
        f.write('\t\theight=6cm,\n')
        f.write('\t\twidth=\\textwidth,\n')
        f.write('\t\tmax space between ticks=50,\n')
        f.write('\t\tminor x tick num=4,\n')
        f.write('\t\tminor y tick num=6,\n')
        f.write('\t\txtick={0, 2000, 4000, 6000, 8000, 10000, 12000, 14000},\n')
        f.write('\t\tytick={0, 1000, 2000, 3000, 4000},\n')
        f.write('\t\ttick style={semithick,color=black},\n')
        f.write('\t\txlabel=number of groundings,\n')
        f.write('\t\tylabel=network creation time (in seconds),\n')
        f.write('\t\ty label style={at={(axis description cs:-0.03,.5)},anchor=south},\n')
        f.write('\t\tlegend style={\n')
        f.write('\t\t\tat={(0.5,1.2)},\n')
        f.write('\t\t\tanchor=north,\n')
        f.write('\t\t\tnodes={anchor=mid},\n')
        f.write('\t\t\tlegend columns=-1,\n')
        f.write('\t\t\t/tikz/every even column/.append style={column sep=4pt}\n')
        f.write('\t\t},\n')
        f.write('\t\tscatter/classes={\n')
        f.write('\t\t\tblocksworld={mark=square*,blue},\n')
        f.write('\t\t\televator={mark=square*,red},\n')
        f.write('\t\t\tfloortile={mark=square*,green},\n')
        f.write('\t\t\thanoi={mark=triangle*,red},\n')
        f.write('\t\t\tparcprinter={mark=triangle*,blue},\n')
        f.write('\t\t\tsokoban={mark=triangle*,green},\n')
        f.write('\t\t\tturnandopen={mark=o,blue},\n')
        f.write('\t\t\ttyreworld={mark=o,red}}]\n')
        f.write('\t\\addplot[scatter,only marks, scatter src=explicit symbolic]\n')
        f.write('\ttable[meta=label] {\n')
        f.write('\t\tx\ty\tlabel\n')
        for time_triple in network_creation_time_points:
            f.write('\t\t%d\t%d\t%s\n' % time_triple)
        f.write('\t};\n')
        f.write('\t\\legend{Blocksworld, Elevator, Floortile, Hanoi, ParcPrinter, Sokoban, Turnandopen, Tyreworld}\n')
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}')

    with open(os.path.join(save_dir, 'groundings_size_graph.tex'), 'w') as f:
        f.write('\\begin{tikzpicture}[trim axis left]\n')
        f.write('\t\\begin{axis}[\n')
        f.write('\t\tscale only axis,\n')
        f.write('\t\tscaled ticks=false,\n')
        f.write('\t\theight=6cm,\n')
        f.write('\t\twidth=\\textwidth,\n')
        f.write('\t\tmax space between ticks=50,\n')
        f.write('\t\tminor x tick num=4,\n')
        f.write('\t\tminor y tick num=6,\n')
        f.write('\t\txtick={0, 2000, 4000, 6000, 8000, 10000, 12000, 14000},\n')
        f.write('\t\tytick={0, 100, 200, 300, 400, 500},\n')
        f.write('\t\ttick style={semithick,color=black},\n')
        f.write('\t\txlabel=number of groundings,\n')
        f.write('\t\tylabel=protobuf network size (in MB),\n')
        f.write('\t\ty label style={at={(axis description cs:-0.03,.5)},anchor=south},\n')
        f.write('\t\tlegend style={\n')
        f.write('\t\t\tat={(0.5,1.2)},\n')
        f.write('\t\t\tanchor=north,\n')
        f.write('\t\t\tnodes={anchor=mid},\n')
        f.write('\t\t\tlegend columns=-1,\n')
        f.write('\t\t\t/tikz/every even column/.append style={column sep=4pt}\n')
        f.write('\t\t},\n')
        f.write('\t\tscatter/classes={\n')
        f.write('\t\t\tblocksworld={mark=square*,blue},\n')
        f.write('\t\t\televator={mark=square*,red},\n')
        f.write('\t\t\tfloortile={mark=square*,green},\n')
        f.write('\t\t\thanoi={mark=triangle*,red},\n')
        f.write('\t\t\tparcprinter={mark=triangle*,blue},\n')
        f.write('\t\t\tsokoban={mark=triangle*,green},\n')
        f.write('\t\t\tturnandopen={mark=o,blue},\n')
        f.write('\t\t\ttyreworld={mark=o,red}}]\n')
        f.write('\t\\addplot[scatter,only marks, scatter src=explicit symbolic]\n')
        f.write('\ttable[meta=label] {\n')
        f.write('\t\tx\ty\tlabel\n')
        for size_triple in network_size_points:
            f.write('\t\t%d\t%d\t%s\n' % size_triple)
        f.write('\t};\n')
        f.write('\t\\legend{Blocksworld, Elevator, Floortile, Hanoi, ParcPrinter, Sokoban, Turnandopen, Tyreworld}\n')
        f.write('\t\\end{axis}\n')
        f.write('\\end{tikzpicture}')


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python3 network_creation_groundings_graph.py <path/to/protobuf/networks_dir> (<path/to/save_dir>)")
        sys.exit(1)
    network_dir = argv[1]
    if len(argv) == 3:
        save_dir = argv[2]
    else:
        save_dir = './'

    network_creation_time_points, network_size_points = extract_points(network_dir)
    write_groundings_graph(network_creation_time_points, network_size_points, save_dir)


if __name__ == "__main__":
    main(sys.argv)
