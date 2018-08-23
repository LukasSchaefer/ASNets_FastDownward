#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

from evaluation_data_scripts.act_probability_graph import main as act_prob_graph
from evaluation_data_scripts.loss_graph import main as loss_graph
from evaluation_data_scripts.time_distribution_bars import main as time_distr_graphs
from evaluation_data_scripts.success_rate_graph import main as success_rate_graph

def write_dom_training_sum(domain_name, domain_dir, included_confs):
    included_confs.sort()
    training_dom_path = os.path.join(domain_dir, 'summary_section.tex')
    if domain_name == 'parcprinter':
        dom_name = 'ParcPrinter'
    elif domain_name == 'turnandopen':
        dom_name = 'TurnAndOpen'
    else:
        dom_name = domain_name[0].capitalize() + domain_name[1:]
    with open(training_dom_path, 'w') as f:
        f.write('\\section{%s}\n\n' % dom_name)
        for conf in included_confs:
            if conf == 'conf1':
                conf_name = '1st configuration: A$^*$ $h^{LM-cut}$ teacher'
            elif conf == 'conf2':
                conf_name = '2nd configuration: A$^*$ $h^{add}$ teacher'
            elif conf == 'conf3':
                conf_name = '3rd configuration: GBFS $h^{FF}$ teacher'
            f.write('\\FloatBarrier\n')
            f.write('\\subsection*{%s}\n' % conf_name)
            f.write('\\begin{figure}[h]\n')
            f.write('\t\\centering\n')
            rel_path = os.path.join(domain_dir, conf + '/time_distribution_' + domain_name + '.tex')
            f.write('\t\\resizebox{0.88\\linewidth}{!}{\n')
            f.write('\t\t\\input{%s}\n\n' % os.path.abspath(rel_path))
            f.write('\t}\n')
            f.write('\t\\mbox{} \\vspace{1cm}\n\n')

            rel_path = os.path.join(domain_dir, conf + '/success_rate_graph_' + domain_name + '.tex')
            f.write('\t\\resizebox{0.88\\linewidth}{!}{\n')
            f.write('\t\t\\input{%s}\n' % os.path.abspath(rel_path))
            f.write('\t}\n')
            f.write('\t\\mbox{} \\vspace{0.2cm}\n\n')

            rel_path = os.path.join(domain_dir, conf + '/loss_graph_' + domain_name + '.tex')
            f.write('\t\\resizebox{0.88\\linewidth}{!}{\n')
            f.write('\t\t\\input{%s}\n' % os.path.abspath(rel_path))
            f.write('\t}\n')
            f.write('\t\\caption{Time distribution, success rate and loss development during training}\n')
            f.write('\\end{figure}\n\n')

            if conf != included_confs[-1]:
                # no newpage for last configuration
                f.write('\\newpage\n\n')

            # f.write('\t\\paragraph{Action probabilities development}\n')
            # f.write('\t\\ \\\\ \n')
            # rel_path = os.path.join(domain_dir, conf + '/action_probability_graph_' + domain_name + '.tex')
            # f.write('\t\\input{%s}\n\n' % os.path.abspath(rel_path))


def main(argv):
    if len(argv) != 3:
        print("Usage: python3 training_graphs_generator.py <path/to/training_log_dir> <path/to/save_dir>")
        print("Note that the benchmark/confx folder hierarchy from the training log dir must already " +
              "exist in the save dir")
        sys.exit(1)
    training_dir = argv[1]
    save_dir = argv[2]

    for domain_name in os.listdir(training_dir):
        domain_path = os.path.join(training_dir, domain_name)
        included_confs = []
        assert os.path.isdir(domain_path)
        for conf_name in os.listdir(domain_path):
            conf_path = os.path.join(domain_path, conf_name)
            assert os.path.isdir(conf_path)
            for f in os.listdir(conf_path):
                if f != 'training_sum.log':
                    continue
                train_log_sum_path = os.path.join(conf_path, f)
                assert os.path.isfile(train_log_sum_path)

                included_confs.append(conf_name)

                save_dir_current = os.path.join(save_dir, domain_name + '/' + conf_name)
                time_distr_graphs(['blub', train_log_sum_path, save_dir_current])
                success_rate_graph(['blub', train_log_sum_path, save_dir_current])
                loss_graph(['blub', train_log_sum_path, save_dir_current])
                act_prob_graph(['blub', train_log_sum_path, save_dir_current])

        if len(included_confs) > 0:
            print('Creating sum for %s' % domain_name)
            write_dom_training_sum(domain_name, os.path.join(save_dir, domain_name), included_confs)


if __name__ == "__main__":
    main(sys.argv)
