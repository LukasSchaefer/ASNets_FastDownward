#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

def save_costs_for_probs(report_lines, line_index, dom_dict, conf):
    prob_name_regex = r'<td>([a-zA-Z0-9-_]+)\.pddl<\/td>'
    cost_regex = r'<td align="right"><span style="color:rgb\(\d+,\d+,\d+\)">(\d+\.\d+)<\/span><\/td>'

    line_index += 11
    line = report_lines[line_index].strip()
    # now on first prob table
    prob_name_match = re.match(prob_name_regex, line)
    while prob_name_match:
        prob_dict = {}
        prob_name = prob_name_match.group(1)

        line_index += 1
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)
        if not cost_match:
            cost = '/'
        else:
            # while this is really ugly I want to get rid of the (seemingly) unnecessary float .00 for the
            # costs which were all natural numbers (this works also if there are truly real cost values)
            cost = str(int(float(cost_match.group(1))))

        prob_dict[conf] = cost
        dom_dict[prob_name] = prob_dict

        line_index += 3
        line = report_lines[line_index].strip()
        prob_name_match = re.match(prob_name_regex, line)

    assert line == ''
    return line_index + 1


def save_coverages_for_dom(report_lines, line_index, dom_dict, dom_size, conf):
    line_index += 1
    solved_probs_regex = r'<td align="right"><span style="color:rgb\(\d+,\d+,\d+\)">(\d+)<\/span><\/td>'
    line = report_lines[line_index].strip()

    solved_probs_match = re.match(solved_probs_regex, line)
    cov = solved_probs_match.group(1) + "/" + dom_size
    dom_dict[conf] = cov

    return line_index + 3


def save_times_for_probs(report_lines, line_index, dom_dict, conf):
    prob_name_regex = r'<td>([a-zA-Z0-9-_]+)\.pddl<\/td>'
    time_regex = r'<td align="right"><span style="color:rgb\(\d+,\d+,\d+\)">(\d+\.\d+)<\/span><\/td>'

    line_index += 11
    line = report_lines[line_index].strip()
    # now on first prob table
    prob_name_match = re.match(prob_name_regex, line)
    while prob_name_match:
        prob_name = prob_name_match.group(1)
        prob_dict = dom_dict[prob_name]

        line_index += 1
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)
        if time_match:
            time = time_match.group(1) + 's'
        else:
            time = '/'

        cost = prob_dict[conf]
        prob_dict[conf] = (cost, time)

        line_index += 3
        line = report_lines[line_index].strip()
        prob_name_match = re.match(prob_name_regex, line)

    assert line == ''
    return line_index + 1



def extract_report_data(report_lines, conf):
    # dict containing one dict for each domain (dom-name as key)
    # each domain dict contains 
    # - conf: coverage
    # - dict for each problem instance (by name)
    # problem dicts contains triple with conf: (plan_cost, search_time, build_time)
    report_dict = {}

    domains = ['blocksworld', 'elevator', 'floortile', 'hanoi', 'parcprinter', 'sokoban', 'turnandopen', 'tyreworld']
    for dom in domains:
        report_dict[dom] = {}

    line_index = 0

    # find costs
    line = report_lines[line_index].strip()
    cost_regex = r'<a id="cost-([a-z]+)" name="cost-[a-z]+"><\/a>'
    cost_match = re.match(cost_regex, line)
    while not cost_match:
        line_index += 1
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)

    cost_match = re.match(cost_regex, line)
    while cost_match:
        dom_name = cost_match.group(1)
        assert dom_name in domains

        line_index = save_costs_for_probs(report_lines, line_index, report_dict[dom_name], conf)
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)

    # now on coverage start
    line_index += 11
    line = report_lines[line_index].strip()
    coverage_regex = r'<td><a href="#coverage-([a-z]+)" onclick="show_table\(document.getElementById\(\'[\w-]+\'\)\);">[a-z]+<\/a> \((\d+)\)<\/td>'
    cov_match = re.match(coverage_regex, line)
    while cov_match:
        dom_name = cov_match.group(1)
        dom_size = cov_match.group(2)

        line_index = save_coverages_for_dom(report_lines, line_index, report_dict[dom_name], dom_size, conf)
        line = report_lines[line_index].strip()
        cov_match = re.match(coverage_regex, line)

    # on Sum coverage table line
    time_regex = r'<a id="total_time-([a-z]+)" name="total_time-[a-z]+"><\/a>'
    line = report_lines[line_index].strip()
    time_match = re.match(time_regex, line)
    while not time_match:
        line_index += 1
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)

    # found first domain time table
    while time_match:
        dom_name = time_match.group(1)
        assert dom_name in domains

        line_index = save_times_for_probs(report_lines, line_index, report_dict[dom_name], conf)
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)

    return report_dict


def merge_report_dicts(report_dicts):
    domains = ['blocksworld', 'elevator', 'floortile', 'hanoi', 'parcprinter', 'sokoban', 'turnandopen', 'tyreworld']
    report_dict = {}

    for conf, conf_report_dict in report_dicts:
        for dom in conf_report_dict.keys():
            dom_dict = conf_report_dict[dom]
            if not dom in report_dict.keys():
                # dom-dict can be transferred to report_dict
                report_dict[dom] = dom_dict
            else:
                # dom-dict already exists in report_dict -> needs to be extended
                merged_dom_dict = report_dict[dom]
                for prob_name, prob_dict in dom_dict.items():
                    if prob_name == conf:
                        # coverage entry -> add it
                        merged_dom_dict[conf] = prob_dict
                    else:
                        # usual problem dict -> will already be included
                        assert prob_name in merged_dom_dict.keys()
                        merged_prob_dict = merged_dom_dict[prob_name]
                        # add values for current conf
                        merged_prob_dict[conf] = prob_dict[conf]
    return report_dict


def extend_network_creation_times(report_dict, network_dir, conf):
    domains = ['blocksworld', 'elevator', 'floortile', 'hanoi', 'parcprinter', 'sokoban', 'turnandopen', 'tyreworld']
    time_regex = r'Model building & saving time: ([0-9]+\.[0-9][0-9])'

    for dom in domains:
        dom_dict = report_dict[dom]
        
        for prob_name in dom_dict.keys():
            if prob_name.startswith('conf') or prob_name.startswith('exp'):
                continue
            prob_dict = dom_dict[prob_name]

            if conf.startswith('exp'):
                conf_string = conf[4:]
            else:
                conf_string = conf
            prob_log_path = os.path.join(network_dir, dom + '/' + conf_string + '/' + prob_name + '.log')
            if not os.path.isfile(prob_log_path):
                build_time = '/'
            else:
                prob_log_file = open(prob_log_path, 'r')
                log_lines = [l.strip() for l in prob_log_file.readlines()]
                time_match = re.match(time_regex, log_lines[0])
                assert time_match
                build_time = time_match.group(1) + 's'

            cost, time = prob_dict[conf]
            prob_dict[conf] = (cost, time, build_time)


def add_network_data_to_tables(report_dict, tables_dir):
    # updating coverage table
    coverage_table_file = open(os.path.join(tables_dir, 'eval_coverage_table.tex'), 'r')
    coverage_table_lines = coverage_table_file.readlines()

    line_index = 0
    # cov_regex = r'[\w]+ & \d+\/\d+ & \d+\/\d+ & \d+\/\d+ & \d+\/\d+ & - & - & - & - & - & -'
    cov_regex = r'[\w]+ & \d+\/\d+ & \d+\/\d+ & \d+\/\d+ & \d+\/\d+ & - & - & -'
    while line_index < len(coverage_table_lines):
        line = coverage_table_lines[line_index]
        match = re.search(cov_regex, line)
        if not match:
            line_index += 1
            continue

        elements = [s.strip() for s in line.split('&')]
        dom_name = elements[0].lower()
        dom_dict = report_dict[dom_name]
        if not dom_dict.keys():
            # domain was not incldued in reports of ASNets
            line_index += 1
            print('Domain %s did not occur in the ASNet reports' % dom_name)
            continue

        elements[-3] = dom_dict['conf1']
        # elements[-5] = dom_dict['exp_conf1']
        elements[-2] = dom_dict['conf2']
        # elements[-3] = dom_dict['exp_conf2']
        elements[-1] = dom_dict['conf3']
        # elements[-1] = dom_dict['exp_conf3']

        # new_line = '\t%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(elements)
        new_line = '\t%s & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(elements)
        coverage_table_lines[line_index] = new_line
        print('Domain %s was updated' % dom_name)
        line_index += 1

    with open(os.path.join(tables_dir, 'eval_coverage_table.tex'), 'w') as f:
        for line in coverage_table_lines:
            f.write(line)


    prob_file_regex = r'\\textbf{([\w\-\_\d]+)}'
    for dom in report_dict.keys():
        dom_dict = report_dict[dom]
        if not dom_dict.keys():
            continue
        eval_table_file = open(os.path.join(tables_dir, dom + '_evaluation_tables.tex'), 'r')
        eval_table_lines = eval_table_file.readlines()
        line_index = 0

        current_problem = ''
        while line_index < len(eval_table_lines):
            line = eval_table_lines[line_index].lstrip()

            prob_file_match = re.search(prob_file_regex, line)
            if prob_file_match:
                current_problem = prob_file_match.group(1)
            elif line.startswith('Plan cost'):
                assert current_problem != ''
                elements = [s.strip() for s in line.split('&')]
                prob_dict = dom_dict[current_problem]
                elements[-3] = prob_dict['conf1'][0]
                # elements[-5] = prob_dict['exp_conf1'][0]
                elements[-2] = prob_dict['conf2'][0]
                # elements[-3] = prob_dict['exp_conf2'][0]
                elements[-1] = prob_dict['conf3'][0]
                # elements[-1] = prob_dict['exp_conf3'][0]
                new_line = '\t%s & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(elements)
                eval_table_lines[line_index] = new_line
            elif line.startswith('Search time'):
                assert current_problem != ''
                elements = [s.strip() for s in line.split('&')]
                prob_dict = dom_dict[current_problem]
                elements[-3] = prob_dict['conf1'][1]
                # elements[-5] = prob_dict['exp_conf1'][1]
                elements[-2] = prob_dict['conf2'][1]
                # elements[-3] = prob_dict['exp_conf2'][1]
                elements[-1] = prob_dict['conf3'][1]
                # elements[-1] = prob_dict['exp_conf3'][1]
                new_line = '\t%s & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(elements)
                eval_table_lines[line_index] = new_line
            elif line.startswith('Model creation time'):
                assert current_problem != ''
                elements = [s.strip() for s in line.split('&')]
                prob_dict = dom_dict[current_problem]
                elements[-3] = prob_dict['conf1'][2]
                # elements[-5] = prob_dict['exp_conf1'][2]
                elements[-2] = prob_dict['conf2'][2]
                # elements[-3] = prob_dict['exp_conf2'][2]
                elements[-1] = prob_dict['conf3'][2]
                # elements[-1] = prob_dict['exp_conf3'][2]
                new_line = '\t%s & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(elements)
                eval_table_lines[line_index] = new_line

            line_index += 1

        with open(os.path.join(tables_dir, dom + '_evaluation_tables.tex'), 'w') as f:
            for line in eval_table_lines:
                f.write(line)


def main(argv):
    if len(argv) != 4:
        print("Usage: python3 network_evaluation_report_data.py <path/to/reports/directory> <path/to/pb_networks/directory> <path/to/tables/directory>")
        print("reports directory needs to contain asnet_eval_report_conf?.html files for valid configuration numbers")
        sys.exit(1)
    report_dir = argv[1]
    assert os.path.isdir(report_dir)
    network_dir = argv[2]
    assert os.path.isdir(network_dir)
    tables_dir = argv[3]
    assert os.path.isdir(tables_dir)

    confs = ['conf1', 'conf2', 'conf3']
    exps = ['', 'exp_']
    report_dicts = []

    
    for exp in exps:
        for conf in confs:
            report_path = os.path.join(report_dir, 'asnet_eval_report_' + exp + conf + '.html')
            report_file = open(report_path, 'r')
            report_lines = [l.strip() for l in report_file.readlines()]
            report_dict = extract_report_data(report_lines, exp + conf)
            extend_network_creation_times(report_dict, network_dir, exp + conf)
            report_dicts.append((exp + conf, report_dict))

    report_dict = merge_report_dicts(report_dicts)
    add_network_data_to_tables(report_dict, tables_dir)


if __name__ == "__main__":
    main(sys.argv)
