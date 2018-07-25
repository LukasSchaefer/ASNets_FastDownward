#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

def save_costs_for_probs(report_lines, line_index, dom_dict):
    prob_name_regex = r'<td>([a-zA-Z0-9-_]+)\.pddl<\/td>'
    cost_regex = r'<td align="right"><span style="color:rgb\(\d+,\d+,\d+\)">(\d+\.\d+)<\/span><\/td>'

    line_index += 14
    line = report_lines[line_index].strip()
    # now on first prob table
    prob_name_match = re.match(prob_name_regex, line)
    while prob_name_match:
        prob_name = prob_name_match.group(1)
        prob_dict = {}

        line_index += 1
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)
        if not cost_match:
            cost = '/'
        else:
            # while this is really ugly I want to get rid of the (seemingly) unnecessary float .00 for the
            # costs which were all natural numbers (this works also if there are truly real cost values)
            cost = str(int(float(cost_match.group(1))))
        prob_dict['astar_add'] = cost

        line_index += 1
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)
        if not cost_match:
            cost = '/'
        else:
            cost = str(int(float(cost_match.group(1))))
        prob_dict['astar_lmcut'] = cost

        line_index += 1
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)
        if not cost_match:
            cost = '/'
        else:
            cost = str(int(float(cost_match.group(1))))
        prob_dict['gbfs_ff'] = cost

        line_index += 1
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)
        if not cost_match:
            cost = '/'
        else:
            cost = str(int(float(cost_match.group(1))))
        prob_dict['lama'] = cost

        dom_dict[prob_name] = prob_dict

        line_index += 3
        line = report_lines[line_index].strip()
        prob_name_match = re.match(prob_name_regex, line)

    assert line == ''
    return line_index + 1


def save_coverages_for_dom(report_lines, line_index, dom_dict, dom_size):
    line_index += 1
    solved_probs_regex = r'<td align="right"><span style="color:rgb\(\d+,\d+,\d+\)">(\d+)<\/span><\/td>'
    line = report_lines[line_index].strip()

    solved_probs_match = re.match(solved_probs_regex, line)
    cov = solved_probs_match.group(1) + "/" + dom_size
    dom_dict['astar_add'] = cov

    line_index += 1
    line = report_lines[line_index].strip()
    solved_probs_match = re.match(solved_probs_regex, line)
    cov = solved_probs_match.group(1) + "/" + dom_size
    dom_dict['astar_lmcut'] = cov

    line_index += 1
    line = report_lines[line_index].strip()
    solved_probs_match = re.match(solved_probs_regex, line)
    cov = solved_probs_match.group(1) + "/" + dom_size
    dom_dict['gbfs_ff'] = cov

    line_index += 1
    line = report_lines[line_index].strip()
    solved_probs_match = re.match(solved_probs_regex, line)
    cov = solved_probs_match.group(1) + "/" + dom_size
    dom_dict['lama'] = cov

    return line_index + 3


def save_times_for_probs(report_lines, line_index, dom_dict):
    prob_name_regex = r'<td>([a-zA-Z0-9-_]+)\.pddl<\/td>'
    time_regex = r'<td align="right"><span style="color:rgb\(\d+,\d+,\d+\)">(\d+\.\d+)<\/span><\/td>'

    line_index += 14
    line = report_lines[line_index].strip()
    # now on first prob table
    prob_name_match = re.match(prob_name_regex, line)
    while prob_name_match:
        prob_name = prob_name_match.group(1)
        prob_dict = dom_dict[prob_name]

        line_index += 1
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)
        cost = prob_dict['astar_add']
        if cost == '/':
            prob_dict['astar_add'] = ('/', '/')
        else:
            time = time_match.group(1)
            prob_dict['astar_add'] = (cost, time + 's')

        line_index += 1
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)
        cost = prob_dict['astar_lmcut']
        if cost == '/':
            prob_dict['astar_lmcut'] = ('/', '/')
        else:
            time = time_match.group(1)
            prob_dict['astar_lmcut'] = (cost, time + 's')

        line_index += 1
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)
        cost = prob_dict['gbfs_ff']
        if cost == '/':
            prob_dict['gbfs_ff'] = ('/', '/')
        else:
            time = time_match.group(1)
            prob_dict['gbfs_ff'] = (cost, time + 's')

        line_index += 1
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)
        cost = prob_dict['lama']
        if cost == '/':
            prob_dict['lama'] = ('/', '/')
        else:
            if not time_match:
                time = '/*'
                prob_dict['lama'] = (cost, time)
            else:
                time = time_match.group(1)
                prob_dict['lama'] = (cost, time + 's')

        dom_dict[prob_name] = prob_dict

        line_index += 3
        line = report_lines[line_index].strip()
        prob_name_match = re.match(prob_name_regex, line)

    assert line == ''
    return line_index + 1



def extract_report_data(report_lines):
    # dict containing one dict for each domain (dom-name as key)
    # each domain dict contains 
    # - planner-name: coverage
    # - dict for each problem instance (by name)
    # problem dicts contains values for planners
    # - planner-name: (plan_cost, search_time)
    report_dict ={}
    blocksworld_dict = {}
    report_dict['blocksworld'] = blocksworld_dict
    elevator_dict = {}
    report_dict['elevator'] = elevator_dict
    floortile_dict = {}
    report_dict['floortile'] = floortile_dict
    hanoi_dict = {}
    report_dict['hanoi'] = hanoi_dict
    parcprinter_dict = {}
    report_dict['parcprinter'] = parcprinter_dict
    sokoban_dict = {}
    report_dict['sokoban'] = sokoban_dict
    turnandopen_dict = {}
    report_dict['turnandopen'] = turnandopen_dict
    tyreworld_dict = {}
    report_dict['tyreworld'] = tyreworld_dict

    domains = ['blocksworld', 'elevator', 'floortile', 'hanoi', 'parcprinter', 'sokoban', 'turnandopen', 'tyreworld']
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

        line_index = save_costs_for_probs(report_lines, line_index, report_dict[dom_name])
        line = report_lines[line_index].strip()
        cost_match = re.match(cost_regex, line)

    # now on coverage start
    line_index += 14
    line = report_lines[line_index].strip()
    coverage_regex = r'<td><a href="#coverage-([a-z]+)" onclick="show_table\(document.getElementById\(\'[\w-]+\'\)\);">[a-z]+<\/a> \((\d+)\)<\/td>'
    cov_match = re.match(coverage_regex, line)
    while cov_match:
        dom_name = cov_match.group(1)
        dom_size = cov_match.group(2)

        line_index = save_coverages_for_dom(report_lines, line_index, report_dict[dom_name], dom_size)
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

        line_index = save_times_for_probs(report_lines, line_index, report_dict[dom_name])
        line = report_lines[line_index].strip()
        time_match = re.match(time_regex, line)

    return report_dict


def write_evaluation_tables(report_data, tables_dir):
    planners = ['astar_lmcut', 'astar_add', 'gbfs_ff', 'lama']

    # creating coverage table
    with open(os.path.join(tables_dir, 'eval_coverage_table.tex'), 'w') as f:
        f.write('\\section{Coverage}\n')
        f.write('\\begin{tabular}{l || l | l | l | l | c | c | c}\n')
        f.write('\tDomain & A$^*$ LM & A$^*$ add & GBFS & LAMA & ASNet LM & ASNet add & ASNet FF\\\\ \\hhline{========}\n')
        for dom in report_data.keys():
            dom_dict = report_data[dom]
            dom_name = dom[0].capitalize() + dom[1:]
            cov_list = [dom_name]
            for planner in planners:
                cov = dom_dict[planner]
                cov_list.append(cov)
            # add - for ASNets
            for _ in range(3):
                cov_list.append('-')
            f.write('\t%s & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(cov_list))
        f.write('\\end{tabular}')


    for dom in report_data.keys():
        dom_dict = report_data[dom]
        dom_name = dom[0].capitalize() + dom[1:]
        with open(os.path.join(tables_dir, dom + '_evaluation_tables.tex'), 'w') as f:
            f.write('\\subsection{%s}\n' % dom_name)
            for prob_name in dom_dict.keys():
                f.write('\n')
                if prob_name in planners:
                    continue
                prob_dict = dom_dict[prob_name]
                costs = []
                search_times = []
                for planner in planners:
                    cost, time = prob_dict[planner]
                    costs.append(cost)
                    search_times.append(time)
                # for ASNets
                for _ in range(3):
                    costs.append('-')
                    search_times.append('-')

                f.write('\\vspace{0.5cm}\n')
                f.write('\\noindent\n')
                # f.write('\\resizebox{1.1\linewidth}{!}{\n')
                f.write('\t\\begin{tabular}{l || l l l l | l l l}\n')
                f.write('\t\t\\textbf{%s} & A$^*$ LM & A$^*$ add & GBFS & LAMA & ASNet LM & ASNet add & ASNet FF\\\\ \\hline\n' % prob_name)
                f.write('\t\tPlan cost & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(costs))
                f.write('\t\tSearch time & %s & %s & %s & %s & %s & %s & %s\\\\ \n' % tuple(search_times))
                f.write('\t\tModel creation time & - & - & - & - & - & - & -\\\\ \n')
                f.write('\t\\end{tabular}\n')
                # f.write('}\n')
            

def main(argv):
    if len(argv) != 3:
        print("Usage: python3 evaluation_report_data.py <path/to/report.html> <path/to/tables/directory>")
        print("Should be executed on LAB report")
        sys.exit(1)
    report_path = argv[1]
    tables_dir = argv[2]
    assert os.path.isdir(tables_dir)

    report_file = open(report_path, 'r')
    report_lines = [l.strip() for l in report_file.readlines()]

    report_dict = extract_report_data(report_lines)

    write_evaluation_tables(report_dict, tables_dir)


if __name__ == "__main__":
    main(sys.argv)
