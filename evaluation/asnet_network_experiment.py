#! /usr/bin/env python

import os
from lab.environments import OracleGridEngineEnvironment
from downward.experiment import FastDownwardASNetExperiment
from downward.reports.absolute import AbsoluteReport

REPO = '/mnt/data_server/schaefer/asnetsfastdownward'
BENCHMARK = '/mnt/data_server/schaefer/asnetsfastdownward/benchmarks/evaluation_domains'
BENCHMARK_TRAINING = '/mnt/data_server/schaefer/asnetsfastdownward/benchmarks'
ENV = OracleGridEngineEnvironment(queue='all.q@@fai0x')
REVISION_CACHE = os.path.expanduser('~/lab/revision-cache')

SUITE = ['turnandopen', 'tyreworld', 'sokoban', 'hanoi', 'floortile', 'blocksworld', 'elevator', 'parcprinter']

ATTRIBUTES = ['unsolvable', 'memory', 'total_search_time', 'total_time', 'plan_length', 'cost', 'coverage', 'error']

exp = FastDownwardASNetExperiment(environment=ENV, revision_cache=REVISION_CACHE)
exp.add_suite(BENCHMARK, SUITE)

# losses = ['']#, '--loss_just_opt']
# layers = ['2']#, '4']
# teacher_searches = [('astar_lmcut', '"astar(lmcut(), transform=asnet_sampling_transform())"')]#,
#                     #('astar_add', '"astar(add(), transform=asnet_sampling_transform())"'),
#                     #('ehc_ff', '"ehc(ff(), transform=asnet_sampling_transform())"')]
# 
# for loss in losses:
#     for layer_num in layers:
#         for teacher_search_name, teacher_search in teacher_searches:
#             conf_string = loss + '_' + layer_num + '_layers' + '_' + teacher_search_name
#             # training
#             for domain in exp._suites[BENCHMARK]:
#                 training_dir = BENCHMARK_TRAINING + '/' + domain + '/training'
#                 run = exp.add_run()
#                 run.add_command('asnet_train_' + domain + '_' + conf_string,
#                                 ['python3', os.path.join(REPO, 'fast-asnet.py'), '--build', 'release64dynamic', '-t',
#                                  '-d', training_dir, '--print_all', loss, '-layers', layer_num, '--teacher_search',
#                                  teacher_search])
# 
#             # evaluation
#             for task in exp._get_tasks():
#                 domain_name = os.path.dirname(domain)
#                 domain = task.domain
#                 problem = task.problem
#                 run = exp.add_run()
#                 # add task ressources
#                 run.add_resource('domain', domain, 'domain.pddl', symlink=True)
#                 run.add_resource('problem', problem, 'problem.pddl', symlink=True)
# 
#                 # build evaluation directory with domain and problem
#                 # loss string
#                 if loss == '':
#                     loss_just_opt = False
#                 elif loss == '--loss_just_opt':
#                     loss_just_opt = True
#                 else:
#                     raise ValueError("Invalid loss string")
# 
#                 weights_path = os.path.join(BENCHMARK_TRAINING + '/' + domain_name + '/training',
#                         'asnets_final_weights.h5')
#                 network_path = problem[:-4] + 'pb'
# 
#                 run.add_command('asnet_create_pb_model_' + problem + '_' + conf_string,
#                                 ['python3', 'build_and_store_asnet_as_pb.py', domain, problem,
#                                  layer_num, str(loss_just_opt), weights_path, network_path])

# DOMAIN_NAME & MODEL_NAME will later be replaced in FastDownwardASNetRun
network_path = BENCHMARK_TRAINING + '/DOMAIN_NAME/training/MODEL_NAME'
exp.add_algorithm('asnet_search', REPO, 'default',
                  ['--search', 'policysearch(p=np(network=asnet(path=' + network_path + ')))'],
                  build_options=['release64dynamic'],
                  driver_options=['--build', 'release64dynamic'])

# # remove protobuf network file
# for task in exp._get_tasks():
#     problem = task.problem
#     run = exp.add_run()
#     run.add_command('remove_asnet_pb', ['rm', '-f', problem[:-4] + 'pb'])

report = os.path.join(exp.eval_dir, 'report.html')
exp.add_report(AbsoluteReport(attributes=ATTRIBUTES), outfile=report)

exp.run_steps()
