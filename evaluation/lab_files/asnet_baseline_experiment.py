#! /usr/bin/env python

import os
from lab.environments import OracleGridEngineEnvironment
from downward.experiment import FastDownwardExperiment
from downward.reports.absolute import AbsoluteReport

NEURALREPO = '/mnt/data_server/schaefer/asnetsfastdownward'
VANILLAREPO = '/mnt/data_server/schaefer/fast-downward'
BENCHMARK = '/mnt/data_server/schaefer/asnetsfastdownward/benchmarks/evaluation_domains'
ENV = OracleGridEngineEnvironment(queue='all.q@@fai0x')
REVISION_CACHE = os.path.expanduser('~/lab/revision-cache')

SUITE = ['turnandopen', 'tyreworld', 'sokoban', 'hanoi', 'floortile', 'blocksworld', 'elevator', 'parcprinter']

ATTRIBUTES = ['unsolvable', 'memory', 'total_search_time', 'total_time', 'plan_length', 'cost', 'coverage', 'error']

exp = FastDownwardExperiment(environment=ENV, revision_cache=REVISION_CACHE)
exp.add_suite(BENCHMARK, SUITE)

# baseline planners:
# baseline 1: LAMA-2011 (executed with vanilla fast-downward)
exp.add_algorithm("lama", VANILLAREPO, "default", [],
        build_options=["release64"], driver_options=["--build", "release64", "--alias", "seq-sat-lama-2011"])

# baseline 2: A* with LM-Cut
exp.add_algorithm("astar_lmcut", NEURALREPO, "default", ["--search", "astar(lmcut())"],
        build_options=["release64dynamic"], driver_options=["--build", "release64dynamic"])

# baseline 3: lazy GBFS with FF-heuristic - not optimal (executed with vanilla fast-downward)
exp.add_algorithm("gbfs_ff", VANILLAREPO, "default", ["--heuristic", "hff=ff(transform=adapt_costs(cost_type=1))",
    "--search", "lazy_greedy([hff], preferred=[hff])"], build_options=["release64"],
    driver_options=["--build", "release64"])

report = os.path.join(exp.eval_dir, 'report.html')
exp.add_report(AbsoluteReport(attributes=ATTRIBUTES), outfile=report)

exp.run_steps()
