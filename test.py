#! /usr/bin/env python
# -*- coding: utf-8 -*-

from src.training.main import main
from src.training.bridges import FastDownwardSamplerBridge
from src.training.bridges.sampling_bridges import SampleFormat
import sys
import gzip

path_tmp = "TMP"


#main(sys.argv[1:])
search_args = """sampling(astar(ff(transform=sampling_transform(), register=ff), transform=sampling_transform()), techniques=[none_none(1), iforward_none(1, distribution=uniform_int_dist(5,15)), iforward_iforward(1, dist_init=uniform_int_dist(5,15),dist_goal=uniform_int_dist(25,45))], use_registered_heuristics=[ff], transform=adapt_costs(ONE))"""
fdb = FastDownwardSamplerBridge(search_args, SampleFormat.FD, True, ".", False)

fdb.sample("/home/ferber/repositories/benchmarks/transport-opt08-strips/p01.pddl", "test_sample")

