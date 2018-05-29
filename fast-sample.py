#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys

# Load dependency module w/o loading the whole package (otherwise,
# changing the dependencies will have no effect anymore
if sys.version_info >= (3, 5):
    import importlib.util
    spec = importlib.util.spec_from_file_location("src.training.dependencies", "src/training/dependencies.py")
    dependencies = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dependencies)
    sys.modules["src.training.dependencies"] = dependencies
    dependencies.setup()
    dependencies.set_external(False, True)
elif sys.version_info < (3,):
    import imp
    dependencies = imp.load_source("src.training.dependencies", "src/training/dependencies.py")
    sys.modules["src.training.dependencies"] = dependencies
    dependencies.setup()
    dependencies.set_external(False, True)
else:
    print("Warning: Dependency preloading not supported by this python version."
          " All dependencies are require.")


from src.training.bridges import FastDownwardSamplerBridge
from src.training.samplers import IterableFileSampler
from src.training.samplers import DirectorySampler
from src.training.bridges.sampling_bridges import StateFormat

import argparse
import shlex
import subprocess
import sys
import logging
log = logging.getLogger()


# DEFAULT SEARCH CONFIGURATION
DEFAULT_TRANSFORMATIONS = [
    "none_none(1)"
    "iforward_none(25, distribution=uniform_int_dist(5,15))",
    "iforward_iforward(15, dist_init=uniform_int_dist(5,15),dist_goal=uniform_int_dist(1,5))"
]
DEFAULT_STR_TRANSFORMATIONS = ", ".join(DEFAULT_TRANSFORMATIONS)
DEFAULT_MAX_TIME_PER_SAMPLING = "10m"
DEFAULT_SEARCH = ("sampling(astar(lmcut(transform=sampling_transform(), "
                  "register=ff), transform=sampling_transform(), max_time=%s), "
                  "techniques=[%s], use_registered_heuristics=[ff], "
                  "transform=adapt_costs(ONE), max_time=120m)"
                  % (DEFAULT_MAX_TIME_PER_SAMPLING, DEFAULT_STR_TRANSFORMATIONS))

CHOICES_FORMAT = []
for name in StateFormat.name2obj:
    CHOICES_FORMAT.append(name)

# Argparser Configurations
DESCRIPTION = """Fast-Downward Data Sampling

Samples data using Fast-Downward. The script has the following two modes which
can be defined multiple times
\t--sample: Samples states for a given list of problems and options
\t--traverse: Traverses a directory and invokes sampling for the detected problems.

The script starts in the --sample mode and allows the user to define a 
sampling for a list of problems he/she manually provides. This can be repeated
multiple times.
The first --traverse block changes this behaviour. The --traverse block defines
how to traverse through a directory and detects problem files. ALL --sampling
blocks after a --traverse block, but before the next traverse block (if
existing) are called with the list of problems detected by the current traverse
block. A --traverse block without associated --sample block is skipped.

Use -h within a block to show the help menu for the block (this ends the processing
of the block and continues with the next block)

Example:
./SCRIPT PROBLEM1 PROBLEM 2 OPTIONS - sample from PROBLEM1 & PROBLEM2 with options
./SCRIPT --sample P1 P2 OPTIONS - sample from P1 & P2 with options
./SCRIPT P1 P2 OPTION1 --traverse OPTIONS2 --sample OPTION3 --sample OPTION4 
      - first samples from P1 & P2 with OPTION1, then traverse with OPTION2 and
        detects problem files p1, ..., pn. Afterwards, it calls sampling for
        p1, ..., pn with OPTION3 and finally with OPTION4. More --traverse and
        --sample blocks could follow follow"""

psample = argparse.ArgumentParser(description="Sample block arguments:")

psample.add_argument("problem", nargs="*", type=str, action="store",
                     help="Path to a problem to sample from. Multiple problems"
                         "can be given.")

psample.add_argument("-a", "--append",
                     action="store_true",
                     help="If sample file exists already append new samples "
                         "instead of overwriting.")
psample.add_argument("-b", "--build", type=str,
                     action="store", default="debug64dynamic",
                     help="Name of the build to use")
psample.add_argument("-c", "--compress", action="store_true",
                     help="Store the sampled entries in a compressed file")
psample.add_argument("-d", "--domain",
                     type=str, action="store", default=None,
                     help="Path to the domain file used by all problems (if not"
                         "given, then the domain file is automatically searched"
                         "for every problem individually).")
psample.add_argument("-f", "--format", choices=CHOICES_FORMAT,
                     action="store", default=StateFormat.Full.name,
                     help="State format in the sampled file.")
psample.add_argument("-fd", "--fast-downward", type=str,
                     action="store", default="./fast-downward.py",
                     help="Path to the fast-downward script (Default"
                          "assumes in current directory fast-downward.py).")
psample.add_argument("-p", "--prune", action="store_true",
                     help="Prune duplicate entries")
psample.add_argument("-r", "--reuse", action="store_true",
                     help="Tells sampler to reuse instead of resample data, if"
                          "for the given problem data already exists. Remark:"
                          "In this context it means, we simply skip sampling for"
                          "a problem if data was previously sampled. It does NOT"
                          "check with which parameters the present data was"
                          "sampled")
psample.add_argument("-s", "--search", type=str,
                     action="store", default=DEFAULT_SEARCH,
                     help="Search for sampling to start in Fast-Downward (the"
                         "given search has to perform the sampling, this script"
                         " is not wrapping your search in a sampling search)")
psample.add_argument("-t", "--target-folder", type=str,
                     action="store", default=None,
                     help="Folder to store for each problem a data file. By"
                         "default the sampled data is stored in the same place"
                         "where its associated problem file is.")
psample.add_argument("-tf", "--target-file", type=str,
                     action="store", default=None,
                     help="Path to file to store for all problem samples. This"
                         " argument cannot be used together with "
                         "\"--target-file\".")
psample.add_argument("-tmp", "--temporary-folder", type=str,
                     action="store", default=None,
                     help="Folder to store temporary files. By default the same"
                         " directory is used where the used problem file is"
                         " stored.")


def parse_sample_args(argv):
    options = psample.parse_args(argv)
    options.format = StateFormat.get(options.format)
    if options.target_file is not None and options.target_folder is not None:
        raise argparse.ArgumentError("\"--target-folder\" and \"--target-file\""
                                     " are two mutually exclusive options."
                                     " Please use chose one of them.")
    if options.target_file is not None:
        options.append = True


    fdb = FastDownwardSamplerBridge(options.search, options.format,
                                    options.build, options.temporary_folder,
                                    options.target_file, options.target_folder,
                                    options.append, False, 0.0, options.reuse,
                                    options.domain, True,
                                    options.fast_downward,
                                    options.prune, True,
                                    options.compress)
    return fdb, options.problem


def sample(argv):
    fdb, problems = parse_sample_args(argv)
    if len(problems) == 0:
        log.warning("No problems defined for sampling.")
    ifs = IterableFileSampler(fdb, problems)
    ifs.initialize()
    ifs.sample()
    ifs.finalize()


ptraverse = argparse.ArgumentParser(description="Traverse block arguments:")

ptraverse.add_argument("root", nargs="+", type=str, action="store",
                     help="Path to root directory of a traversal. Multiple"
                          "roots can be given, the traversals from each root"
                          "are independent.")

ptraverse.add_argument("-a", "--args", type=str,
                     action="store", default=None,
                     help="Single string describing a set of arguments to add"
                          "before the sampling call arguments (this is set even"
                          "before the list of problems and can be used for "
                          "passing arguments to an external script which "
                          "performs the sampling.")
ptraverse.add_argument("-b", "--batch", type=int,
                     action="store", default=None,
                     help="WORKS ONLY WITH --execute TOGETHER. Submit the"
                          "problems found during traversal in batches to the"
                          "script to execute.")
ptraverse.add_argument("-df", "--directory-filter", type=str,
                     action="append", default=[],
                     help="A subdirectory name has to match the regex otherwise"
                          "it is not traversed. By default no regex matches are"
                          "required. This argument can be given any number of"
                          "time to add additional filters (the directory name has"
                          "to match ALL regexes)")
ptraverse.add_argument("-e", "--execute", type=str,
                     action="store", default=None,
                     help="Path to script to execute for the sampling runs. If"
                          "none is given, then the scripts samples in its own"
                          "process, otherwise"
                          "it calls an external script in a subprocess.")
ptraverse.add_argument("-m", "--max-depth", type=int,
                     action="store", default=None,
                     help="Maximum depth from the root which is traversed ("
                          "default has no maximum, 0 means traversing no"
                          "subfolders, only the content of the root)")
ptraverse.add_argument("-p", "--problem-filter", type=str,
                     action="append", default=[],
                     help="A problem file name has to match the regex otherwise"
                          "it is not registered. By default no regex matches are"
                          "required. This argument can be given any number of"
                          "time to add additional filters (the file name has"
                          "to match ALL regexes)")
ptraverse.add_argument("-s", "--selection-depth", type=int,
                     action="store", default=None,
                     help="Minimum depth from the root which has to be traversed"
                          " before problem files are registered (default has "
                          "no minimum)")
ptraverse.add_argument("--dry", action="store_true",
                     help="Show only the call arguments it should do, but does "
                          "not perform calls (and therefore, samplings)")


def traverse_directories(argv, sample_settings):
    options = ptraverse.parse_args(argv)

    options.args = [] if options.args is None else shlex.split(options.args)
    for idx in range(len(sample_settings)):
        sample_settings[idx] = options.args + sample_settings[idx]
        if options.execute is None:
            sample_settings[idx] = parse_sample_args(sample_settings[idx])[0]
        else:
            sample_settings[idx].insert(0, options.execute)

    ds = DirectorySampler([] if options.execute is not None else sample_settings,
                          options.root,
                          options.directory_filter, options.problem_filter,
                          options.max_depth, options.selection_depth)
    ds.initialize()

    if options.dry:
        print("Problems: ", ds._iterable)
        for setting in sample_settings:
            print("\tSetting: ", setting)
        return

    if options.execute is None:
        ds.sample()
        ds.finalize()
    else:
        ins_idx = len(options.args) + 1
        for setting in sample_settings:
            local_settings = list(setting)
            start = 0
            step = len(ds._iterable) if options.batch is None else options.batch
            while start < len(ds._iterable):
                end = start + step
                local_settings[ins_idx:ins_idx] = ds._iterable[start:end]
                start = end


                subprocess.call(local_settings)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ["h", "help", "-h", "-help", "--help"]:
        print(DESCRIPTION, "\n\n")
        psample.print_help(sys.stdout)
        print("\n\n")
        ptraverse.print_help(sys.stdout)
    else:

        mode = 0 # 0=ind. samp, 1 = traverse block, 2=dep. sample block
        independent_sample_runs = []
        traversing = []
        last_idx = 1

        def process_block(idx):
            if last_idx == idx == 1:
                return
            block = sys.argv[last_idx:idx]

            if mode == 0:
                independent_sample_runs.append(block)
            elif mode == 1:
                traversing.append((block, []))
            elif mode == 2:
                traversing[-1][1].append(block)
            else:
                raise RuntimeError("Internal Error during separation of blocks")


        for idx in range(1, len(sys.argv)):
            next = sys.argv[idx]
            if next not in["--traverse", "--sample"]:
                continue

            process_block(idx)
            if next == "--traverse":
                mode = 1
            elif mode == 1:
                mode = 2

            last_idx = idx + 1
        process_block(len(sys.argv))

        for independent in independent_sample_runs:
            sample(independent)
        for (traverse_options, sample_settings) in traversing:
            for s in sample_settings:
                traverse_directories(traverse_options, sample_settings)

