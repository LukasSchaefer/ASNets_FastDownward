#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from subprocess import check_output

def main(argv):
    if not len(argv) == 3:
        print("Usage: ./generate_problems.py <n_from> <n_to>")
        print("to generate tyreworld problems with <n_from> until <n_to> number of tyres")
        raise ValueError("Wrong number of arguments!")
    
    n_from = int(argv[1])
    n_to = int(argv[2])

    if n_from < 0 or n_from > n_to or (n_from + 50 < n_to):
        print("Only use positive, valid intervals which do not include more than 50 (natural) numbers")

    for n in range(n_from, n_to + 1):
        print("Generating ../p" + str(n) + ".pddl tyreworld problem file")
        cmd = ("./tyreworld", "-n", str(n))

        problem_file = "../p" + str(n) + ".pddl"

        with open(problem_file, 'w') as f:
            out = check_output(cmd)
            f.write(out.decode("utf-8"))


if __name__ == "__main__":
    main(sys.argv)
