#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys

def generate_file(N):
    # Define
    s = "(define (problem hanoi-{0})\n".format(N)

    # Domain
    s += "  (:domain hanoi-domain)\n"

    # Objects
    k = ""
    for i in range(1,N+1):
        k += "d%i "%i
    s += "  (:objects peg1 peg2 peg3 %s)\n" % k

    # Init
    k = ""
    for i in range(1,N+1):
        k += "    (smaller d{0} peg1)(smaller d{0} peg2)(smaller d{0} peg3)\n".format(i)

    k += "\n"
    for i in range(1,N+1):
        k += "    "
        for j in range(i+1,N+1):
            k += "(smaller d%i d%i)" % (i,j)
        k += "\n"

    k += "    (clear p1)(clear p2)(clear d1)\n    "
    for i in range(1,N+1):
        k += "(disk d%i)"%i
    k += "\n    "
    for i in range(1,N):
        k += "(on d%i d%i)" % (i,i+1)
    k += "(on d%i peg3)\n" % N

    s += "  (:init \n%s  )\n" % k

    # Goal
    k = ""
    for i in range(1,N):
        k += "(on d%i d%i)" % (i,i+1)
    k += "(on d%i peg1)" % N

    s += "  (:goal \n    (and %s )\n  )" % k
    s += "\n)"


    if N < 10:
        diff_string = "0" + str(N)
    else:
        diff_string = str(N)
    problem_file = "d-" + diff_string + ".pddl"
    print("Generating " + problem_file + " hanoi problem file")
    with open(problem_file, 'w') as f:
        f.write(s)


def main(argv):
    N = 10 # default number of disks
    N_from = 0
    N_to = 0
    if len(sys.argv) == 2:
        if int(sys.argv[1]) <= 0:
            sys.stderr.write("N must be 1 or greater, default value used: "+str(N)+"\n")
            exit(-1)
        else:
            N = int(sys.argv[1])
    elif len(sys.argv) == 3:
        N_from = int(sys.argv[1])
        N_to = int(sys.argv[2])
        if N_from < 1 or N_from > N_to or (N_from + 50 < N_to):
            sys.stderr.write("Only use positive, valid intervals which do not include more than 50 (natural) numbers")
            exit(-1)
    else:
        sys.stderr.write("Usage: 'python towerHanoiPddlMaker.py N'\n")
        sys.stderr.write("or 'python towerHanoiPddlMaker.py N_from N_to'\n")
        sys.stderr.write("No integer N provided, default value used: "+str(N)+"\n")
    
    if N_from and N_to:
        for n in range(N_from, N_to + 1):
            generate_file(n)
    else:
        generate_file(N)


if __name__ == "__main__":
    main(sys.argv)
