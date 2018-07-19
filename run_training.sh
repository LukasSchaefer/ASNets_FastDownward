# config1 training
# python3 fast-asnet.py -t -d benchmarks/blocksworld/training --print_all --sort_problems > evaluation/network_runs/training/blocksworld/conf1/training.log
# cp benchmarks/blocksworld/training/asnet_final_weights.h5 evaluation/network_runs/training/blocksworld/conf1/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/elevator/training --print_all --sort_problems > evaluation/network_runs/training/elevator/conf1/training.log
# cp benchmarks/elevator/training/asnet_final_weights.h5 evaluation/network_runs/training/elevator/conf1/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/floortile/training --print_all --sort_problems > evaluation/network_runs/training/floortile/conf1/training.log
# cp benchmarks/floortile/training/asnet_final_weights.h5 evaluation/network_runs/training/floortile/conf1/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/hanoi/training --print_all --sort_problems > evaluation/network_runs/training/hanoi/conf1/training.log
# cp benchmarks/hanoi/training/asnet_final_weights.h5 evaluation/network_runs/training/hanoi/conf1/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/parcprinter/training --print_all --sort_problems > evaluation/network_runs/training/parcprinter/conf1/training.log
# cp benchmarks/parcprinter/training/asnet_final_weights.h5 evaluation/network_runs/training/parcprinter/conf1/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/sokoban/training --print_all --sort_problems > evaluation/network_runs/training/sokoban/conf1/training.log
# cp benchmarks/sokoban/training/asnet_final_weights.h5 evaluation/network_runs/training/sokoban/conf1/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/turnandopen/training --print_all --sort_problems > evaluation/network_runs/training/turnandopen/conf1/training.log
# cp benchmarks/turnandopen/training/asnet_final_weights.h5 evaluation/network_runs/training/turnandopen/conf1/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/tyreworld/training --print_all --sort_problems > evaluation/network_runs/training/tyreworld/conf1/training.log
# cp benchmarks/tyreworld/training/asnet_final_weights.h5 evaluation/network_runs/training/tyreworld/conf1/asnet_final_weights.h5

# config2 training
# python3 fast-asnet.py -t -d benchmarks/blocksworld/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/blocksworld/conf2/training.log
# cp benchmarks/blocksworld/training/asnet_final_weights.h5 evaluation/network_runs/training/blocksworld/conf2/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/elevator/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/elevator/conf2/training.log
# cp benchmarks/elevator/training/asnet_final_weights.h5 evaluation/network_runs/training/elevator/conf2/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/floortile/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/floortile/conf2/training.log
# cp benchmarks/floortile/training/asnet_final_weights.h5 evaluation/network_runs/training/floortile/conf2/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/hanoi/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/hanoi/conf2/training.log
# cp benchmarks/hanoi/training/asnet_final_weights.h5 evaluation/network_runs/training/hanoi/conf2/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/parcprinter/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/parcprinter/conf2/training.log
# cp benchmarks/parcprinter/training/asnet_final_weights.h5 evaluation/network_runs/training/parcprinter/conf2/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/sokoban/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/sokoban/conf2/training.log
# cp benchmarks/sokoban/training/asnet_final_weights.h5 evaluation/network_runs/training/sokoban/conf2/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/turnandopen/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/turnandopen/conf2/training.log
# cp benchmarks/turnandopen/training/asnet_final_weights.h5 evaluation/network_runs/training/turnandopen/conf2/asnet_final_weights.h5
# python3 fast-asnet.py -t -d benchmarks/tyreworld/training --print_all --sort_problems --teacher_search "astar(add(),transform=asnet_sampling_transform())" > evaluation/network_runs/training/tyreworld/conf2/training.log
# cp benchmarks/tyreworld/training/asnet_final_weights.h5 evaluation/network_runs/training/tyreworld/conf2/asnet_final_weights.h5

# conf3
python3 fast-asnet.py -t -d benchmarks/blocksworld/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/blocksworld/conf3/training.log
mv benchmarks/blocksworld/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/blocksworld/conf3/asnet_final_weights.h5
python3 fast-asnet.py -t -d benchmarks/elevator/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/elevator/conf3/training.log
mv benchmarks/elevator/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/elevator/conf3/asnet_final_weights.h5
python3 fast-asnet.py -t -d benchmarks/floortile/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/floortile/conf3/training.log
mv benchmarks/floortile/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/floortile/conf3/asnet_final_weights.h5
python3 fast-asnet.py -t -d benchmarks/hanoi/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/hanoi/conf3/training.log
mv benchmarks/hanoi/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/hanoi/conf3/asnet_final_weights.h5
python3 fast-asnet.py -t -d benchmarks/parcprinter/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/parcprinter/conf3/training.log
mv benchmarks/parcprinter/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/parcprinter/conf3/asnet_final_weights.h5
python3 fast-asnet.py -t -d benchmarks/sokoban/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/sokoban/conf3/training.log
mv benchmarks/sokoban/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/sokoban/conf3/asnet_final_weights.h5
python3 fast-asnet.py -t -d benchmarks/tyreworld/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/tyreworld/conf3/training.log
mv benchmarks/tyreworld/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/tyreworld/conf3/asnet_final_weights.h5
python3 fast-asnet.py -t -d benchmarks/turnandopen/training --print_all --sort_problems --teacher_search "lazy_greedy([ff()],preferred=[ff()],transform=asnet_sampling_transform())" > evaluation/network_runs/training/with_accumulate/turnandopen/conf3/training.log
mv benchmarks/turnandopen/training/asnet_final_weights.h5 evaluation/network_runs/training/with_accumulate/turnandopen/conf3/asnet_final_weights.h5
