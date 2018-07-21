#! /usr/bin/env python
# -*- coding: utf-8 -*-

# script to extract training statistics out of training log files

import re
import sys
import os


class ProblemTrainingParsed:
    def  __init__(self, problem_name):
        self.problem_name = problem_name
        self.epochs = 0
        # nested list with
        # first list corresponds to epochs
        # second list corresponds for problem epochs in this epoch
        # second list contains the loss values for each training epoch
        self.loss_values_by_epochs = []
        # same structure as loss values with values being pairs of format
        # (act_name, probability) for action with act_name being chosen in the
        # network search of the sampling search with probability
        self.chosen_act_name_and_prob = []
        # #epochs list-entries with each being the time it took to create the
        # keras model
        self.model_creation_times = []
        # list of nested lists corresponding to epochs containing time it took
        # to complete the sampling search for each problem epoch
        self.sampling_search_times = []
        # list of nested lists corresponding to epochs containing time it took
        # to complete the training run for each problem epoch
        self.training_epoch_times = []
        # list of ints indicating how many of the network searches in the problem epochs
        # of the respective epoch were successfull = reaching a goal
        self.successfull_explorations = []

    
    def parse_model_creation(self, log_lines, log_line_index, current_epoch):
        """
        :param log_lines: list of all lines of the log file
        :param log_line_index: indicating which line from log_lines should be processed next
        :param current_epoch: index number of current epoch
        """
        line = log_lines[log_line_index].strip()
        match = re.match(r'Keras model creation time: ([0-9]+\.[0-9]*)s', line)
        while not match:
            log_line_index += 1
            line = log_lines[log_line_index].strip()
            match = re.match(r'Keras model creation time: ([0-9]+\.[0-9]*)s', line)
        self.model_creation_times.append(match.group(1))
        assert len(self.model_creation_times) == (current_epoch + 1)
        log_line_index += 1
        return log_line_index


    def parse_problem_epoch(self, log_lines, log_line_index, current_epoch, problem_epoch):
        """
        Beginning at line "Starting problem epoch x / y"

        :param log_lines: list of all lines of the log file
        :param log_line_index: indicating which line from log_lines should be processed next
        :param current_epoch: index number of current epoch
        :param problem_epoch: index number of problem epoch
        """
        problem_epoch_losses = []
        problem_epoch_act_name_probs = []

        log_line_index += 1
        line = log_lines[log_line_index].strip()
        # jump over translation and starting procedure/ prints of sampling search
        while line != 'Conducting policy search':
            log_line_index += 1
            line = log_lines[log_line_index].strip()
        log_line_index += 1
        # now on "Policy reached state with id ..."
        line = log_lines[log_line_index].strip()
        get_op_prob_string = r'Policy reached state with id #[0-9]+ by applying op ([a-zA-Z0-9-_\s]+) which had probability ([0-9]+\.?[0-9]*)'
        match = re.match(get_op_prob_string, line)
        while match:
            act_pair = (match.group(1), float(match.group(2)))
            problem_epoch_act_name_probs.append(act_pair)
            log_line_index += 1
            line = log_lines[log_line_index].strip()
            match = re.match(get_op_prob_string, line)

        self.chosen_act_name_and_prob[-1].append(problem_epoch_act_name_probs)

        # extract sampling search total time
        line = log_lines[log_line_index].strip()
        sampling_search_string = r'Sampling search time: ([0-9]+\.[0-9]*)s'
        match = re.match(sampling_search_string, line)
        while not match:
            log_line_index += 1
            line = log_lines[log_line_index].strip()
            match = re.match(sampling_search_string, line)

        self.sampling_search_times[-1].append(float(match.group(1)))

        # extract all losses
        # jump to first "1/1 [=====...] ... - loss: x" line
        log_line_index += 6
        line = log_lines[log_line_index].strip()
        loss_string = r'[\w\W]* - loss: ([0-9]+\.[0-9]*)'
        while not line.startswith('Network training time'):
            match = re.match(loss_string, line)
            if match:
                loss = float(match.group(1))
                problem_epoch_losses.append(loss)
            # go to next line
            log_line_index += 1
            line = log_lines[log_line_index].strip()
        self.loss_values_by_epochs[-1].append(problem_epoch_losses)
        
        line = log_lines[log_line_index].strip()
        training_time_string = r'Network training time: ([0-9]+\.[0-9]*)s'
        match = re.match(training_time_string, line)
        self.training_epoch_times[-1].append(float(match.group(1)))

        log_line_index += 1
        return log_line_index


    def parse_epoch(self, log_lines, log_line_index, current_epoch):
        """
        Starts at "Building keras ASNet model" line
        Finishes at empty line after "x / y network explorations were successfull for this problem"

        :param log_lines: list of all lines of the log file
        :param log_line_index: indicating which line from log_lines should be processed next
        :param current_epoch: index number of current epoch
        """
        # initialize values with new list for epoch
        self.epochs += 1
        self.loss_values_by_epochs.append([])
        self.chosen_act_name_and_prob.append([])
        self.sampling_search_times.append([])
        self.training_epoch_times.append([])

        line = log_lines[log_line_index].strip()
        assert line == 'Building keras ASNet model'
        log_line_index += 1
        log_line_index = self.parse_model_creation(log_lines, log_line_index, current_epoch)

        # jump to start of problem epoch
        log_line_index += 4
        line = log_lines[log_line_index].strip()
        match = re.match(r'Starting problem epoch (\d) / \d', line)
        while match:
            log_line_index = self.parse_problem_epoch(log_lines, log_line_index, current_epoch, int(match.group(1)) - 1)
            line = log_lines[log_line_index].strip()
            if line == '':
                log_line_index += 1
                line = log_lines[log_line_index].strip()
            elif line.startswith('Network finalization time:'):
                log_line_index += 2
                break
            match = re.match(r'Starting problem epoch (\d) / \d', line)
        
        if line.startswith("Training is taking over "):
            # early stopping at this point
            return log_line_index + 1

        # on "x / y network explorations were successfull for this problem"
        line = log_lines[log_line_index].strip()
        match = re.match(r'(\d) / \d network explorations were successfull for this problem', line)
        assert match
        self.successfull_explorations.append(int(match.group(1)))
        log_line_index += 1
        return log_line_index


    def dump_problem_epoch(self, epoch_index, problem_epoch_index):
        print("problem epoch data for epoch %d, problem epoch %d" % (epoch_index + 1, problem_epoch_index + 1))
        print("\tsampling search time: %ss" % self.sampling_search_times[epoch_index][problem_epoch_index])
        print("\tduring this search the following actions were chosen:")
        for act_name, probability in self.chosen_act_name_and_prob[epoch_index][problem_epoch_index]:
            print("\t\t%s was chosen with probability %f" % (act_name, probability))
        print("\ttraining time: %ss" % self.training_epoch_times[epoch_index][problem_epoch_index])
        print("\tduring the training the following losses were computed:")
        losses = self.loss_values_by_epochs[epoch_index][problem_epoch_index]
        for loss in losses:
            print("\t\tloss: %f" % loss)
        print("\tOverall the loss development was %f -> %f" % (losses[0], losses[-1]))


    def dump_epoch(self, epoch_index):
        print("Training data for problem %s in epoch %d:" % (self.problem_name, epoch_index + 1))
        print("model creation time: %ss" % self.model_creation_times[epoch_index])
        for problem_epoch_index in range(len(self.sampling_search_times[epoch_index])):
            self.dump_problem_epoch(epoch_index, problem_epoch_index)
        if len(self.successfull_explorations) > epoch_index:
            print("In the epoch %d for problem %s %d explorations in the sampling searches reached a goal"\
                    % (epoch_index + 1, self.problem_name, self.successfull_explorations[epoch_index]))

    
    def dump(self):
        print("Training log for problem: %s" % self.problem_name)
        for epoch_index in range(len(self.sampling_search_times)):
            self.dump_epoch(epoch_index)
            print()
            print()


class TrainingParser:
    def __init__(self, log_lines, domain_name):
        """
        :param log_lines: list of all lines of the log file
        :param domain_name: name of domain the training was run for
        """
        self.log_lines = log_lines
        self.domain_name = domain_name
        self.log_line_index = 0
        self.current_epoch = -1
        # dict accessing parsed problems by problem name
        self.parsed_problems = {}
        # success rates per epoch
        self.success_rates = []


    def parse_epoch(self):
        """
        Parse exactly one epoch of the log file
        Starts with the line just AFTER "Starting epoch x"
        Ends on the line "Starting epoch x" or "Saving final weights ..."
        """
        # starts at "Training already taskes ..."
        self.log_line_index += 1
        line = self.log_lines[self.log_line_index].strip()
        # first follows an empty line
        assert line == ""
        self.log_line_index += 1
        line = self.log_lines[self.log_line_index].strip()
        match = re.match(r'Processing problem file benchmarks[\d]*\/' + self.domain_name + r'\/training\/([a-zA-Z0-9-_]*\.pddl)\s', line)
        while match:
            problem_name = match.group(1)
            self.log_line_index += 1
            if problem_name not in self.parsed_problems.keys():
                parsed_problem = ProblemTrainingParsed(problem_name)
                self.log_line_index = parsed_problem.parse_epoch(self.log_lines, self.log_line_index, self.current_epoch)
                self.parsed_problems[problem_name] = parsed_problem
            else:
                parsed_problem = self.parsed_problems[problem_name]
                self.log_line_index = parsed_problem.parse_epoch(self.log_lines, self.log_line_index, self.current_epoch)
            line = self.log_lines[self.log_line_index].strip()
            if line.startswith('Epochs success rate:'):
                match = re.match(r'Epochs success rate: (\d+)', line)
                success_rate = int(match.group(1))
                self.success_rates.append(success_rate)
                self.log_line_index += 1
                line = self.log_lines[self.log_line_index].strip()

            if line == '':
                self.log_line_index += 1
                line = self.log_lines[self.log_line_index].strip()               
            elif line.startswith('Saving final weights in'):
                return
            elif line == 'EARLY TRAINING TERMINATION:':
                self.log_line_index += 2
                return
            else:
                assert line.startswith('Epochs success rate:')
                match = re.match(r'Epochs success rate: (\d+)', line)
                success_rate = int(match.group(1))
                self.success_rates.append(success_rate)
                self.log_line_index += 2
                line = self.log_lines[self.log_line_index].strip()               
            match = re.match(r'Processing problem file benchmarks[\d]*\/' + self.domain_name + r'\/training\/([a-zA-Z0-9-_]*\.pddl)\s', line)


    def parse(self):
        """
        Parse entire logfile
        """
        line = self.log_lines[self.log_line_index].strip()
        while line == '' or line.startswith('Parsing time'):
            self.log_line_index += 1
            line = self.log_lines[self.log_line_index].strip()

        starting_epoch_string = r'Starting epoch ([0-9]*)'
        match = re.match(starting_epoch_string, line)
        while match:
            self.log_line_index += 1
            epoch = int(match.group(1))
            self.current_epoch += 1
            assert (self.current_epoch + 1) == epoch, "Epoch counter seems wrong! Was %d but expected to be" % (self.current_epoch, epoch - 1)
            self.parse_epoch()
            line = self.log_lines[self.log_line_index].strip()
            match = re.match(starting_epoch_string, line)
        # Last epoch is finished -> done
        line = self.log_lines[self.log_line_index].strip()
        assert line.startswith('Saving final weights in')
        self.log_line_index += 1
        line = self.log_lines[self.log_line_index].strip()
        match = re.match(r'Entire training time: ([0-9]+\.[0-9]*)s', line)
        if match:
            self.entire_training_time = float(match.group(1))
        else:
            print("No entire training time given!")
        self.number_of_epochs = self.current_epoch + 1


    def dump_per_problem(self):
        print("Training log data for domain %s:" % self.domain_name)
        print("printing the data problem-wise")
        for parsed_problem in self.parsed_problems.values():
            parsed_problem.dump()
            print()


    def dump_chronological(self):
        print("Training log data for domain %s:" % self.domain_name)
        print("printing the data chronological")
        for epoch_index in range(self.current_epoch + 1):
            print("Epoch %d:" % (epoch_index + 1))
            for parsed_problem in self.parsed_problems.values():
                if parsed_problem.epochs > epoch_index:
                    parsed_problem.dump_epoch(epoch_index)
            if epoch_index < len(self.success_rates):
                print("Success rate: %d" % (self.success_rates[epoch_index]))
            print()


def main(argv):
    if len(argv) != 3:
        print("Usage: ./training_parser.py <path/to/training.log> <Domain-name>")
        sys.exit(1)
    training_log_path = argv[1]
    domain_name = argv[2]

    log_file = open(training_log_path, 'r')
    log_lines = [l.strip() for l in log_file.readlines()]

    parser = TrainingParser(log_lines, domain_name)
    parser.parse()
    parser.dump_chronological()


if __name__ == "__main__":
    main(sys.argv)
