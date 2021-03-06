#ifndef TASK_UTILS_SAMPLING_H
#define TASK_UTILS_SAMPLING_H

#include "regression_utils.h"

#include "../task_proxy.h"
#include "predecessor_generator.h"

#include <exception>
#include <functional>
#include <memory>
#include <vector>

class State;
class TaskProxy;
class PartialAssignment;

namespace successor_generator {
    class SuccessorGenerator;
}

namespace utils {
    class CountdownTimer;
    class RandomNumberGenerator;
}


namespace sampling {

    struct SamplingTimeout : public std::exception {
    };

    /*
      Perform 'num_samples' random walks with biomially distributed walk
      lenghts. Whenever a dead end is detected or a state has no
      successors, restart from the initial state. The function
      'is_dead_end' should return whether a given state is a dead end. If
      omitted, no dead end detection is performed. If 'timer' is given the
      sampling procedure will run for at most the specified time limit and
      possibly return less than 'num_samples' states.
     */
    std::vector<State> sample_states_with_random_walks(
            TaskProxy task_proxy,
            const successor_generator::SuccessorGenerator &successor_generator,
            int num_samples,
            int init_h,
            double average_operator_cost,
            utils::RandomNumberGenerator &rng,
            std::function<bool(State) > is_dead_end = [] (const State &) {
                return false;
            },
    const utils::CountdownTimer *timer = nullptr);

    /*
      Performs a single sampling run with forward walks of length 'length'.
      Whenever a dead end is detected or a state has no
      successors, restart from the initial state. The function
      'is_dead_end' should return whether a given state is a dead end. If
      omitted, no dead end detection is performed.
     */
    State sample_state_with_random_forward_walk(
            TaskProxy &task_proxy,
            const successor_generator::SuccessorGenerator &successor_generator,
            const State &initial_state,
            int length,
            utils::RandomNumberGenerator &rng,
            std::function<bool(State) > is_dead_end = [] (const State &) {
                return false;
            });


    std::vector<PartialAssignment> sample_partial_assignments_with_random_backward_walks(
            RegressionTaskProxy &regression_task_proxy,
            const predecessor_generator::PredecessorGenerator &predecessor_generator,
            int num_samples,
            int init_h,
            double average_operator_cost,
            utils::RandomNumberGenerator &rng,
            std::function<bool(PartialAssignment) > is_dead_end = [] (const PartialAssignment &) {
                return false;
            },
    const utils::CountdownTimer *timer = nullptr);

    PartialAssignment sample_partial_assignment_with_random_backward_walk(
            RegressionTaskProxy &regression_task_proxy,
            const predecessor_generator::PredecessorGenerator &predecessor_generator,
            const PartialAssignment goal_assignment,
            int length,
            utils::RandomNumberGenerator &rng,
            std::function<bool(PartialAssignment) > is_dead_end = [] (const PartialAssignment &) {
                return false;
            });
}


#endif
