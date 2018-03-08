#include "sampling.h"

#include "successor_generator.h"

#include "../task_utils/task_properties.h"
#include "../utils/countdown_timer.h"
#include "../utils/rng.h"

using namespace std;


namespace sampling {

vector<State> sample_states_with_random_walks(
    TaskProxy task_proxy,
    const successor_generator::SuccessorGenerator &successor_generator,
    int num_samples,
    int init_h,
    double average_operator_cost,
    utils::RandomNumberGenerator &rng,
    function<bool (State) > is_dead_end,
    const utils::CountdownTimer *timer) {
    vector<State> samples;

    const State initial_state = task_proxy.get_initial_state();

    int n;
    if (init_h == 0) {
        n = 10;
    } else {
        /*
          Convert heuristic value into an approximate number of actions
          (does nothing on unit-cost problems).
          average_operator_cost cannot equal 0, as in this case, all operators
          must have costs of 0 and in this case the if-clause triggers.
         */
        assert(average_operator_cost != 0);
        int solution_steps_estimate = int((init_h / average_operator_cost) + 0.5);
        n = 4 * solution_steps_estimate;
    }
    double p = 0.5;
    /* The expected walk length is np = 2 * estimated number of solution steps.
       (We multiply by 2 because the heuristic is underestimating.) */

    samples.reserve(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        if (timer && timer->is_expired())
            throw SamplingTimeout();

        // Calculate length of random walk according to a binomial distribution.
        int length = 0;
        for (int j = 0; j < n; ++j) {
            double random = rng(); // [0..1)
            if (random < p)
                ++length;
        }


        samples.push_back(sample_state_with_random_forward_walk(
            task_proxy, successor_generator, initial_state,
            length, rng, is_dead_end));
    }
    return samples;
}

State sample_state_with_random_forward_walk(
    TaskProxy &task_proxy,
    const successor_generator::SuccessorGenerator &successor_generator,
    const State &initial_state,
    int length,
    utils::RandomNumberGenerator &rng,
    function<bool (State) > is_dead_end) {



    // Sample one state with a random walk of length length.
    State current_state(initial_state);
    vector<OperatorID> applicable_operators;
    for (int j = 0; j < length; ++j) {
        applicable_operators.clear();
        successor_generator.generate_applicable_ops(current_state,
            applicable_operators);
        // If there are no applicable operators, do not walk further.
        if (applicable_operators.empty()) {
            break;
        } else {
            OperatorID random_op_id = *rng.choose(applicable_operators);
            OperatorProxy random_op = task_proxy.get_operators()[random_op_id];
            assert(task_properties::is_applicable(random_op, current_state));
            current_state = current_state.get_successor(random_op);
            /* If current state is a dead end, then restart the random walk
               with the initial state. */
            if (is_dead_end(current_state))
                current_state = State(initial_state);
        }
    }
    // The last state of the random walk is used as a sample.
    return current_state;
}

vector<PartialAssignment> sample_partial_assignments_with_random_backward_walks(
    RegressionTaskProxy &regression_task_proxy,
    const predecessor_generator::PredecessorGenerator &predecessor_generator,
    int num_samples,
    int init_h,
    double average_operator_cost,
    utils::RandomNumberGenerator &rng,
    function<bool (PartialAssignment) > is_dead_end,
    const utils::CountdownTimer *timer) {
    vector<PartialAssignment> samples;

    const PartialAssignment initial_state = regression_task_proxy.get_goal_assignment();

    int n;
    if (init_h == 0) {
        n = 10;
    } else {
        /*
          Convert heuristic value into an approximate number of actions
          (does nothing on unit-cost problems).
          average_operator_cost cannot equal 0, as in this case, all operators
          must have costs of 0 and in this case the if-clause triggers.
         */
        assert(average_operator_cost != 0);
        int solution_steps_estimate = int((init_h / average_operator_cost) + 0.5);
        n = 4 * solution_steps_estimate;
    }
    double p = 0.5;
    /* The expected walk length is np = 2 * estimated number of solution steps.
       (We multiply by 2 because the heuristic is underestimating.) */

    samples.reserve(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        if (timer && timer->is_expired())
            throw SamplingTimeout();

        // Calculate length of random walk according to a binomial distribution.
        int length = 0;
        for (int j = 0; j < n; ++j) {
            double random = rng(); // [0..1)
            if (random < p)
                ++length;
        }


        samples.push_back(sample_partial_assignment_with_random_backward_walk(
            regression_task_proxy, predecessor_generator, initial_state,
            length, rng, is_dead_end));
    }
    return samples;
}

PartialAssignment sample_partial_assignment_with_random_backward_walk(
    RegressionTaskProxy &regression_task_proxy,
    const predecessor_generator::PredecessorGenerator &predecessor_generator,
    const PartialAssignment goal_assignment,
    int length,
    utils::RandomNumberGenerator &rng,
    function<bool (PartialAssignment) > is_dead_end) {



    // Sample one state with a random walk of length length.
    PartialAssignment current_state(goal_assignment);
    vector<OperatorID> applicable_operators;
    for (int j = 0; j < length; ++j) {
        applicable_operators.clear();
        predecessor_generator.generate_applicable_ops(current_state,
            applicable_operators);
        // If there are no applicable operators, do not walk further.
        if (applicable_operators.empty()) {
            break;
        } else {
            OperatorID random_op_id = *rng.choose(applicable_operators);
            RegressionOperatorProxy random_op = regression_task_proxy.get_regression_operator(random_op_id);
            assert(task_properties::is_applicable(random_op, current_state));
            current_state = random_op.get_anonym_predecessor(current_state);
            /* If current state is a dead end, then restart the random walk
               with the initial state. */
            if (is_dead_end(current_state))
                current_state = PartialAssignment(goal_assignment);
        }
    }
    // The last state of the random walk is used as a sample.
    return current_state;
}

/*
pair<State, int> sample_state_with_regression_random_walk(
    const Task &task,
    int solution_steps_estimate) {
    // Start in partial goal state.
    State partial_state = *task.get_initial_regression_state();

    // The expected walk length is np = estimated number of solution steps.
    int n = 2 * solution_steps_estimate;
    double p = 0.5;

    // Calculate length of random walk according to a binomial distribution.
    int length = 0;
    for (int j = 0; j < n; ++j) {
        double random = rng(); // [0..1)
        if (random < p)
            ++length;
    }

    // Sample one state with a random walk of 'length' steps.
    int distance = 0;
    for (int j = 0; j < length; ++j) {
        vector<const Operator *> applicable_ops =
            task.get_applicable_operators(partial_state);
        if (applicable_ops.empty()) {
            // If there are no reverse applicable operators, do not walk further.
            break;
        } else {
            const Operator *op = *rng.choose(applicable_ops);
            partial_state = State(partial_state, *op);
            distance += op->get_cost();
        }
    }
    return make_pair(partial_state, distance);
}
 */




}
