#include "asnet_sampling_search.h"

#include "../evaluation_context.h"
#include "../option_parser.h"
#include "../plugin.h"

#include "../tasks/modified_init_goals_task.h"
#include "../task_utils/successor_generator.h"
#include "../task_utils/lexicographical_access.h"
#include "../utils/memory.h"

#include <fstream>

using namespace std;

namespace asnet_sampling_search {

template <typename t> void vector_into_stream(vector<t> vec, ostringstream &oss) {
    oss << "[";
    std::move(vec.begin(), vec.end()-1, std::ostream_iterator<t>(oss, ","));
    oss << vec.back() << "]";
}

vector<int> reverse_operator_sorted_vec(vector<int> operator_indeces_sorted) {
    vector<int> operator_indeces_sorted_reversed(operator_indeces_sorted.size());
    for (unsigned int sorted_index = 0; sorted_index < operator_indeces_sorted.size(); sorted_index++) {
        int unsorted_index = operator_indeces_sorted[sorted_index];
        operator_indeces_sorted_reversed[unsorted_index] = sorted_index;
    }
    return operator_indeces_sorted_reversed;
}

ASNetSamplingSearch::ASNetSamplingSearch(const Options &opts)
: SearchEngine(opts),
  search_parse_tree(prepare_search_parse_tree(opts.get_unparsed_config())),
  problem_hash(opts.get<string>("hash")),
  target_location(opts.get<string>("target")),
  use_non_goal_teacher_paths(opts.get<bool>("use_non_goal_teacher_paths")),
  use_teacher_search(opts.get<bool>("use_teacher_search")),
  additional_input_features(opts.get<string>("additional_input_features")),
  facts_sorted(lexicographical_access::get_facts_lexicographically(task_proxy)),
  operator_indeces_sorted(lexicographical_access::get_operator_indeces_lexicographically(task_proxy)),
  operator_indeces_sorted_reversed(reverse_operator_sorted_vec(operator_indeces_sorted)),
  network_search(opts.get<shared_ptr<SearchEngine>>("network_search")) {
    if (find(supported_additional_input_features.begin(), supported_additional_input_features.end(),
        additional_input_features) == supported_additional_input_features.end()) {
        // additional input features are not supported
        throw "Additional input features " + additional_input_features + "are not supported";
    }
    if (additional_input_features == "landmarks" || additional_input_features == "binary_landmarks") {
        landmark_generator = utils::make_unique_ptr<lm_cut_heuristic::LandmarkCutLandmarks>(task_proxy);
    }
}

options::ParseTree ASNetSamplingSearch::prepare_search_parse_tree(
    const std::string& unparsed_config) const {
    options::ParseTree pt = options::generate_parse_tree(unparsed_config);
    return subtree(pt, options::first_child_of_root(pt));
}

/*
  * function extracts fact_goal_values.
  * These are binary value for every fact indicating whether the fact is part of the goal. Values
  * are ordered lexicographically by fact-names in a "," separated list form.
*/
void ASNetSamplingSearch::goal_into_stream(ostringstream &goal_stream) const {
    vector_into_stream<int>(fact_goal_values, goal_stream);
}

/*
  * function extracts fact_values.
  * These are binary value for every fact indicating whether the fact is currently true in the state.
  * The values are ordered lexicographically by fact-names in a "," separated list form.
*/
void ASNetSamplingSearch::state_into_stream(const GlobalState &state, ostringstream &state_stream) const {
    unsigned long number_of_facts = facts_sorted.size();
    vector<int> fact_values;
    fact_values.resize(number_of_facts);
    for (unsigned int fact_index = 0; fact_index < number_of_facts; fact_index++) {
        pair<int, int> fact = facts_sorted[fact_index];
        if (state[fact.first] == fact.second) {
            // fact_index th fact is currently true
            fact_values[fact_index] = 1;
        } else {
            // fact_index th fact is currently not true
            fact_values[fact_index] = 0;
        }
    }
    vector_into_stream<int>(fact_values, state_stream);
}

/*
  * function extracts action_applicable_values.
  * These are binary value for every action indicating whether it is applicable in the state.
  * The values are ordered lexicographically by action-names in a "," separated list form.
*/
vector<int> ASNetSamplingSearch::applicable_values_into_stream(
            const GlobalState &state, const OperatorsProxy &ops,
	    ostringstream &applicable_stream) const {
    // collect all applicable actions in state
    vector<OperatorID> applicable_op_ids;
    g_successor_generator->generate_applicable_ops(state, applicable_op_ids);

    unsigned long number_of_operators = operator_indeces_sorted.size();
    vector<int> applicable_values;
    applicable_values.resize(number_of_operators);

    for (unsigned int op_index = 0; op_index < number_of_operators; op_index++) {
        int unsorted_index = operator_indeces_sorted[op_index];
        OperatorID op_id = ops[unsorted_index].get_global_operator_id();
        // check whether op_id is among the ids of applicable operators
        if (std::find(applicable_op_ids.begin(), applicable_op_ids.end(), op_id) != applicable_op_ids.end()) {
            // op is applicable in state
            applicable_values[op_index] = 1;
        } else {
            // op is not applicable in state
            applicable_values[op_index] = 0;
        }
    }
    vector_into_stream<int>(applicable_values, applicable_stream);
    return applicable_values;
}

/*
  * SHOULD NOT BE NECESSARY BECAUSE THESE VALUES WOULD ONLY BE USED FOR THE LOSS BUT THERE WE HAVE
  * THE COMPUTED PREDICTION = PROBABILITIES EITHER WAY
  * 
  * function extracts action_network_probs:
  * These are float value for every action representing the probability to choose the action in the state
  * according to the network policy.
  * The values are ordered lexicographically by action-names in a "," separated list form.
*/
/*
void ASNetSamplingSearch::network_probs_into_stream(
            const GlobalState &state, const OperatorsProxy &ops, ostringstream &network_probs_stream) const {
    // create evaluation context to be able to call compute_result and extract the policy result
    EvaluationContext eval_context = EvaluationContext(state, nullptr, true);
    EvaluationResult policy_result = network_policy->compute_result(eval_context);
    // extract policy result with preferred operators and their probabilities/ preferences
    vector<OperatorID> preferred_operator_ids = policy_result.get_preferred_operators();
    vector<float> preferred_operator_preferences = policy_result.get_operator_preferences();

    unsigned long number_of_operators = operator_indeces_sorted.size();
    vector<float> network_prob_values;
    network_prob_values.resize(number_of_operators);

    for (unsigned int op_index = 0; op_index < number_of_operators; op_index++) {
        int unsorted_index = operator_indeces_sorted[op_index];
        OperatorID op_id = ops[unsorted_index].get_global_operator_id();

        // check whether op_id is among the ids of preferred operators
        auto pos = std::find(preferred_operator_ids.begin(), preferred_operator_ids.end(), op_id);
        if (pos != preferred_operator_ids.end()) {
            // op is among the preferred -> use pos to get index
            long index = std::distance(preferred_operator_ids.begin(), pos);
            // use index to extract corresponding probability
            network_prob_values[op_index] = preferred_operator_preferences[index];
        } else {
            // op is not among the preferred operators -> not considered
            network_prob_values[op_index] = 0.0;
        }
    }
    vector_into_stream<float>(network_prob_values, network_probs_stream);
}
*/

/*
  * function extracts action_opt_values:
  * These are binary value for every action indicating whether the action starts a found plan according to
  * the teacher-search.
  * The values are ordered lexicographically by action-names in a "," separated list form.
*/
void ASNetSamplingSearch::action_opt_values_into_stream(
    const GlobalState &state, vector<int> applicable_values,
    StateRegistry &sr, const OperatorsProxy &ops,
    ostringstream &action_opts_stream) {
    unsigned long number_of_operators = operator_indeces_sorted.size();
    vector<float> action_opt_values;
    action_opt_values.resize(number_of_operators);

    // set state as new initial state for new searches
    set_modified_task_with_new_initial_state(state.get_id(), sr);

    // get modified teacher search using state as the initial state
    shared_ptr<SearchEngine> teacher_search_from_state = get_new_teacher_search_with_modified_task();
    // search and compute plan-cost if solution was found
    bool no_solution = false;
    try {
        teacher_search_from_state->search();
    } catch (const char* msg) {
	// task was deemed unsolvable/ unsolved -> set all values for which this is not the case to 1
	cout << "EXCEPTION CAUGHT" << endl;
	no_solution = true;
    }
    int plan_cost_from_state = -1;
    if (!no_solution && teacher_search_from_state->found_solution()) {
        plan_cost_from_state = 0;
        // sum up costs on plan
        for (OperatorID op_id : teacher_search_from_state->get_plan()) {
            plan_cost_from_state += ops[op_id].get_cost();
        }
    }

    for (unsigned int op_index = 0; op_index < number_of_operators; op_index++) {
        if (applicable_values[op_index] == 0) {
            // action is not applicable -> action_opt_value = 0
            action_opt_values[op_index] = 0;
        } else {
            // action is applicable -> check whether it starts an optimal plan according to teacher-search
            
            // get operator to sorted op_index
            int unsorted_index = operator_indeces_sorted[op_index];
            OperatorProxy op = ops[unsorted_index];

            // get state reached by action
            GlobalState succ_state = sr.get_successor_state(state, op);
            // get modified teacher search using succ_state as the initial state
            set_modified_task_with_new_initial_state(succ_state.get_id(), sr);
            shared_ptr<SearchEngine> teacher_search_from_succ_state = get_new_teacher_search_with_modified_task();
            // search and compute plan-cost if solution was found
            bool subtask_no_solution = false;
            try {
                    teacher_search_from_succ_state->search();
            } catch (const char* msg) {
            // subtask was deemed unsolvable/ unsolved -> set opt value to 0
            cout << "EXCEPTION CAUGHT" << endl;
            subtask_no_solution = true;
            }

            if (subtask_no_solution || !teacher_search_from_succ_state->found_solution()) {
                // applying action and search does not find a solution -> not good path
                action_opt_values[op_index] = 0;
                continue;
            } else {
                // solution found from succ_state
                if (!no_solution && teacher_search_from_state->found_solution()) {
                    // both search found a solution -> compute the cost from succ_state + op.cost() and compare
                    int plan_cost_from_succ_state = op.get_cost();
                    for (OperatorID op_id : teacher_search_from_succ_state->get_plan()) {
                        plan_cost_from_succ_state += ops[op_id].get_cost();
                    }
                    if (plan_cost_from_succ_state <= plan_cost_from_state) {
                        // operator is starting optimal or better plan (after teacher search)
                        action_opt_values[op_index] = 1;
                        continue;
                    } else {
                        // plan starting with operator is worse -> not good path
                        action_opt_values[op_index] = 0;
                        continue;
                    }
                } else {
                    // solution found from succ_state, but not from state
                    // -> from succ_state was better -> 1
                    action_opt_values[op_index] = 1;
                    continue;
                }
            }
        }
    }
    vector_into_stream<float>(action_opt_values, action_opts_stream);
}


/*
  * function extracts non-binary landmark features for additonal input features
  * These include two int values for each operator with the first indicating in
  * how many disjunctive action landmarks the action is included and the second
  * indicating in how many disjunctive action landmarks the action is the only
  * included. The disj. action LMs are computed with the LM-cut heuristic
  * The values are ordered lexicographically by action-names in a "," separated list form.
*/
void ASNetSamplingSearch::landmark_values_input_stream(
    const GlobalState &global_state, const TaskProxy &tp,
    std::ostringstream &add_input_features_stream) const {
            
    vector< vector<int> > cuts_ids;
    // compute LM-cut landmarks and collect cut op ids
    State state(*modified_task, global_state.get_values());
    // state = tp.convert_ancestor_state(state);
    landmark_generator->compute_landmarks(
        state,
        nullptr,
        [&cuts_ids](vector<int> cut, int /*cut_cost*/) {cuts_ids.push_back(cut); });

    unsigned long number_of_operators = operator_indeces_sorted.size();
    vector<int> additional_landmark_features(number_of_operators * 2, 0);

    OperatorsProxy operators = tp.get_operators();

    for (vector<int> cut : cuts_ids) {
        cout << "Cut:" << endl;
        bool single_element = false;
        if (cut.size() == 1) {
            single_element = true;
        }
        for (int unsorted_op_index : cut) {
	    OperatorProxy op = operators[unsorted_op_index];
	    cout << op.get_name() << endl;
            int sorted_index = operator_indeces_sorted_reversed[unsorted_op_index];
            // op was contained in contain -> increment counter
            additional_landmark_features[2 * sorted_index] += 1;
            if (single_element) {
                // op was only action in the cut -> increment 2nd counter
                additional_landmark_features[2 * sorted_index + 1] += 1;
            }
        }
    }

    vector_into_stream<int>(additional_landmark_features, add_input_features_stream);
}


/*
  * function extracts non-binary landmark features for additonal input features
  * These include three binary values for each operator with
  *  - first value indicating whether the action is the sole action in at least
  *    one landmark
  *  - second value indicating whether the action is included in at least one
  *    landmark of two or more actions
  *  - third value indicating whether the action appears in no landmark at all
  * how many disjunctive action landmarks the action is included and the second
  * indicating in how many disjunctive action landmarks the action is the only
  * included. The disj. action LMs are computed with the LM-cut heuristic
  * The values are ordered lexicographically by action-names in a "," separated list form.
*/
void ASNetSamplingSearch::binary_landmark_values_input_stream(
    const GlobalState &global_state, //const TaskProxy &tp,
    std::ostringstream &add_input_features_stream) const {
            
    vector< vector<int> > cuts_ids;
    // compute LM-cut landmarks and collect cut op ids
    State state(*modified_task, global_state.get_values());
    // state = tp.convert_ancestor_state(state);

    landmark_generator->compute_landmarks(
        state,
        nullptr,
        [&cuts_ids](vector<int> cut, int /*cut_cost*/) {cuts_ids.push_back(cut); });

    unsigned long number_of_operators = operator_indeces_sorted.size();
    vector<int> additional_landmark_features(number_of_operators * 3, 0);
    for (unsigned int op_id = 0; op_id < number_of_operators; op_id++) {
        /*
         * initialize third value of each action with 1 (at first it appeared
         * in no LM yet)
         */
        additional_landmark_features[op_id * 3 + 2] = 1;
    }

    for (vector<int> cut : cuts_ids) {
        bool single_element = false;
        if (cut.size() == 1) {
            single_element = true;
        }
        bool two_or_more_elements = false;
        if (cut.size() >= 2) {
            two_or_more_elements = true;
        }
        for (int unsorted_op_index : cut) {
            int sorted_index = operator_indeces_sorted_reversed[unsorted_op_index];
            if (single_element) {
                // op was only action in the cut -> set first binary value
                additional_landmark_features[3 * sorted_index] = 1;
            }
            if (two_or_more_elements) {
                // op included in a cut with at least 2 actions
                additional_landmark_features[3 * sorted_index + 1] = 1;
            }
            // op was contained in a cut
            additional_landmark_features[3 * sorted_index + 2] = 0;
        }
    }

    vector_into_stream<int>(additional_landmark_features, add_input_features_stream);
}


/*
  * Extracts the state_representation and puts it into stream for the sample
  * 
  * Format to represent a state:
  * <HASH>; <FACT_GOAL_VALUES>; <FACT_VALUES>; <ACTION_APPLICABLE_VALUES>; <ACTION_OPT_VALUES>(; <ADDITIONAL_FEATURES>)
  * 
  * using ";" as a separator
  * 
  * fields explained:
  * - <HASH>: hash-value indicating the problem instance
  * - <FACT_GOAL_VALUES>: binary value for every fact indicating whether the fact is part of the goal. Values
  *                       are ordered lexicographically by fact-names in a "," separated list form
  * - <FACT_VALUES>: binary values (0,1) indicating whether a fact is true. Values are ordered lexicographically
  *                  by fact-names and are all "," separated in a list and are given for every fact (e.g. [0,1,1,0]) 
  * - <ACTION_APPLICABLE_VALUES>: binary values indicating whether an action is applicable in the current state.
  *                               Ordering again is lexicographically by action-names and the values are in a ","
  *                               separated list form for all actions.
  * - <ACTION_OPT_VALUES>: binary value for each action indicating whether the action starts a found plan according to
  *                        the teacher-search. Again ordered lexicographically by action-names in a "," separated list.
  * - <ADDITIONAL_FEATURES>: additional input features given for each action in lexicographical order. The supported
  *                          features are listed in supported_additional_input_features
*/
void ASNetSamplingSearch::extract_sample_entries_trajectory(
    const Trajectory &trajectory, const TaskProxy &tp,
    StateRegistry &sr, const OperatorsProxy &ops, ostream &stream) {
    
    // extract fact_goal_values into goal_stream
    ostringstream goal_stream;
    goal_into_stream(goal_stream);

    for (StateID state_id : trajectory) {
	cout << "Extracting sample stream for state " << state_id << endl;
        GlobalState state = sr.lookup_state(state_id);

        // extract fact_values into state_stream
        ostringstream state_stream;
        state_into_stream(state, state_stream);

        // extract action_applicable_values into applicable_stream
        ostringstream applicable_stream;
        vector<int> applicable_values = applicable_values_into_stream(state, ops, applicable_stream);

        // extract action_network_probs into network_probs_stream
        // ostringstream network_probs_stream;
        // network_probs_into_stream(state, ops, network_probs_stream);

        // extract action_opt_values into action_opts_stream
        ostringstream action_opts_stream;
        action_opt_values_into_stream(state, applicable_values, sr, ops, action_opts_stream);

        ostringstream additional_input_features_stream;
        if (additional_input_features != "none") {
            // extract additional input features
            if (additional_input_features == "landmarks") {
                landmark_values_input_stream(state, tp, additional_input_features_stream);
            } else if (additional_input_features == "binary_landmarks") {
                binary_landmark_values_input_stream(state, /*tp,*/ additional_input_features_stream);
            }
        }

        stream << problem_hash << ";" << goal_stream.str() << ";"
               << state_stream.str() << ";" << applicable_stream.str() << ";"
               /*<< network_probs_stream.str() << ";" */ << action_opts_stream.str();
       if (additional_input_features != "none") {
           stream << ";" << additional_input_features_stream.str();
       }

        stream << "~";
    }
}


/* 
  * After applying the network_search->search(), this function is used to extract
  * all the information of the trajectories to collect all state sampling information
  * and return it as a string
  * 
  * For detailed information on the format used to represent a state look above at the comment
*/
std::string ASNetSamplingSearch::extract_exploration_sample_entries() {
    StateRegistry &sr = network_search->get_non_const_state_registry();
    const SearchSpace &ss = network_search->get_search_space();
    const TaskProxy &tp = network_search->get_task_proxy();
    OperatorsProxy ops = tp.get_operators();
    const GoalsProxy gps = tp.get_goals();

    ostringstream new_entries;
    new_entries << "# Network-Search Samples~";

    if (network_search->found_solution()) {
        new_entries << "GOAL_EXPLORATION~";
        const GlobalState goal_state = network_search->get_goal_state();
        Plan plan = network_search->get_plan();
        Trajectory trajectory;
        ss.trace_path(goal_state, trajectory);
        // add all StateIDs from trajectory to list of explored states
        network_explored_states.insert(network_explored_states.end(), trajectory.begin(), trajectory.end());

        extract_sample_entries_trajectory(trajectory, tp, sr, ops, new_entries);
    } else {
        // no solution found -> termination due to timeout, dead-end or trajectory limit reached
        new_entries << "NO_GOAL_EXPLORATION~";
        const GlobalState last_state = network_search->get_last_state();
        Plan plan;
        Trajectory trajectory;
        ss.trace_path(last_state, plan, trajectory);
        // add all StateIDs from trajectory to list of explored states
        network_explored_states.insert(network_explored_states.end(), trajectory.begin(), trajectory.end());

	cout << network_explored_states.size() << " states extracted by network search" << endl;
        extract_sample_entries_trajectory(trajectory, tp, sr, ops, new_entries);
    }

    string post = new_entries.str();
    replace(post.begin(), post.end(), '\n', '\t');
    replace(post.begin(), post.end(), '~', '\n');

    return post;
}

/* 
  * After applying the teacher_search->search(), this function is used to extract
  * all the information of the trajectories to collect all state sampling information
  * and return it as a string
  * 
  * For detailed information on the format used to represent a state look above at the comment
*/
std::string ASNetSamplingSearch::extract_teacher_sample_entries() {
    StateRegistry &sr = teacher_search->get_non_const_state_registry();
    const SearchSpace &ss = teacher_search->get_search_space();
    const TaskProxy &tp = teacher_search->get_task_proxy();
    OperatorsProxy ops = tp.get_operators();
    const GoalsProxy gps = tp.get_goals();

    ostringstream new_entries;
    new_entries << "# Teacher-Search Sampled States~";

    if (teacher_search->found_solution()) {
        const GlobalState goal_state = teacher_search->get_goal_state();
        Plan plan = teacher_search->get_plan();
        Trajectory trajectory;
        cout << "Teacher search start extracting trajectory from goal" << endl;
        ss.trace_path(goal_state, trajectory);

        extract_sample_entries_trajectory(trajectory, tp, sr, ops, new_entries);
    } else if (use_non_goal_teacher_paths) {
        // no solution found -> termination due to timeout, dead-end or trajectory limit reached
        const GlobalState last_state = teacher_search->get_last_state();
        Plan plan;
        Trajectory trajectory;
        ss.trace_path(last_state, plan, trajectory);

        extract_sample_entries_trajectory(trajectory, tp, sr, ops, new_entries);
    }

    string post = new_entries.str();
    replace(post.begin(), post.end(), '\n', '\t');
    replace(post.begin(), post.end(), '~', '\n');

    return post;
}

void ASNetSamplingSearch::set_modified_task_with_new_initial_state(StateID state_id, const StateRegistry &sr) {
    // extract state values as new initial state
    GlobalState init_state = sr.lookup_state(state_id);
    vector<int> init_state_values;
    init_state_values.reserve(init_state.get_values().size());
    for (int val : init_state.get_values()) {
        init_state_values.push_back(val);
    }

    // extract goal facts
    vector<FactPair> goal_facts;
    GoalsProxy goals_proxy = task_proxy.get_goals();
    goal_facts.reserve(goals_proxy.size());
    for (unsigned int i = 0; i < goals_proxy.size(); i++) {
        goal_facts.push_back(goals_proxy[i].get_pair());
    }
    modified_task = make_shared<extra_tasks::ModifiedInitGoalsTask>(task,
        std::move(init_state_values),
        std::move(goal_facts));

}

shared_ptr<SearchEngine> ASNetSamplingSearch::get_new_teacher_search_with_modified_task() const {
    OptionParser engine_parser(search_parse_tree, false);
    return engine_parser.start_parsing<shared_ptr<SearchEngine>>();
}

void ASNetSamplingSearch::initialize() {
    cout << "Initializing ASNet Sampling Manager..." << endl;
    add_header_samples(samples);
    // set teacher_search
    OptionParser engine_parser(search_parse_tree, false);
    teacher_search = engine_parser.start_parsing<shared_ptr<SearchEngine>>();
    // set goal values
    unsigned long number_of_facts = facts_sorted.size();
    fact_goal_values.resize(number_of_facts);
    for (unsigned int fact_index = 0; fact_index < number_of_facts; fact_index++) {
        if (std::find(g_goal.begin(), g_goal.end(), facts_sorted[fact_index]) != g_goal.end()) {
            // fact_index th fact is a goal
            fact_goal_values[fact_index] = 1;
        } else {
            // fact_index th fact is not a goal
            fact_goal_values[fact_index] = 0;
        }
    }
    cout << "done." << endl;
}

SearchStatus ASNetSamplingSearch::step() {
    cout << "Sampling Search calling network search" << endl;
    network_search->search();
    samples << extract_exploration_sample_entries();
    save_plan_intermediate();

    if (use_teacher_search) {
        for (StateID & state_id : network_explored_states) {
            // explore states with teacher policy from state_id onwards
            set_modified_task_with_new_initial_state(state_id, network_search->get_state_registry());
            teacher_search = get_new_teacher_search_with_modified_task();
	    try {
                teacher_search->search();
	    } catch (const char* msg) {
		// task was deemed unsolvable/ unsolved -> skip this state
		cout << "EXCEPTION CAUGHT" << endl;
		continue;
	    }

            switch (teacher_search->get_status()) {
                case FAILED:
                    // if search reached dead-end -> don't sample states
                    break;
                case TIMEOUT:
                    if (!use_non_goal_teacher_paths) {
                        /* if search went into timeout, only sample if non-goal paths
                        should be used for sampling */
                        break;
                    }
                    [[fallthrough]];
                default:
                    // teacher search found goal
                    samples << extract_teacher_sample_entries();
                    save_plan_intermediate();
            }
        }
    }

    return SOLVED;
}

void ASNetSamplingSearch::print_statistics() const {
    int new_samples = 0;

    if (!(samples.str() == "")) {
        string s = samples.str();
        for (unsigned int i = 0; i <= s.size(); i++) {
            if (s[i] == '\n') {
                new_samples++;
            }
        }
    }
    cout << "Generated Samples: " << (generated_samples + new_samples) << endl;
}

void ASNetSamplingSearch::add_header_samples(ostream &stream) const {
    stream << "# Everything in a line after '#' is a comment" << endl;
    stream << "# (NO_)GOAL_EXPLORATION indicate whether the executed network search exploration reached a goal state (or not)" << endl;
    stream << "# Entry format:<HASH>; <FACT_GOAL_VALUES>; <FACT_VALUES>; <ACTION_APPLICABLE_VALUES>; <ACTION_OPT_VALUES>(; <ADDITIONAL_FEATURES>)" << endl;
    stream << "# <HASH> := hash value to identify where the sample comes from" << endl;
    stream << "# <FACT_GOAL_VALUES> := binary value for every fact indicating whether the fact is part of the goal. Values are ordered lexicographically by fact-names in a \",\" separated list form." << endl;
    stream << "# <FACT_VALUES> := binary values (0,1) indicating whether a fact is true. Values are ordered lexicographically by fact-names and are all \",\" separated in a list and are given for every fact (e.g. [0,1,1,0])." << endl;
    stream << "# <ACTION_APPLICABLE_VALUES> := binary values indicating whether an action is applicable in the current state. Ordering again is lexicographically by action-names and the values are in a \",\" separated list form for all actions." << endl;
    stream << "# <ACTION_OPT_VALUES> := binary value for each action indicating whether the action starts a found plan according to the teacher-search. Again ordered lexicographically by action-names in a \",\" separated list." << endl;
    stream << "# <ADDITIONAL_FEATURES>:= additional input features given for each action in lexicographical order. The supported features are listed in supported_additional_input_features." << endl;
}

void ASNetSamplingSearch::save_plan_intermediate() {
    if (!(samples.str() == "")) {
        string s = samples.str();

        ofstream outfile(target_location, ios::app);
        outfile << s;
        //number of entries is a bit more complicated, because we filter
        //comments out.
        bool comment = false;
        int line_length = 0;
        for (unsigned int i = 0; i <= s.size(); i++) {
            if (s[i] == '\n') {
                if (line_length > 0)
                    generated_samples++;
                comment = false;
                line_length = 0;
            } else if (s[i] == '#') {
                comment = true;
            } else if (!comment) {
                line_length++;
            }
        }
        samples.str("");
        samples.clear();
    }
}

void ASNetSamplingSearch::add_sampling_options(OptionParser &parser) {
    SearchEngine::add_options_to_parser(parser);
    parser.add_option<shared_ptr<SearchEngine>> ("search",
        "Search engine to use as teacher-search guidance");
    parser.add_option<std::string> ("target",
        "Place to save the sampled data (currently only appending files "
        "is supported", "sample.data");
    parser.add_option<std::string> ("hash",
        "MD5 hash of the input problem. This can be used to "
        "differentiate which problems created which entries.", "None");
    parser.add_option<bool> ("use_non_goal_teacher_paths",
        "Bool value indicating whether paths/ trajectories of the teacher search "
        "not reaching a goal state should be sampled", "true");
    parser.add_option<bool> ("use_teacher_search",
        "Bool value indicating whether the teacher search should be used for sampling. "
        "If false: only the network search exploration is used for sampling BUT teacher "
        "search in general is still needed for opt values", "true");
    parser.add_option<std::string> ("additional_input_features",
        "Name of additional input features to be used for the network. These "
        "will be extracted and added to the samples", "none");
    parser.add_option<shared_ptr<SearchEngine>>("network_search",
        "Policy search using the network policy", "policysearch(p=np(network=asnet(path=asnet.pb)))");
}

static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
        ASNetSamplingSearch::add_sampling_options(parser);
        parser.document_synopsis("ASNet sampling search", "");
        Options opts = parser.parse();

        if (parser.dry_run())
            return nullptr;
        else
            return make_shared<ASNetSamplingSearch>(opts);
}

static PluginShared<SearchEngine> _plugin("asnet_sampling_search", _parse);

/*
 * Use modified_task as transformed task in search.
 * Used to manipulate the initial state to run teacher-search from specific sampled
 * states onward.
 */
std::shared_ptr<AbstractTask> modified_task = g_root_task();

static shared_ptr<AbstractTask> _parse_sampling_transform(
    OptionParser &parser) {
    if (parser.dry_run()) {
        return nullptr;
    } else {

        return modified_task;
    }
}

static PluginShared<AbstractTask> _plugin_sampling_transform(
    "asnet_sampling_transform", _parse_sampling_transform);
}
