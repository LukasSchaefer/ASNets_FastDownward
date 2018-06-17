#include "asnet_sampling_search.h"

#include "../evaluation_context.h"
#include "../globals.h"
#include "../open_list_factory.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../tasks/modified_init_goals_task.h"
#include "../task_utils/successor_generator.h"
#include "../task_utils/sampling.h"
#include "../utils/rng.h"
#include "../utils/rng_options.h"

#include "../task_utils/sampling.h"
#include "../utils/rng.h"

#include "../task_utils/lexicographical_access.h"
#include "../task_utils/regression_utils.h"
#include "../task_utils/predecessor_generator.h"

#include "policy_search.h"
#include "../policies/network_policy.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <stdlib.h>

using namespace std;

namespace asnet_sampling_search {

const string SAMPLE_FILE_MAGIC_WORD = "# MAGIC FIRST LINE ASNET SAMPLING";

template <typename t> void vector_into_stream(vector<t> vec, ostringstream oss) {
    oss << "[";
    std::move(vec.begin(), vec.end()-1, std::ostream_iterator<t>(oss, ","));
    oss << vec.back() << "]";
}

ASNetSamplingSearch::ASNetSamplingSearch(const Options &opts)
: SearchEngine(opts),
problem_hash(opts.get<string>("hash")),
target_location(opts.get<string>("target")),
teacher_policy(opts.get<Policy *>("teacher_policy")),
exploration_trajectory_limit(opts.get<int>("trajectory_limit")),
facts_sorted(lexicographical_access::get_facts_lexicographically(task_proxy)),
operator_indeces_sorted(lexicographical_access::get_operator_indeces_lexicographically(task_proxy)) {
    network_policy = NetworkPolicy(opts);
    opts.set<Policy *>("policy", asnet_policy);
    network_search = PolicySearch(opts);

    // no trajectory-limit for teacher search/ policy
    opts.set<int>("trajectory_limit", -1);
    opts.set<Policy *>("policy", teacher_policy);
    teacher_search = PolicySearch(opts);
}

std::string ASNetSamplingSearch::extract_modification_hash(
    State init, GoalsProxy goals) const {
    ostringstream oss;
    init.dump_pddl(oss);
    goals.dump_pddl(oss);
    std::string merged = oss.str();
    return to_string(ASNetSamplingSearch::shash(merged));
}

/*
  * function extracts fact_goal_values.
  * These are binary value for every fact indicating whether the fact is part of the goal. Values
  * are ordered lexicographically by fact-names in a "," separated list form.
*/
void ASNetSamplingSearch::goal_into_stream(ostringstream goal_stream) const {
    unsigned int number_of_facts = facts_sorted.size();
    vector<int> fact_goal_values;
    fact_goal_values.resize(number_of_facts);
    for (int fact_index = 0; fact_index < number_of_facts; fact_index++) {
        if (std::find(g_goal.begin(), g_goal.end(), facts_sorted[fact_index]) != g_goal.end()) {
            // fact_index th fact is a goal
            fact_goal_values[fact_index] = 1;
        } else {
            // fact_index th fact is not a goal
            fact_goal_values[fact_index] = 0;
        }
    }
    vector_into_stream<int>(fact_goal_values, goal_stream);
}

/*
  * function extracts fact_values.
  * These are binary value for every fact indicating whether the fact is currently true in the state.
  * The values are ordered lexicographically by fact-names in a "," separated list form.
*/
void ASNetSamplingSearch::state_into_stream(GlobalState &state, ostringstream state_stream) const {
    unsigned int number_of_facts = facts_sorted.size();
    vector<int> fact_values;
    fact_values.resize(number_of_facts);
    for (int fact_index = 0; fact_index < number_of_facts; fact_index++) {
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
void ASNetSamplingSearch::applicable_values_into_stream(GlobalState &state, ostringstream applicable_stream) const {
    // collect all applicable actions in state
    vector<OperatorID> applicable_op_ids;
    g_successor_generator->generate_applicable_ops(state, applicable_op_ids);

    // get all OperatorProxys
    OperatorsProxy op_proxys = task_proxy.get_operators();

    unsigned int number_of_operators = operator_indeces_sorted.size();
    vector<int> applicable_values;
    applicable_values.resize(number_of_operators);

    for (int op_index = 0; op_index < number_of_operators; op_index++) {
        int unsorted_index = operator_indeces_sorted[op_index];
        OperatorID op_id = op_proxys[unsorted_index].get_global_operator_id();
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
}

/*
  * function extracts action_network_probs:
  * These are float value for every action representing the probability to choose the action in the state
  * according to the network policy.
  * The values are ordered lexicographically by action-names in a "," separated list form.
*/
void ASNetSamplingSearch::network_probs_into_stream(GlobalState &state, ostringstream network_probs_stream) const {
    // create evaluation context to be able to call compute_result and extract the policy result
    EvaluationContext eval_context(state);
    EvaluationResult policy_result = network_policy->compute_result()
    // extract policy result with preferred operators and their probabilities/ preferences
    vector<OperatorID> preferred_operator_ids = policy_result.get_preferred_operators();
    vector<float> preferred_operator_preferences = policy_result.get_operator_preferences();

    // get all OperatorProxys
    OperatorsProxy op_proxys = task_proxy.get_operators();

    unsigned int number_of_operators = operator_indeces_sorted.size();
    vector<float> network_prob_values;
    network_prob_values.resize(number_of_operators);

    for (int op_index = 0; op_index < number_of_operators; op_index++) {
        int unsorted_index = operator_indeces_sorted[op_index];
        OperatorID op_id = op_proxys[unsorted_index].get_global_operator_id();

        // check whether op_id is among the ids of preferred operators
        auto pos = std::find(preferred_operator_ids.begin(), preferred_operator_ids.end(), op_id);
        if (pos != preferred_operator_ids.end()) {
            // op is among the preferred -> use pos to get index
            int index = std::distance(preferred_operator_ids.begin(), pos);
            // use index to extract corresponding probability
            network_prob_values[op_index] = preferred_operator_preferences[index];
        } else {
            // op is not among the preferred operators -> not considered
            network_prob_values[op_index] = 0.0;
        }
    }
    vector_into_stream<float>(network_prob_values, network_probs_stream);
}

/*
  * function extracts action_opt_values:
  * These are binary value for every action indicating whether the action starts a found plan according to
  * the teacher-search.
  * The values are ordered lexicographically by action-names in a "," separated list form.
*/
void ASNetSamplingSearch::action_opt_values_into_stream(GlobalState &state, ostringstream action_opts_stream) const {
    // TODO: check whether actions start the plan found for teacher-search from state going
}

/*
  * Extracts the state_representation and puts it into stream for the sample
  * 
  * Format to represent a state:
  * <HASH>; <FACT_GOAL_VALUES>; <fact_values>; <ACTION_APPLICABLE_VALUES>; <ACTION_NETWORK_PROBABILITIES>; <ACTION_OPT_VALUES>
  * 
  * using ";" as a separator
  * 
  * fields explained:
  * - <HASH>:     hash-value indicating the problem instance
  * - <fact_goal_values>: binary value for every fact indicating whether the fact is part of the goal. Values
  *                       are ordered lexicographically by fact-names in a "," separated list form
  * - <FACT_GOAL_VALUES>: binary values (0,1) indicating whether a fact is true. Values are ordered lexicographically
  *                       by fact-names and are all "," separated in a list and are given for every fact (e.g. [0,1,1,0]) 
  * - <ACTION_APPLICABLE_VALUES>: binary values indicating whether an action is applicable in the current state.
  *                               Ordering again is lexicographically by action-names and the values are in a ","
  *                               separated list form for all actions.
  * - <ACTION_NETWORK_PROBABILITIES>: float value for every action representing the probability to choose the action in the
  *                                   state according to the network policy. Values are again ordered lexicographically by
  *                                   action-names and values are "," separated, e.g. [0.0,0.1,0.3,0.6]
  * - <ACTION_OPT_VALUES>: binary value for each action indicating whether the action starts a found plan according to
  *                        the teacher-search. Again ordered lexicographically by action-names in a "," separated list.
*/
void ASNetSamplingSearch::extract_sample_entries_trajectory(
    const Plan &plan, const Trajectory &trajectory,
    const StateRegistry &sr, OperatorsProxy &ops,
    ostream &stream) const {
    
    // extract fact_goal_values into goal_stream
    ostringstream goal_stream;
    goal_into_stream(goal_stream);

    for (int state_index = 0; state_index < trajectory.size(); state_index++) {
        GlobalState state = sr.lookup_state(trajectory[state_index]);

        // extract fact_values into state_stream
        ostringstream state_stream;
        state_into_stream(&state, state_stream);

        // extract action_applicable_values into applicable_stream
        ostringstream applicable_stream;
        applicable_values_into_stream(&state, applicable_stream);

        // extract action_network_probs into network_probs_stream
        ostringstream network_probs_stream;
        network_probs_into_stream(&state, network_probs_stream);

        // extract action_opt_values into action_opts_stream
        ostringstream action_opts_stream;
        action_opt_values_into_stream(&state, action_opts_stream);

        stream << problem_hash << ";" << goal_stream.str() << ";"
               << state_stream.str() << ";" << applicable_stream.str() << ";"
               << network_probs_stream.str() << ";" << action_opts_stream.str();

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
    StateRegistry &sr = network_search->get_state_registry();
    const SearchSpace &ss = network_search->get_search_space();
    const TaskProxy &tp = network_search->get_task_proxy();
    OperatorsProxy ops = tp.get_operators();
    const GoalsProxy gps = tp.get_goals();

    ostringstream new_entries;

    if (network_search->found_solution()) {
        const GlobalState goal_state = network_search->get_goal_state();
        Plan plan = network_search->get_plan();
        Trajectory trajectory;
        ss.trace_path(goal_state, trajectory);
        // add all StateIDs from trajectory to list of explored states
        network_explored_states.insert(network_explored_states.end(), trajectory.begin(), trajectory.end());

        extract_sample_entries_trajectory(plan, trajectory, sr, ops, new_entries);
    } else {
        // no solution found -> termination due to timeout, dead-end or trajectory limit reached
        const StateID last_state = network_search->get_last_state_id();
        Plan plan = network_search->get_plan_to_last_state();
        Trajectory trajectory;
        ss.trace_path(last_state, trajectory);
        // add all StateIDs from trajectory to list of explored states
        network_explored_states.insert(network_explored_states.end(), trajectory.begin(), trajectory.end());

        extract_sample_entries_trajectory(plan, trajectory, sr, ops, new_entries);
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
    StateRegistry &sr = teacher_search->get_state_registry();
    const SearchSpace &ss = teacher_search->get_search_space();
    const TaskProxy &tp = teacher_search->get_task_proxy();
    OperatorsProxy ops = tp.get_operators();
    const GoalsProxy gps = tp.get_goals();

    ostringstream new_entries;

    if (teacher_search->found_solution()) {
        const GlobalState goal_state = teacher_search->get_goal_state();
        Plan plan = teacher_search->get_plan();
        Trajectory trajectory;
        ss.trace_path(goal_state, trajectory);

        extract_sample_entries_trajectory(plan, trajectory, sr, ops, new_entries);
    } else {
        // no solution found -> termination due to timeout, dead-end or trajectory limit reached
        const StateID last_state = teacher_search->get_last_state_id();
        Plan plan = teacher_search->get_plan_to_last_state();
        Trajectory trajectory;
        ss.trace_path(last_state, trajectory);

        extract_sample_entries_trajectory(plan, trajectory, sr, ops, new_entries);
    }

    string post = new_entries.str();
    replace(post.begin(), post.end(), '\n', '\t');
    replace(post.begin(), post.end(), '~', '\n');

    return post;
}

void ASNetSamplingSearch::initialize() {
    cout << "Initializing ASNet Sampling Manager...";
    add_header_samples(samples);

    cout << "done." << endl;
}

SearchStatus SamplingSearch::step() {
    network_search->search();
    samples << extract_exploration_sample_entries();
    save_plan_intermediate();

    for (StateID & state_id : network_explored_states) {
        // explore states with teacher policy from state_id onwards
        teacher_search.set_current_eval_context(state_id);
        teacher_search->search();
        samples << extract_teacher_sample_entries();
        save_plan_intermediate();
    }

    return SOLVED;
}

void ASNetSamplingSearch::print_statistics() const {
    int new_samples = 0;

    if (samples.str().compare("")) {
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
    stream << SAMPLE_FILE_MAGIC_WORD << endl;
    stream << "# Everything in a line after '#' is a comment" << endl;
    stream << "# Entry format:<HASH>; <FACT_GOAL_VALUES>; <FACT_VALUES>; <ACTION_APPLICABLE_VALUES>; <ACTION_NETWORK_PROBABILITIES>; <ACTION_OPT_VALUES>" << endl;
    stream << "# <HASH> := hash value to identify where the sample comes from" << endl;
    stream << "# <FACT_GOAL_VALUES> := binary value for every fact indicating whether the fact is part of the goal. Values are ordered lexicographically by fact-names in a \",\" separated list form." << endl;
    stream << "# <FACT_VALUES> := binary values (0,1) indicating whether a fact is true. Values are ordered lexicographically by fact-names and are all \",\" separated in a list and are given for every fact (e.g. [0,1,1,0])." << endl;
    stream << "# <ACTION_APPLICABLE_VALUES> := binary values indicating whether an action is applicable in the current state. Ordering again is lexicographically by action-names and the values are in a \",\" separated list form for all actions." << endl;
    stream << "# <ACTION_NETWORK_PROBABILITIES> := float value for every action representing the probability to choose the action in the state according to the network policy. Values are again ordered lexicographically by action-names and values are \",\" separated, e.g. [0.0,0.1,0.3,0.6]." << endl;
    stream << "# <ACTION_OPT_VALUES> := binary value for each action indicating whether the action starts a found plan according to the teacher-search. Again ordered lexicographically by action-names in a \",\" separated list." << endl;
}

void ASNetSamplingSearch::save_plan_intermediate() {
    if (samples.str().compare("")) {
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
    parser.add_option<std::string> ("target",
        "Place to save the sampled data (currently only appending files"
        "is supported", "None");
    parser.add_option<Policy *> ("teacher_policy",
        "Teacher policy which is used to sample (usually optimal) trajectories "
        "along states from trajectories explored by the network policy.");
    parser.add_option<std::string> ("hash",
        "MD5 hash of the input problem. This can be used to "
        "differentiate which problems created which entries.", "None");
    parser.add_option<int> ("trajectory_limit",
        "Int to represent the length limit for explored trajectories during",
        "network policy exploration", 300);
    parser.add_option<shared_ptr<neural_networks::AbstractNetwork>>("network",
        "Network to sample with (Built for ASNets)", "asnet");
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

/* Global variable for search algorithms to store arbitrary paths (plans [
   operator ids] and trajectories [state ids]).
   (the storing of the solution trajectory is independent of this variable).*/
Path::Path(StateID start) {

    trajectory.push_back(start);
}

Path::~Path() { }

void Path::add(OperatorID op, StateID next) {

    plan.push_back(op);
    trajectory.push_back(next);
}

const Path::Plan &Path::get_plan() const {

    return plan;
}

const Trajectory &Path::get_trajectory() const {
    return trajectory;
}


std::vector<Path> paths;
}