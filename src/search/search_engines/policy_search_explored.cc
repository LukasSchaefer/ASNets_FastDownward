#include "policy_search_explored.h"

#include "../policy.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../task_utils/regression_utils.h"
#include "../task_utils/successor_generator.h"

using namespace std;
using utils::ExitCode;

namespace policy_search_explored {
    using Plan = std::vector<OperatorID>;

    PolicySearchExplored::PolicySearchExplored(
    const Options &opts)
    : SearchEngine(opts),
      policy(opts.get<Policy *>("p")),
      use_heuristic_dead_end_detection(opts.get<bool>("dead_end_detection")),
      exploration_trajectory_limit(opts.get<int>("trajectory_limit")),
      current_eval_context(state_registry.get_initial_state(), &statistics, true, true),
      explored_states(vector<StateID>()) {
        if (use_heuristic_dead_end_detection) {
            dead_end_heuristic = opts.get<Heuristic *>("dead_end_detection_heuristic");
            // only use this dead-end detection if it is reliable on the task
            if (!dead_end_heuristic->dead_ends_are_reliable()) {
                use_heuristic_dead_end_detection = false;
            }
        }
	if (use_heuristic_dead_end_detection) {
	    cout << "policy search uses heuristic dead-end detection" << endl;
	}
    }

    PolicySearchExplored::~PolicySearchExplored() {
    }

    void PolicySearchExplored::set_current_eval_context(StateID state_id) {
        GlobalState state = state_registry.lookup_state(state_id);
        current_eval_context = EvaluationContext(state, &statistics, true, true);
    }

    void PolicySearchExplored::initialize() {
        assert(policy);
        cout << "Conducting policy search with explored list" << endl;

        bool dead_end = false;
        if (use_heuristic_dead_end_detection) {
            dead_end = current_eval_context.is_heuristic_infinite(dead_end_heuristic);
        } else {
            dead_end = current_eval_context.is_policy_dead_end(policy);
        }
        statistics.inc_evaluated_states();

        if (dead_end) {
            cout << "Initial state is a dead end, no solution" << endl;
            if (use_heuristic_dead_end_detection) {
                if (dead_end_heuristic->dead_ends_are_reliable()) {
                    utils::exit_with(ExitCode::UNSOLVABLE);
                } else{
                    utils::exit_with(ExitCode::UNSOLVED_INCOMPLETE);
                }
            } else {
                if (policy->dead_ends_are_reliable())
                    utils::exit_with(ExitCode::UNSOLVABLE);
                else
                    utils::exit_with(ExitCode::UNSOLVED_INCOMPLETE);
            }
        }

        SearchNode node = search_space.get_node(current_eval_context.get_state());
        node.open_initial();
    }

    SearchStatus PolicySearchExplored::step() {
        // cout << "PolicySearch explored list call" << endl;
        if (check_goal_and_set_plan(current_eval_context.get_state())) {
	    // cout << "Goal found!" << endl;
            return SOLVED;
        }

	// cout << "No Goal found" << endl;

        if (exploration_trajectory_limit != -1 && trajectory_length >= exploration_trajectory_limit) {
            cout << "No solution - trajectory limit reached" << endl;
            return TRAJECTORY_LIMIT_REACHED;
        }

        assert(current_eval_context.contains_policy());
	// cout << "Contains policy was checked" << endl;

        // add current state to explored list
        explored_states.push_back(current_eval_context.get_state().get_id());

        // collect current state and search node
        GlobalState parent_state = current_eval_context.get_state();
        SearchNode parent_node = search_space.get_node(parent_state);

        // collect policy output in current EvaluationContext
	// cout << "Policy output is extracted" << endl;
        vector<OperatorID> operator_ids = current_eval_context.get_preferred_operators(policy);
        vector<float> operator_prefs = current_eval_context.get_preferred_operator_preferences(policy);

        // preferences correspond to operator id by index
	// cout << "Policy output is checked" << endl;
        assert(operator_ids.size() == operator_prefs.size());

        // find most probable/ preferenced operator
        StateID best_new_state_id = StateID::no_state;
        unsigned int best_op_index = 0;
        float highest_op_probability = 0;
	// cout << "Determining best op" << endl;
        for (unsigned int index = 0; index < operator_ids.size(); index++) {
            float probability = operator_prefs[index];
            if (probability > highest_op_probability) {
                // check if it leads to an already explored state
                OperatorID op_id = operator_ids[index];
                OperatorProxy op_proxy  = task_proxy.get_operators()[op_id];
                GlobalState new_state = state_registry.get_successor_state(parent_state, op_proxy);
		cout << "Check whether op leads to already explored state" << endl;
                if (find(explored_states.begin(), explored_states.end(), new_state.get_id()) != explored_states.end()) {
		    // cout << "Action " << op_proxy.get_name() << " lead to state " << new_state.get_id() << " already encountered -> ignored" << endl;
                    // new_state was already explored
                    continue;
                }

                highest_op_probability = probability;
                best_op_index = index;
                best_new_state_id = new_state.get_id();
		cout << "New best op found" << endl;
            }
        }

        // reach new state
	// cout << "Best op determined" << endl;
        if (best_new_state_id == StateID::no_state) {
            // no applicable action lead to a new state -> dead-end
	    // cout << "no op was applicable or lead to new state" << endl;
            return FAILED;
        }
	// cout << "Getting op_id" << endl;
        OperatorID op_id = operator_ids[best_op_index];
	// cout << "Getting op" << endl;
        OperatorProxy op_proxy = task_proxy.get_operators()[op_id];
	// cout << "Getting state" << endl;
        GlobalState new_state = state_registry.lookup_state(best_new_state_id);
        cout << "Policy reached state with id " << new_state.get_id() << " by applying action " << op_proxy.get_name() << " which had probability " << highest_op_probability << endl;
        SearchNode node = search_space.get_node(new_state);
        statistics.inc_generated();

	// cout << "checking for new state" << endl;
        if (node.is_new()) {
            // create eval_context with policy
            EvaluationContext eval_context(new_state, &statistics, true, true);
            statistics.inc_evaluated_states();

            bool dead_end = false;
	    // cout << "check whether state is dead-end..." << endl;
            if (use_heuristic_dead_end_detection) {
                // cout << "using heuristic" << endl;
                dead_end = eval_context.is_heuristic_infinite(dead_end_heuristic);
            } else {
                // cout << "using policy" << endl;
                dead_end = eval_context.is_policy_dead_end(policy);
            }
            if (dead_end) {
                node.mark_as_dead_end();
                statistics.inc_dead_ends();
                cout << "No solution - FAILED" << endl;
                return FAILED;
            }

            node.open(parent_node, op_proxy);

            current_eval_context = eval_context;
            trajectory_length++;
            return IN_PROGRESS;
        }

        cout << "No solution - FAILED" << endl;
        return FAILED;
    }

    static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
        parser.document_synopsis("Policy search with explored states memory", "");
        parser.add_option<Policy *>("p", "policy");
        parser.add_option<bool>("dead_end_detection",
        "Boolean value indicating whether early dead-end detection using "
        "a heuristic function should be used during search", "true");
        parser.add_option<Heuristic *>("dead_end_detection_heuristic",
        "heuristic used for early dead-end detection", "ff");
        parser.add_option<int> ("trajectory_limit",
        "Int to represent the length limit for explored trajectories during "
        "network policy exploration", "-1");
        SearchEngine::add_options_to_parser(parser);
        Options opts = parser.parse();

        if (parser.dry_run())
            return nullptr;
        else
            return make_shared<PolicySearchExplored>(opts);
    }

    static PluginShared<SearchEngine> _plugin("policysearch_explored", _parse);
}
