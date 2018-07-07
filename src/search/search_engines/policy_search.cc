#include "policy_search.h"

#include "../policy.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../task_utils/regression_utils.h"
#include "../task_utils/successor_generator.h"

using namespace std;
using utils::ExitCode;

namespace policy_search {
    using Plan = std::vector<OperatorID>;

    PolicySearch::PolicySearch(
    const Options &opts)
    : SearchEngine(opts),
      policy(opts.get<Policy *>("p")),
      use_heuristic_dead_end_detection(opts.get<bool>("dead_end_detection")),
      exploration_trajectory_limit(opts.get<int>("trajectory_limit")),
      current_eval_context(state_registry.get_initial_state(), &statistics, true, true) {
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

    PolicySearch::~PolicySearch() {
    }

    void PolicySearch::set_current_eval_context(StateID state_id) {
        GlobalState state = state_registry.lookup_state(state_id);
        current_eval_context = EvaluationContext(state, &statistics, true, true);
    }

    void PolicySearch::initialize() {
        assert(policy);
        cout << "Conducting policy search" << endl;

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

    SearchStatus PolicySearch::step() {
        if (check_goal_and_set_plan(current_eval_context.get_state())) {
            return SOLVED;
        }

        if (exploration_trajectory_limit != -1 && trajectory_length >= exploration_trajectory_limit) {
            cout << "No solution - trajectory limit reached" << endl;
            return TRAJECTORY_LIMIT_REACHED;
        }

        assert(current_eval_context.contains_policy());

        // collect current state and search node
        GlobalState parent_state = current_eval_context.get_state();
        SearchNode parent_node = search_space.get_node(parent_state);

        // collect policy output in current EvaluationContext
        vector<OperatorID> operator_ids = current_eval_context.get_preferred_operators(policy);
        vector<float> operator_prefs = current_eval_context.get_preferred_operator_preferences(policy);

        // preferences correspond to operator id by index
        assert(operator_ids.size() == operator_prefs.size());

	// check if these are all applicable actions
	vector<OperatorID> applicable_ops;
	g_successor_generator->generate_applicable_ops(parent_state, applicable_ops);

        // find most probable/ preferenced operator
        int most_probable_op_index = -1;
        float highest_op_probability = 0;
        for (unsigned int index = 0; index < operator_ids.size(); index++) {
            float probability = operator_prefs[index];
	    if (probability > 0) {
		OperatorID op_id = operator_ids[index];
		assert(find(applicable_ops.begin(), applicable_ops.end(), op_id) != applicable_ops.end());
	    }
            if (probability > highest_op_probability) {
                highest_op_probability = probability;
                most_probable_op_index = index;
            }
        }

        // collect most probable operator information
        OperatorID op_id = operator_ids[most_probable_op_index];
        OperatorProxy op_proxy  = task_proxy.get_operators()[op_id];

	// reach new state
        GlobalState new_state = state_registry.get_successor_state(parent_state, op_proxy);
	cout << "Policy reached state with id " << new_state.get_id() << " by applying op " << op_proxy.get_name() << endl;
        SearchNode node = search_space.get_node(new_state);
        statistics.inc_generated();

        if (node.is_new()) {
            // create eval_context with policy
            EvaluationContext eval_context(new_state, &statistics, true, true);
            statistics.inc_evaluated_states();

            bool dead_end = false;
            if (use_heuristic_dead_end_detection) {
                dead_end = eval_context.is_heuristic_infinite(dead_end_heuristic);
            } else {
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
        parser.document_synopsis("Policy search", "");
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
            return make_shared<PolicySearch>(opts);
    }

    static PluginShared<SearchEngine> _plugin("policysearch", _parse);
}
