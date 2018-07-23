#include "open_list_policy_search.h"

#include "../policy.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../task_utils/regression_utils.h"
#include "../task_utils/successor_generator.h"

#include<tuple>

using namespace std;
using utils::ExitCode;

namespace open_list_policy_search {
    using Plan = std::vector<OperatorID>;

    bool compareOperatorPrefs(std::pair<OperatorID, float> &a, std::pair<OperatorID, float> &b)
    {
        return a.second < b.second;
    }

    OpenListPolicySearch::OpenListPolicySearch(
    const Options &opts)
    : SearchEngine(opts),
      policy(opts.get<Policy *>("p")),
      use_heuristic_dead_end_detection(opts.get<bool>("dead_end_detection")),
      open_states(deque<StateID>()),
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

    OpenListPolicySearch::~OpenListPolicySearch() {
    }

    void OpenListPolicySearch::initialize() {
        assert(policy);
        cout << "Conducting policy search with open list" << endl;

        GlobalState initial_state = state_registry.get_initial_state();
        open_states.push_front(initial_state.get_id());
        explored_states.push_back(initial_state.get_id());
        SearchNode node = search_space.get_node(initial_state);
	node.open_initial();
    }

    SearchStatus OpenListPolicySearch::step() {
	cout << "Size of open list at step: " << open_states.size() << endl;
        if (open_states.empty()) {
	    cout << "Empty open list" << endl;
            cout << "No solution - FAILED" << endl;
            return FAILED;
        }

        StateID current_state_id = open_states.front();
        open_states.pop_front();

        GlobalState current_state = state_registry.lookup_state(current_state_id);

        if (check_goal_and_set_plan(current_state)) {
            return SOLVED;
        }

        // collect current state and search node
        SearchNode current_node = search_space.get_node(current_state);
        // create eval_context with policy
        EvaluationContext current_eval_context(current_state, &statistics, true, true);

        // check for dead-end
        bool dead_end = false;
        if (use_heuristic_dead_end_detection) {
            dead_end = current_eval_context.is_heuristic_infinite(dead_end_heuristic);
        } else {
            dead_end = current_eval_context.is_policy_dead_end(policy);
        }
        if (dead_end) {
            cout << "Dead-end encountered" << endl;
            current_node.mark_as_dead_end();
            statistics.inc_dead_ends();
	    return IN_PROGRESS;
        }

        cout << "Computing policy for " << current_state.get_id() << endl;
        // collect policy output in current EvaluationContext
        vector<OperatorID> operator_ids = current_eval_context.get_preferred_operators(policy);
        vector<float> operator_prefs = current_eval_context.get_preferred_operator_preferences(policy);

        // preferences correspond to operator id by index
        assert(operator_ids.size() == operator_prefs.size());

        // collect pairs of op_ids and preferences
        vector<pair<OperatorID, float>> operator_with_prefs;
        for (unsigned int index = 0; index < operator_ids.size(); index++) {
            pair<OperatorID, float> op_pair = make_pair(operator_ids[index], operator_prefs[index]);
            operator_with_prefs.push_back(op_pair);
        }

        statistics.inc_evaluated_states();
        // sort after increasing preference
        std::sort(operator_with_prefs.begin(), operator_with_prefs.end(), compareOperatorPrefs);

        /* add reached states to open_list with smallest preference first
        (greatest probability first in deque in the end) */
        for (pair<OperatorID, float> op_pair : operator_with_prefs) {
            OperatorID op_id = op_pair.first;
            OperatorProxy op_proxy  = task_proxy.get_operators()[op_id];

            // reach new state
            GlobalState new_state = state_registry.get_successor_state(current_state, op_proxy);
            if (find(explored_states.begin(), explored_states.end(), new_state.get_id()) != explored_states.end()) {
               // state was already explored -> skip
               continue;
            }
            SearchNode node = search_space.get_node(new_state);
            statistics.inc_generated();
            node.open(current_node, op_proxy);
            open_states.push_front(new_state.get_id());
            explored_states.push_back(new_state.get_id());
        }
        return IN_PROGRESS;
    }

    static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
        parser.document_synopsis("Policy search with open list", "");
        parser.add_option<Policy *>("p", "policy");
        parser.add_option<bool>("dead_end_detection",
        "Boolean value indicating whether early dead-end detection using "
        "a heuristic function should be used during search", "true");
        parser.add_option<Heuristic *>("dead_end_detection_heuristic",
        "heuristic used for early dead-end detection", "ff");
        SearchEngine::add_options_to_parser(parser);
        Options opts = parser.parse();

        if (parser.dry_run())
            return nullptr;
        else
            return make_shared<OpenListPolicySearch>(opts);
    }

    static PluginShared<SearchEngine> _plugin("openlistpolsearch", _parse);
}
