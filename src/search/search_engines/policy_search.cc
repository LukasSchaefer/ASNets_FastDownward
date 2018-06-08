#include "policy_search.h"

#include "../policy.h"
#include "../option_parser.h"
#include "../plugin.h"

using namespace std;
using utils::ExitCode;

namespace policy_search {
    PolicySearch::PolicySearch(
    const Options &opts)
    : SearchEngine(opts),
      policy(opts.get<Policy *>("p")),
      current_eval_context(state_registry.get_initial_state(), &statistics) {
}

PolicySearch::~PolicySearch() {
}

void PolicySearch::initialize() {
    assert(policy);
    cout << "Conducting policy search" << endl;

    current_eval_context.set_contains_policy();
    bool dead_end = current_eval_context.is_policy_dead_end(policy);
    statistics.inc_evaluated_states();

    if (dead_end) {
        cout << "Initial state is a dead end, no solution" << endl;
        if (policy->dead_ends_are_reliable())
            utils::exit_with(ExitCode::UNSOLVABLE);
        else
            utils::exit_with(ExitCode::UNSOLVED_INCOMPLETE);
    }

    SearchNode node = search_space.get_node(current_eval_context.get_state());
    node.open_initial();
}

SearchStatus PolicySearch::step() {
    search_progress.check_progress(current_eval_context);

    if (check_goal_and_set_plan(current_eval_context.get_state())) {
        return SOLVED;
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

    // find most probable/ preferenced operator
    int most_probable_op_index = -1;
    float highest_op_probability = 0;
    for (unsigned int index = 0; index < operator_ids.size(); index++) {
        float probability = operator_prefs[index];
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
    SearchNode node = search_space.get_node(new_state);
    statistics.inc_generated();

    if (node.is_new()) {
        // create eval_context with policy
        EvaluationContext eval_context(new_state, &statistics, true);
        statistics.inc_evaluated_states();

        if (eval_context.is_policy_dead_end(policy)) {
            node.mark_as_dead_end();
            statistics.inc_dead_ends();
            cout << "No solution - FAILED" << endl;
            return FAILED;
        }
        node.open(parent_node, op_proxy);

        current_eval_context = eval_context;
        return IN_PROGRESS;
    }

    cout << "No solution - FAILED" << endl;
    return FAILED;
}

static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
    parser.document_synopsis("Policy search", "");
    parser.add_option<Policy *>("p", "policy");
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<PolicySearch>(opts);
}

static PluginShared<SearchEngine> _plugin("policysearch", _parse);
}