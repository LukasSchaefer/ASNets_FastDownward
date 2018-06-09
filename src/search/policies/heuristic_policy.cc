#include "heuristic_policy.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../evaluation_context.h"
#include "../search_statistics.h"
#include "../task_utils/successor_generator.h"
#include "../operator_id.h"

#include <iostream>
using namespace std;

namespace heuristic_policy {
HeuristicPolicy::HeuristicPolicy(const Options &opts)
    : Policy(opts),
      state_registry(
        *task, *g_state_packer, *g_axiom_evaluator, task->get_initial_state_values()),
      heuristic(opts.get<Heuristic *>("h")) {
    cout << "Initializing heuristic policy..." << endl;
}

HeuristicPolicy::~HeuristicPolicy() {
}

PolicyResult HeuristicPolicy::compute_policy(const GlobalState &global_state) {
    // collect all applicable actions
    vector<OperatorID> applicable_ops;
    g_successor_generator->generate_applicable_ops(global_state, applicable_ops);

    // look for action leading to the state with best (lowest) heuristic value
    int h_best = -1;
    OperatorID op_best = OperatorID::no_operator;
    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];

        GlobalState succ_state = state_registry.get_successor_state(global_state, op);

        // HACK: need some context to be able to call compute_result which is necessary to get at the preferred operators
        // computed with the heuristic
        SearchStatistics statistics = SearchStatistics();
        EvaluationContext context = EvaluationContext(succ_state, -1, true, &statistics, true);
        EvaluationResult heuristic_result = heuristic->compute_result(context);
        int h = heuristic_result.get_h_value();
        // better or first action found
        if (h_best == -1 || h < h_best) {
            h_best = h;
            op_best = op_id;
        }
    }

    vector<OperatorID> preferred_operators = vector<OperatorID>();
    if (op_best != OperatorID::no_operator) {
        // add only best action to preferred operator list
        preferred_operators.push_back(op_best);
    }

    PolicyResult policy_result = pair<std::vector<OperatorID>, std::vector<float>>(preferred_operators, vector<float>());
    return policy_result;
}

bool HeuristicPolicy::dead_ends_are_reliable() const {
    return heuristic->dead_ends_are_reliable();
}

static Policy *_parse(OptionParser &parser) {
    parser.document_synopsis("Heuristic Policy", "");
    parser.add_option<Heuristic *>("h", "heuristic");
    Policy::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new HeuristicPolicy(opts);
}


static Plugin<Policy> _plugin("heur_pol", _parse);
}