#include "heuristic_policy.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../heuristics/additive_heuristic.h"
#include "../evaluation_context.h"
#include "../search_statistics.h"

#include <iostream>
using namespace std;

namespace heuristic_policy {
HeuristicPolicy::HeuristicPolicy(const Options &opts)
    : Policy(opts) {
    cout << "Initializing heuristic policy..." << endl;
    // this could be any heuristic which sets preferred operators
    heuristic = &additive_heuristic::AdditiveHeuristic(opts);
}

HeuristicPolicy::~HeuristicPolicy() {
}

pair<std::vector<OperatorID>, std::vector<float>> HeuristicPolicy::compute_policy(const GlobalState &global_state) {
    // HACK: need some context to be able to call compute_result which is necessary to get at the preferred operators
    // computed with the heuristic
    EvaluationContext context = EvaluationContext(global_state, -1, true, &SearchStatistics(), true);
    EvaluationResult heuristic_result = heuristic->compute_result(context);
    vector<OperatorID> preferred_operators = heuristic_result.get_preferred_operators();

    pair<std::vector<OperatorID>, std::vector<float>> policy_result = pair<std::vector<OperatorID>, std::vector<float>>(preferred_operators, vector<float>());
    return policy_result;
}

static Policy *_parse(OptionParser &parser) {
    parser.document_synopsis("Heuristic Policy", "");
    Policy::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new HeuristicPolicy(opts);
}


static Plugin<Policy> _plugin("heur_pol", _parse);
}