#include "heuristic_pref_ops_policy.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../heuristics/additive_heuristic.h"
#include "../evaluation_context.h"
#include "../search_statistics.h"

#include <iostream>
using namespace std;

namespace heuristic_pref_ops_policy {
HeuristicPrefOpsPolicy::HeuristicPrefOpsPolicy(const Options &opts)
    : Policy(opts) {
    cout << "Initializing heuristic preferred operators policy..." << endl;
    // this could be any heuristic which sets preferred operators
    additive_heuristic::AdditiveHeuristic hadd(opts);
    heuristic = &hadd;
}

HeuristicPrefOpsPolicy::~HeuristicPrefOpsPolicy() {
}

PolicyResult HeuristicPrefOpsPolicy::compute_policy(const GlobalState &global_state) {
    // HACK: need some context to be able to call compute_result which is necessary to get at the preferred operators
    // computed with the heuristic
    SearchStatistics statistics = SearchStatistics();
    EvaluationContext context = EvaluationContext(global_state, -1, true, &statistics, true);
    EvaluationResult heuristic_result = heuristic->compute_result(context);
    vector<OperatorID> preferred_operators = heuristic_result.get_preferred_operators();

    PolicyResult policy_result = pair<std::vector<OperatorID>, std::vector<float>>(preferred_operators, vector<float>());
    return policy_result;
}

bool HeuristicPrefOpsPolicy::dead_ends_are_reliable() const {
    return heuristic->dead_ends_are_reliable();
}

static Policy *_parse(OptionParser &parser) {
    parser.document_synopsis("Heuristic Preferred Operators Policy", "");
    Policy::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new HeuristicPrefOpsPolicy(opts);
}


static Plugin<Policy> _plugin("heur_pref_ops_pol", _parse);
}
