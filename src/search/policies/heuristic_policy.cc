#include "heuristic_policy.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../evaluation_context.h"
#include "../search_statistics.h"

#include <iostream>
using namespace std;

namespace heuristic_policy {
HeuristicPolicy::HeuristicPolicy(const Options &opts)
    : Policy(opts),
      heuristic(opts.get_list<Heuristic *>("heuristic")){
    cout << "Initializing heuristic policy..." << endl;
}

HeuristicPolicy::~HeuristicPolicy() {
}

PolicyResult HeuristicPolicy::compute_policy(const GlobalState &global_state) {
    vector<OperatorID> applicable_ops;
    g_successor_generator->generate_applicable_ops(global_state, applicable_ops);
    int h_best = -1;
    OperatorID op_best;
    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];

        GlobalState succ_state = state_registry.get_successor_state(global_state, op);

        int h = heuristic->compute_heuristic(global_state);
        if (h_best == -1 || h < h_best) {
            h_best = h;
            op_best = op_id;
        }
    }

    vector<OperatorID> preferred_operators = vector<OperatorID>();
    preferred_operators.append(op_best);

    PolicyResult policy_result = pair<std::vector<OperatorID>, std::vector<float>>(preferred_operators, vector<float>());
    return policy_result;
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