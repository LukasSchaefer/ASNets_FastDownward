#include "network_heuristic.h"

#include "../global_state.h"
#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"

#include <cassert>

using namespace std;

namespace network_heuristic {
// construction and destruction
NetworkHeuristic::NetworkHeuristic(const Options &opts)
    : Heuristic(opts),
      network(opts.get<shared_ptr<neural_networks::AbstractNetwork>>("network")) {
    cout << "Initializing network heuristic..." << endl;
    network->verify_heuristic();
    
}

NetworkHeuristic::~NetworkHeuristic() {
}

int NetworkHeuristic::compute_heuristic(const GlobalState &global_state) {
    State state = convert_global_state(global_state);
    return compute_heuristic(state);
}

int NetworkHeuristic::compute_heuristic(const State &state) {
    network->evaluate(state);
    int h = network->get_heuristic();
    
    if(network->is_preferred()){
        for (const OperatorID &oid: network->get_preferred()){
            set_preferred(oid);
        }
    }
    
    return h;
}


static Heuristic *_parse(OptionParser &parser) {
    parser.document_synopsis("network heuristic", "");
    parser.document_language_support("action costs", "supported");
    parser.document_language_support("conditional effects", "supported");
    parser.document_language_support("axioms", "supported");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "no");
    parser.document_property("preferred operators",
        "maybe (depends on network)");

    Heuristic::add_options_to_parser(parser);
    parser.add_option<shared_ptr<neural_networks::AbstractNetwork>>("network",
        "Network for state evaluations.");
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new NetworkHeuristic(opts);
}

static Plugin<Heuristic> _plugin("nh", _parse);
}


