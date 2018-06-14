#include "network_policy.h"

#include "../global_state.h"
#include "../option_parser.h"
#include "../plugin.h"

#include "../task_utils/task_properties.h"

#include <cassert>

using namespace std;

namespace network_policy {
// construction and destruction
NetworkPolicy::NetworkPolicy(const Options &opts)
    : Policy(opts),
      network(opts.get<shared_ptr<neural_networks::AbstractNetwork>>("network")) {
    cout << "Initializing network policy..." << endl;
    network->verify_policy();
    network->initialize();
    
    
}

NetworkPolicy::~NetworkPolicy() {
}

PolicyResult NetworkPolicy::compute_policy(const GlobalState &global_state) {
    State state = convert_global_state(global_state);
    return compute_policy(state);
}

PolicyResult NetworkPolicy::compute_policy(const State &state) {
    network->evaluate(state);
    PolicyResult policy_output = network->get_policy();
    
    return policy_output;
}

bool NetworkPolicy::dead_ends_are_reliable() const {
    return network->dead_ends_are_reliable();
}

static Policy *_parse(OptionParser &parser) {
    parser.document_synopsis("network policy", "");
    parser.document_property("preferred operators",
        "yes (representing policy)");

    Policy::add_options_to_parser(parser);
    parser.add_option<shared_ptr<neural_networks::AbstractNetwork>>("network",
        "Network for state evaluations.");
    Options opts = parser.parse();
    if (parser.dry_run())
        return 0;
    else
        return new NetworkPolicy(opts);
}

static Plugin<Policy> _plugin("np", _parse);
}
