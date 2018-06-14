#ifndef NETWORK_POLICY_H
#define NETWORK_POLICY_H

#include "../policy.h"

#include "../neural_networks/abstract_network.h"

#include <memory>

namespace network_policy {
// Give state to network and uses its policy output as policy
class NetworkPolicy : public Policy {
    
protected:
    std::shared_ptr<neural_networks::AbstractNetwork> network;
    
    virtual PolicyResult compute_policy(const GlobalState &state);
    PolicyResult compute_policy(const State &state);
public:
    explicit NetworkPolicy(const options::Options &options);
    ~NetworkPolicy();
};
}

#endif