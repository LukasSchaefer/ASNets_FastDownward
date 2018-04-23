#ifndef HEURISTICS_NETWORK_HEURISTIC_H
#define HEURISTICS_NETWORK_HEURISTIC_H

#include "../heuristic.h"

#include "../neural_networks/abstract_network.h"

#include <memory>

namespace network_heuristic {

class NetworkHeuristic : public Heuristic {
    
protected:
    std::shared_ptr<neural_networks::AbstractNetwork> network;
    
    virtual int compute_heuristic(const GlobalState &global_state);
    int compute_heuristic(const State &state);
public:
    explicit NetworkHeuristic(const options::Options &options);
    ~NetworkHeuristic();
};
}

#endif
