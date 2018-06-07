#ifndef HEURISTIC_POLICY_H
#define HEURISTIC_POLICY_H

#include "../policy.h"
#include "../heuristic.h"

#include <vector>

class FactProxy;
class GlobalState;
class OperatorProxy;

namespace heuristic_policy {

class HeuristicPolicy : public Policy {
    Heuristic *heuristic;
protected:
    PolicyResult compute_policy(const GlobalState &state);
public:
    HeuristicPolicy(const options::Options &options);
    ~HeuristicPolicy();
    bool dead_ends_are_reliable() const;
};
}

#endif