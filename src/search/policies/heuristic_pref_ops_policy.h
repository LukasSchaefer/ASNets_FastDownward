#ifndef HEURISTIC_PREF_OPS_POLICY_H
#define HEURISTIC_PREF_OPS_POLICY_H

#include "../policy.h"
#include "../heuristic.h"

#include <vector>

class FactProxy;
class GlobalState;
class OperatorProxy;

namespace heuristic_pref_ops_policy {

/*
    Policy which uses internally an heuristic which computes
    preferred operators which are then uniformally used as
    an policy
    Currently used additive heuristic! (set in constructor)
*/
class HeuristicPrefOpsPolicy : public Policy {
    Heuristic *heuristic;
protected:
    PolicyResult compute_policy(const GlobalState &state);
public:
    HeuristicPrefOpsPolicy(const options::Options &options);
    ~HeuristicPrefOpsPolicy();
    bool dead_ends_are_reliable() const;
};
}

#endif