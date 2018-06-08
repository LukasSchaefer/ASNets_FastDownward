#ifndef HEURISTIC_POLICY_H
#define HEURISTIC_POLICY_H

#include "../policy.h"
#include "../heuristic.h"

#include <vector>

class FactProxy;
class GlobalState;
class OperatorProxy;

namespace heuristic_policy {

/*
    Policy which uses an heuristic and only follows it by choosing
    the action leading to the most promising state (according to the heuristic)
*/
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