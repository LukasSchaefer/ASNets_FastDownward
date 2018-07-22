#ifndef SEARCH_ENGINES_OPEN_LIST_POLICY_SEARCH_H
#define SEARCH_ENGINES_OPEN_LIST_POLICY_SEARCH_H

#include "../evaluation_context.h"
#include "../search_engine.h"
#include "../heuristic.h"

#include <utility>
#include <vector>
#include <deque>

namespace options {
class Options;
}

namespace open_list_policy_search {

/*
  Policy Search, following a given Policy by naively choosing the
  (first of all) most probable operator(s)
*/
class OpenListPolicySearch : public SearchEngine {
    Policy *policy;
    bool use_heuristic_dead_end_detection = true;
    Heuristic *dead_end_heuristic;
    std::deque<StateID> open_states;
    std::vector<StateID> explored_states;


protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit OpenListPolicySearch(const options::Options &opts);
    virtual ~OpenListPolicySearch() override;

    void set_current_eval_context(StateID state_id);
};
}

#endif