#ifndef SEARCH_ENGINES_POLICY_SEARCH_H
#define SEARCH_ENGINES_POLICY_SEARCH_H

#include "../evaluation_context.h"
#include "../open_list.h"
#include "../search_engine.h"
#include "../heuristic.h"

#include <utility>
#include <vector>

namespace options {
class Options;
}

namespace policy_search {

/*
  Policy Search, following a given Policy by naively choosing the
  (first of all) most probable operator(s)
*/
class PolicySearch : public SearchEngine {
    Policy *policy;
    bool use_heuristic_dead_end_detection = true;
    const int exploration_trajectory_limit;
    EvaluationContext current_eval_context;
    int trajectory_length = 0;
    Heuristic *dead_end_heuristic;


protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit PolicySearch(const options::Options &opts);
    virtual ~PolicySearch() override;

    void set_current_eval_context(StateID state_id);
};
}

#endif
