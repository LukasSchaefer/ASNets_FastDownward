#ifndef SEARCH_ENGINES_POLICY_SEARCH_H
#define SEARCH_ENGINES_POLICY_SEARCH_H

#include "../evaluation_context.h"
#include "../open_list.h"
#include "../search_engine.h"

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

    EvaluationContext current_eval_context;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit PolicySearch(const options::Options &opts);
    virtual ~PolicySearch() override;
};
}

#endif