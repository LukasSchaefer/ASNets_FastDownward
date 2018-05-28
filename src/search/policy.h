#ifndef POLICY_H
#define POLICY_H

#include "evaluator.h"
#include "operator_id.h"
#include "per_state_information.h"
#include "task_proxy.h"

#include "algorithms/ordered_set.h"

#include <memory>
#include <vector>
#include <unordered_set>

class GlobalState;
class TaskProxy;

namespace options {
class OptionParser;
class Options;
}

class Policy : public Evaluator {
    struct PEntry {
        bool dirty;
        std::vector<OperatorID> operator_ids;
        std::vector<float> operator_preferences;

        PEntry(std::vector<OperatorID> operator_ids, std::vector<float> operator_preferences)
            : dirty(false), operator_ids(operator_ids), operator_preferences(operator_preferences) {
        }

        PEntry() : dirty(true), operator_ids(std::vector<OperatorID>()), operator_preferences(std::vector<float>()) {
        }
    };

protected:
    /*
      Cache for saving policy results
      Before accessing this cache always make sure that the cache_policy_values
      flag is set to true - as soon as the cache is accessed it will create
      entries for all existing states
    */
    PerStateInformation<PEntry> policy_cache;
    bool cache_policy_values;

    const std::string register_name;
    const bool registered;

protected:

    virtual PEntry compute_policy(const GlobalState &state) = 0;

    // what is this for? Needed in Policy?
    // State convert_global_state(const GlobalState &global_state) const;

public:
    explicit Policy(const options::Options &options);
    virtual ~Policy() override;

    virtual void get_involved_heuristics(std::set<Heuristic *> &hset) {
    }

    static void add_options_to_parser(options::OptionParser &parser);

    virtual EvaluationResult compute_result(
        EvaluationContext &eval_context) override;
};

#endif