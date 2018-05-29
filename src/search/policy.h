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
    /*
        Entries for Policy results including
        dirty: true if vectors/ values are not set yet 
        operator_ids: vector of IDs for operators the policy considers
        operator_preferences: vector of operator preferences (= probabilities) for the same operators
            with index matching the operator IDs of the previous vector
    */
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

    /* 
        registration name and flag for g_registered_policies map in globals
    */
    const std::string register_name;
    const bool registered;

protected:

    /*
        main function to implement for concrete policies returning
        the policy result for a given state
    */
    virtual PEntry compute_policy(const GlobalState &state) = 0;

public:
    explicit Policy(const options::Options &options);
    virtual ~Policy() override;

    virtual void get_involved_heuristics(std::set<Heuristic *> &hset) {
    }

    virtual bool dead_ends_are_reliable() const = 0;

    static void add_options_to_parser(options::OptionParser &parser);

    virtual EvaluationResult compute_result(
        EvaluationContext &eval_context) override;
};

#endif