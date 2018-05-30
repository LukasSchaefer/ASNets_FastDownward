#ifndef POLICY_H
#define POLICY_H

#include "evaluator.h"
#include "operator_id.h"
#include "per_state_information.h"

#include <vector>
#include <tuple>

namespace options {
class OptionParser;
class Options;
}

struct PEntry;

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

        // non-dirty constructor with ids and preferences
        PEntry(std::vector<OperatorID> operator_ids, std::vector<float> operator_preferences)
            : dirty(false), operator_ids(operator_ids), operator_preferences(operator_preferences) {
        }

        /*
            non-dirty constructor only using ids (e.g. can be used when all operators should have same
            preference which is handled in Policy::compute_result)
        */
        PEntry(std::vector<OperatorID> operator_ids)
            : dirty(false), operator_ids(operator_ids), operator_preferences(std::vector<float>()) {
        }

        // dirty empty constructor
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
        main function to implement for concrete policies returning the policy
        result as pair of operator ids and preferences for a given state
    */
    virtual std::pair<std::vector<OperatorID>, std::vector<float>> compute_policy(const GlobalState &state) = 0;

public:
    explicit Policy(const options::Options &options);
    virtual ~Policy() override;

    virtual bool dead_ends_are_reliable() const = 0;
    virtual void get_involved_heuristics(std::set<Heuristic *> &hset) {
        return;
    }

    static void add_options_to_parser(options::OptionParser &parser);

    virtual EvaluationResult compute_result(
        EvaluationContext &eval_context) override;
};

#endif