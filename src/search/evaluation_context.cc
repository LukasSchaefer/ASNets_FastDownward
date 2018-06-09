#include "evaluation_context.h"

#include "evaluation_result.h"
#include "evaluator.h"
#include "search_statistics.h"
#include "policy.h"

#include <cassert>

using namespace std;


EvaluationContext::EvaluationContext(
    const HeuristicCache &cache, int g_value, bool is_preferred,
    SearchStatistics *statistics, bool calculate_preferred,
    bool contains_policy)
    : cache(cache),
      contains_policy_information(contains_policy),
      g_value(g_value),
      preferred(is_preferred),
      statistics(statistics),
      calculate_preferred(calculate_preferred) {
}

EvaluationContext::EvaluationContext(
    const GlobalState &state, int g_value, bool is_preferred,
    SearchStatistics *statistics, bool calculate_preferred,
    bool contains_policy)
    : EvaluationContext(HeuristicCache(state), g_value, is_preferred, statistics, calculate_preferred, contains_policy) {
}

EvaluationContext::EvaluationContext(
    const GlobalState &state,
    SearchStatistics *statistics, bool calculate_preferred,
    bool contains_policy)
    : EvaluationContext(HeuristicCache(state), INVALID, false, statistics, calculate_preferred, contains_policy) {
}

const EvaluationResult &EvaluationContext::get_result(Evaluator *evaluator) {
    EvaluationResult &result = cache[evaluator];
    if (result.is_uninitialized()) {
        result = evaluator->compute_result(*this);
        if (statistics &&
            evaluator->is_used_for_counting_evaluations() &&
            result.get_count_evaluation()) {
            statistics->inc_evaluations();
        }
    }
    return result;
}

const HeuristicCache &EvaluationContext::get_cache() const {
    return cache;
}

const GlobalState &EvaluationContext::get_state() const {
    return cache.get_state();
}

int EvaluationContext::get_g_value() const {
    assert(g_value != INVALID);
    return g_value;
}

bool EvaluationContext::is_preferred() const {
    assert(g_value != INVALID);
    return preferred;
}

void EvaluationContext::set_contains_policy() {
    contains_policy_information = true;
}

bool EvaluationContext::contains_policy() const {
    return contains_policy_information;
}

bool EvaluationContext::is_heuristic_infinite(Evaluator *heur) {
    return get_result(heur).is_infinite();
}

bool EvaluationContext::is_policy_dead_end(Evaluator *policy) {
    return get_result(policy).get_preferred_operators().empty();
}

int EvaluationContext::get_heuristic_value(Evaluator *heur) {
    int h = get_result(heur).get_h_value();
    assert(h != EvaluationResult::INFTY);
    return h;
}

int EvaluationContext::get_heuristic_value_or_infinity(Evaluator *heur) {
    return get_result(heur).get_h_value();
}

const vector<OperatorID> & EvaluationContext::get_preferred_operators(Evaluator *heur) {
    return get_result(heur).get_preferred_operators();
}

const std::vector<float> &EvaluationContext::get_preferred_operator_preferences(Evaluator *eval) {
    return get_result(eval).get_operator_preferences();
}

bool EvaluationContext::get_calculate_preferred() const {
    return calculate_preferred;
}
