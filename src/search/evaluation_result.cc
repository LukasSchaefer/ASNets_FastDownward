#include "evaluation_result.h"

using namespace std;

const int EvaluationResult::INFTY = numeric_limits<int>::max();

EvaluationResult::EvaluationResult() : h_value(UNINITIALIZED) {
}

bool EvaluationResult::is_uninitialized() const {
    // first condition for heuristics, second for policies
    return (h_value == UNINITIALIZED ||
        (!preferred_operators.empty() && !operator_preferences.empty()));
}

bool EvaluationResult::is_infinite() const {
    return h_value == INFTY;
}

int EvaluationResult::get_h_value() const {
    return h_value;
}

const vector<OperatorID> &EvaluationResult::get_preferred_operators() const {
    return preferred_operators;
}

const vector<float> &EvaluationResult::get_operator_preferences() const {
    return operator_preferences;
}

bool EvaluationResult::get_count_evaluation() const {
    return count_evaluation;
}

void EvaluationResult::set_h_value(int value) {
    h_value = value;
}

void EvaluationResult::set_preferred_operators(
    vector<OperatorID> &&preferred_ops) {
    preferred_operators = move(preferred_ops);
}

void EvaluationResult::set_operator_preferences(
    std::vector<float> &&operator_preferences) {
    operator_preferences = move(operator_preferences);
}

void EvaluationResult::set_count_evaluation(bool count_eval) {
    count_evaluation = count_eval;
}
