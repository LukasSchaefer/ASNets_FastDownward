#ifndef EVALUATION_RESULT_H
#define EVALUATION_RESULT_H

#include "operator_id.h"

#include <limits>
#include <vector>

class EvaluationResult {
    static const int UNINITIALIZED = -2;

    int h_value;
    std::vector<OperatorID> preferred_operators;
    // probabilities to represent policy with preferred operators
    // ith value is probability to choose ith preferred operator
    std::vector<float> operator_preferences;
    bool count_evaluation;
public:
    // "INFINITY" is an ISO C99 macro and "INFINITE" is a macro in windows.h.
    static const int INFTY;

    EvaluationResult();

    /* TODO: Can we do without this "uninitialized" business?

       One reason why it currently exists is to simplify the
       implementation of the EvaluationContext class, where we don't
       want to perform two separate map operations in the case where
       we determine that an entry doesn't yet exist (lookup) and hence
       needs to be created (insertion). This can be avoided most
       easily if we have a default constructor for EvaluationResult
       and if it's easy to test if a given object has just been
       default-constructed.
    */

    /*
      TODO: Can we get rid of count_evaluation?
      The EvaluationContext needs to know (for statistics) if the
      heuristic actually computed the heuristic value or just looked it
      up in a cache. Currently this information is passed over the
      count_evaluation flag, which is somewhat awkward.
    */

    bool is_uninitialized() const;
    bool is_infinite() const;
    int get_h_value() const;
    bool get_count_evaluation() const;
    const std::vector<OperatorID> &get_preferred_operators() const;
    const std::vector<float> &get_operator_preferences() const;

    void set_h_value(int value);
    void set_preferred_operators(std::vector<OperatorID> &&preferred_operators);
    void set_operator_preferences(std::vector<float> &&operator_preferences);
    void set_count_evaluation(bool count_eval);
};

#endif
