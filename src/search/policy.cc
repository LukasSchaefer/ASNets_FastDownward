#include "policy.h"

#include "evaluation_context.h"
#include "evaluation_result.h"
#include "globals.h"
#include "option_parser.h"
#include "plugin.h"

#include "task_utils/task_properties.h"
#include "tasks/cost_adapted_task.h"

#include <cassert>
#include <cstdlib>
#include <limits>

using namespace std;

Policy::Policy(const Options &opts)
    : Evaluator(opts.get_unparsed_config(), true, true, true),
      policy_cache(PEntry()),
      cache_policy_values(opts.get<bool>("cache_estimates")),
      register_name(opts.get<string>("register")),
      registered(g_register_policy(this->register_name, this)) {
}

Policy::~Policy() {
    if (registered){
        g_unregister_policy(register_name, this);
    }
}

EvaluationResult Policy::compute_result(EvaluationContext &eval_context) {
    EvaluationResult result;

    assert(eval_context.contains_policy);

    const GlobalState &state = eval_context.get_state();

    vector<OperatorID> operator_ids;
    vector<float> operator_preferences;

    if (cache_policy_values && !policy_cache[state].dirty) {
        operator_ids = policy_cache[state].operator_ids;
        operator_preferences = policy_cache[state].operator_preferences;
        result.set_count_evaluation(false);
    } else {
        PEntry policy_result = compute_policy(state);
        operator_ids = policy_result.operator_ids;
        operator_preferences = policy_result.operator_preferences;
        if (cache_policy_values) {
            policy_cache[state] = PEntry(operator_ids, operator_preferences);
        }
        result.set_count_evaluation(true);
    }

    // TODO fix the error here (some C stuff I currently don't see :/)
    result.set_preferred_operators(operator_ids);
    result.set_operator_preferences(operator_preferences);
    return result;
}