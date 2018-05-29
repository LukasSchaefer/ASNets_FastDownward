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

    assert(eval_context.contains_policy());

    const GlobalState &state = eval_context.get_state();

    vector<OperatorID> operator_ids;
    vector<float> operator_preferences;

    if (cache_policy_values && !policy_cache[state].dirty) {
        operator_ids = policy_cache[state].operator_ids;
        operator_preferences = policy_cache[state].operator_preferences;
        result.set_count_evaluation(false);
    } else {
        pair<std::vector<OperatorID>, std::vector<float>> policy_result = compute_policy(state);
        operator_ids = policy_result.first;
        operator_preferences = policy_result.second;
        if (!operator_ids.empty() && operator_preferences.empty()) {
            // only operator_ids set -> use uniform distribution = all preferences are equal
            unsigned int number_of_operators = operator_ids.size();
            operator_preferences.resize(number_of_operators);
            for (unsigned int i = 0; i < number_of_operators; i++) {
                operator_preferences[i] = 1/number_of_operators;
            }
        }
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

void Policy::add_options_to_parser(OptionParser &parser) {
    parser.add_option<string>("register", "Registers a policy pointer by a"
        "given name on the task object.", "None");
    parser.add_option<bool>("cache_estimates", "cache policy estimates", "true");

}

static PluginTypePlugin<Policy> _type_plugin(
    "Policy",
    // TODO: Add information for wiki page
    "");