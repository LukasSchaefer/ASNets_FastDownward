#include "regression_utils.h"

#include "../globals.h"

#include "../task_utils/task_properties.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

using namespace std;

RegressionCondition::RegressionCondition(int var, int value)
: data(FactPair(var, value)) { }

bool RegressionCondition::is_satisfied(const PartialAssignment &assignment) const {
    int current_val = assignment[data.var].get_value();
    return current_val == PartialAssignment::UNASSIGNED || current_val == data.value;
}

RegressionConditionProxy::RegressionConditionProxy(const AbstractTask& task, const RegressionCondition& condition)
: task(&task), condition(condition) { }

RegressionConditionProxy::RegressionConditionProxy(const AbstractTask& task, int var_id, int value)
: RegressionConditionProxy(task, RegressionCondition(var_id, value)) { }

RegressionEffect::RegressionEffect(int var, int value)
: data(FactPair(var, value)) { }

RegressionEffectProxy::RegressionEffectProxy(const AbstractTask& task, const RegressionEffect& effect)
: task(&task), effect(effect) { }

RegressionEffectProxy::RegressionEffectProxy(const AbstractTask& task, int var_id, int value)
: RegressionEffectProxy(task, RegressionEffect(var_id, value)) { }

RegressionOperator::RegressionOperator(OperatorProxy &op)
: original_index(op.get_id()),
cost(op.get_cost()),
name(op.get_name()),
is_an_axiom(op.is_axiom()){
    /*
      1) pre(v) = x, eff(v) = y  ==>  rpre(v) = y, reff(v) = x
      2) pre(v) = x, eff(v) = -  ==>  rpre(v) = x, reff(v) = x
      3) pre(v) = -, eff(v) = y  ==>  rpre(v) = y, reff(v) = u
     */
    unordered_set<int> precondition_vars;
    unordered_map<int, int> vars_to_effect_values;

    for (EffectProxy effect : op.get_effects()) {
        FactProxy fact = effect.get_fact();
        int var_id = fact.get_variable().get_id();
        vars_to_effect_values[var_id] = fact.get_value();
        original_effect_vars.insert(var_id);
    }

    // Handle cases 1 and 2 where preconditions are defined.
    for (FactProxy precondition : op.get_preconditions()) {
        int var_id = precondition.get_variable().get_id();
        precondition_vars.insert(var_id);
        effects.emplace_back(var_id, precondition.get_value());
        if (vars_to_effect_values.count(var_id)) {
            // Case 1, effect defined.
            preconditions.emplace_back(var_id, vars_to_effect_values[var_id]);
        } else {
            // Case 2, effect undefined.
            preconditions.emplace_back(var_id, precondition.get_value());
        }
    }

    // Handle case 3 where preconditions are undefined.
    for (EffectProxy effect : op.get_effects()) {
        FactProxy fact = effect.get_fact();
        int var_id = fact.get_variable().get_id();
        if (precondition_vars.count(var_id) == 0) {
            preconditions.emplace_back(var_id, fact.get_value());
            //WHY linker error if using PartialAssignment::UNASSIGNED directly?
            const int unassigned = PartialAssignment::UNASSIGNED;
            effects.emplace_back(var_id, unassigned);//PartialAssignment::UNASSIGNED);
        }
    }
}

bool RegressionOperator::is_applicable(const PartialAssignment &assignment) const {
    return (any_of(original_effect_vars.begin(), original_effect_vars.end(),
        [&] (int var_id) {
            return assignment.assigned(var_id);
        })
    && all_of(preconditions.begin(), preconditions.end(),
        [&](const RegressionCondition & condition) {
            return condition.is_satisfied(assignment);
        }));
}

inline vector<int> get_domain_sizes(const AbstractTask &task){
    vector<int> sizes;
    for (int i = 0; i < task.get_num_variables(); ++i) {
        sizes.push_back(task.get_variable_domain_size(i));
    }
    return sizes;
}

inline vector<RegressionOperator> extract_regression_operators(const AbstractTask& task, TaskProxy &tp) {
    task_properties::verify_no_axioms(tp);
    task_properties::verify_no_conditional_effects(tp);

    vector<RegressionOperator> rops;
    for (OperatorProxy op : OperatorsProxy(task)) {
        rops.emplace_back(op);
    }
    return rops;
}

RegressionTaskProxy::RegressionTaskProxy(const AbstractTask& task)
: TaskProxy(task),
possess_mutexes(g_num_mutexes() > 0),
domain_sizes(get_domain_sizes(task)),
operators(extract_regression_operators(task, *this)) { }


bool contains_mutex(const vector<int> &values) {

    for (size_t var1 = 0; var1 < values.size(); ++var1) {
        assert(utils::in_bounds(var1, values));
            int value1 = values[var1];
        if (value1 == PartialAssignment::UNASSIGNED) {
            continue;
        }
        FactPair fp1(var1, value1);

        for (size_t var2 = var1 + 1; var2 < values.size(); ++var2) {
            assert(utils::in_bounds(var2, values));
                int value2 = values[var2];
            if (value2 == PartialAssignment::UNASSIGNED) {
                continue;
            }
            FactPair fp2(var2, value2);

            if (are_mutex(fp1, fp2)) {

                return true;
            }
        }
    }
    return false;
}

/*
  Replace values[var] with non-mutex value. Return true iff such a
  non-mutex value could be found.
 */
static bool extend_with_non_mutex_value(vector<int> &values, int var,
    int domain_size, utils::RandomNumberGenerator &rng) {
    utils::in_bounds(var, values);

        int &value = values[var];
        vector<int> domain(domain_size);
        iota(domain.begin(), domain.end(), 0);
        rng.shuffle(domain);
    for (int new_value : domain) {
        value = new_value;
        if (!contains_mutex(values)) {

            return true;
        }
    }
    return false;
}


static const int MAX_TRIES_EXTEND = 100;

    static bool replace_dont_cares_with_non_mutex_values(
    const RegressionTaskProxy &task_proxy, vector<int> &values,
    utils::RandomNumberGenerator &rng) {

    int num_vars = task_proxy.get_variables().size();
        /* Try extending the partial state for a fixed number of times
           before giving up. It may be impossible to find a non-mutex
           extension. */
    for (int round = 0; round < MAX_TRIES_EXTEND; ++round) {
        vector<int> new_values = values;
        for (VariableProxy var : task_proxy.get_variables()) {
            int domain_size = var.get_domain_size();
                int var_id = var.get_id();
                int &value = new_values[var_id];
            if (value == PartialAssignment::UNASSIGNED) {
                bool found_non_mutex_value =
                    extend_with_non_mutex_value(new_values, var_id, domain_size, rng);
                if (!found_non_mutex_value) {
                    break;
                }
            }
            if (var.get_id() == num_vars - 1) {
                values = new_values;

                return true;
            }
        }
    }
    return false;
}



pair<bool, State> RegressionTaskProxy::convert_to_full_state(PartialAssignment& assignment,
    bool check_mutexes, utils::RandomNumberGenerator &rng) const {

    vector<int> new_values = assignment.get_values();
        bool success = true;
    if (check_mutexes) {
        success = replace_dont_cares_with_non_mutex_values((*this),
            new_values, rng);
    } else {
        for (VariableProxy var : VariablesProxy(*task)) {
            int &value = new_values[var.get_id()];
            if (value == PartialAssignment::UNASSIGNED) {
                int domain_size = var.get_domain_size();
                    value = rng(domain_size);
            }
        }
    }
    return make_pair(success, State(*task, move(new_values)));
}
