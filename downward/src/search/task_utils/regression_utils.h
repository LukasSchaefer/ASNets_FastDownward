#ifndef TASK_UTILS_REGRESSION_UTILS_H
#define TASK_UTILS_REGRESSION_UTILS_H

#include "../abstract_task.h"
#include "../global_state.h"
#include "../operator_id.h"
#include "../task_proxy.h"

#include "../utils/rng.h"

#include <cassert>
#include <string>
#include <utility>
#include <vector>

class RegressionCondition {
public:
    FactPair data;

    RegressionCondition(int var, int value);
    ~RegressionCondition() = default;

    bool operator<(const RegressionCondition &other) const {
        return data.var < other.data.var
                || (data.var == other.data.var && data.value < other.data.value);
    }

    bool operator==(const RegressionCondition &other) const {
        return data.var == other.data.var && data.value == other.data.value;
    }

    bool operator!=(const RegressionCondition &other) const {
        return !(*this == other);
    }

    bool is_satisfied(const PartialAssignment &assignment) const;
};

class RegressionConditionProxy {
    const AbstractTask *task;
    RegressionCondition condition;
public:
    RegressionConditionProxy(const AbstractTask &task, int var_id, int value);
    RegressionConditionProxy(const AbstractTask &task, const RegressionCondition &fact);
    ~RegressionConditionProxy() = default;

    VariableProxy get_variable() const {
        return VariableProxy(*task, condition.data.var);
    }

    int get_value() const {
        return condition.data.value;
    }

    FactPair get_pair() const {
        return condition.data;
    }

    RegressionCondition get_condition() const {
        return condition;
    }

    std::string get_name() const {
        return task->get_fact_name(condition.data);
    }

    bool operator==(const RegressionConditionProxy &other) const {
        assert(task == other.task);
        return condition == other.condition;
    }

    bool operator!=(const RegressionConditionProxy &other) const {
        return !(*this == other);
    }

    bool is_mutex(const RegressionConditionProxy &other) const {
        return task->are_facts_mutex(condition.data, other.condition.data);
    }
};

class RegressionEffect {
public:
    FactPair data;

    RegressionEffect(int var, int value);
    ~RegressionEffect() = default;

    bool operator<(const RegressionEffect &other) const {
        return data.var < other.data.var
                || (data.var == other.data.var && data.value < other.data.value);
    }

    bool operator==(const RegressionEffect &other) const {
        return data.var == other.data.var && data.value == other.data.value;
    }

    bool operator!=(const RegressionEffect &other) const {
        return !(*this == other);
    }
};

class RegressionEffectProxy {
    const AbstractTask *task;
    RegressionEffect effect;
public:
    RegressionEffectProxy(const AbstractTask &task, int var_id, int value);
    RegressionEffectProxy(const AbstractTask &task, const RegressionEffect &fact);
    ~RegressionEffectProxy() = default;

    VariableProxy get_variable() const {
        return VariableProxy(*task, effect.data.var);
    }

    int get_value() const {
        return effect.data.value;
    }

    FactPair get_pair() const {
        return effect.data;
    }

    RegressionEffect get_effect() const {
        return effect;
    }

    std::string get_name() const {
        return task->get_fact_name(effect.data);
    }

    bool operator==(const RegressionEffectProxy &other) const {
        assert(task == other.task);
        return effect == other.effect;
    }

    bool operator!=(const RegressionEffectProxy &other) const {
        return !(*this == other);
    }

    bool is_mutex(const RegressionEffectProxy &other) const {
        return task->are_facts_mutex(effect.data, other.effect.data);
    }
};

class RegressionOperator {
    std::vector<RegressionCondition> preconditions;
    std::vector<RegressionEffect> effects;
    std::unordered_set<int> original_effect_vars;
    std::string name;
    int cost;

public:
    explicit RegressionOperator(OperatorProxy &op);
    ~RegressionOperator() = default;

    int get_cost() const {
        return cost;
    }

    const std::string &get_name() const {
        return name;
    }

    const std::vector<RegressionCondition> &get_preconditions() const {
        return preconditions;
    }

    const std::vector<RegressionEffect> &get_effects() const {
        return effects;
    }

    bool is_applicable(const PartialAssignment &assignment) const;

};

class RegressionOperatorProxy {
    const AbstractTask *task;
    int index;
    bool is_an_axiom;
public:

    OperatorProxy(const AbstractTask &task, int index, bool is_axiom)
    : task(&task), index(index), is_an_axiom(is_axiom) {
    }
    ~OperatorProxy() = default;

    bool operator==(const OperatorProxy &other) const {
        assert(task == other.task);
        return index == other.index && is_an_axiom == other.is_an_axiom;
    }

    bool operator!=(const OperatorProxy &other) const {
        return !(*this == other);
    }

    PreconditionsProxy get_preconditions() const {
        return PreconditionsProxy(*task, index, is_an_axiom);
    }

    EffectsProxy get_effects() const {
        return EffectsProxy(*task, index, is_an_axiom);
    }

    int get_cost() const {
        return task->get_operator_cost(index, is_an_axiom);
    }

    bool is_axiom() const {
        return is_an_axiom;
    }

    std::string get_name() const {
        return task->get_operator_name(index, is_an_axiom);
    }

    int get_id() const {
        return index;
    }

    OperatorID get_global_operator_id() const {
        assert(!is_an_axiom);
        return task->get_global_operator_id(OperatorID(index));
    }
};

class RegressionTaskProxy {
    const AbstractTask *task;
    const bool possess_mutexes;
    const std::vector<int> domain_sizes;
    const std::vector<RegressionOperator> operators;

public:

    explicit RegressionTaskProxy(const AbstractTask &task);
    ~RegressionTaskProxy() = default;

    VariablesProxy get_variables() const {
        return VariablesProxy(*task);
    }

    std::vector<RegressionOperator> &get_operators() const {
        return operators;
    }

    bool has_mutexes() {
        return possess_mutexes;
    }

    PartialAssignment get_goals() const {
        GoalsProxy gp = GoalsProxy(task);
        std::vector<int> values(gp.size(),
                PartialAssignment::UNASSIGNED);
        for (FactProxy goal : gp) {
            int var_id = goal.get_variable().get_id();
            int value = goal.get_value();
            assert(in_bounds(var_id, values));
            values[var_id] = value;
        }
        return PartialAssignment(*task, std::move(values));
    }

    State get_initial_state() const {
        return State(*task, task->get_initial_state_values());
    }

    std::pair<bool, State> convert_to_full_state(PartialAssignment &assignment,
            bool check_mutexes, utils::RandomNumberGenerator &rng);

    const causal_graph::CausalGraph &get_causal_graph() const;
};


#endif

