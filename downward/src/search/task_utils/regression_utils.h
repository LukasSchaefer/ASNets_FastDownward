#ifndef TASK_UTILS_REGRESSION_UTILS_H
#define TASK_UTILS_REGRESSION_UTILS_H

#include "../abstract_task.h"
#include "../global_state.h"
#include "../operator_id.h"
#include "../task_proxy.h"

#include "../utils/rng.h"

#include <cassert>
#include <set>
#include <string>
#include <utility>
#include <vector>

/* Special thanks to Jendrik Seipp from whom I got an initial code base for
   regression.*/

class RegressionCondition {
    friend class RegressionConditionProxy;
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
    const RegressionCondition condition;
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

class RegressionConditionsProxy : public ConditionsProxy {
    const std::vector<RegressionCondition> conditions;

public:
    using ItemType = RegressionConditionProxy;

    RegressionConditionsProxy(const AbstractTask &task,
            const std::vector<RegressionCondition> & conditions)
    : ConditionsProxy(task), conditions(conditions) {
    }
    ~RegressionConditionsProxy() = default;

    std::size_t size() const override {
        return conditions.size();
    }

    bool empty() const {
        return size() == 0;
    }

    FactProxy operator[](std::size_t fact_index) const override {
        assert(fact_index < size());
        return FactProxy(*task, conditions[fact_index].data);
    }

    RegressionCondition get(std::size_t fact_index) const {
        return conditions[fact_index];
    }
};

class RegressionEffect {
    friend class RegressionEffectProxy;
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

class RegressionEffectsProxy : public ConditionsProxy {
    const std::vector<RegressionEffect> effects;

public:
    using ItemType = RegressionEffectProxy;

    RegressionEffectsProxy(const AbstractTask &task,
            const std::vector<RegressionEffect> & effects)
    : ConditionsProxy(task), effects(effects) {
    }
    ~RegressionEffectsProxy() = default;

    std::size_t size() const override {
        return effects.size();
    }

    bool empty() const {
        return size() == 0;
    }

    FactProxy operator[](std::size_t fact_index) const override {
        assert(fact_index < size());
        return FactProxy(*task, effects[fact_index].data);
    }

    RegressionEffect get(std::size_t fact_index) const {
        return effects[fact_index];
    }
};

class RegressionOperator {
    friend class RegressionOperatorProxy;
    const int original_index;
    const int cost;
    const std::string name;
    const bool is_an_axiom;

    std::vector<RegressionCondition> preconditions;
    std::vector<RegressionEffect> effects;
    std::set<int> original_effect_vars;


public:
    explicit RegressionOperator(OperatorProxy &op);
    ~RegressionOperator() = default;

    bool operator==(const RegressionOperator &other) const {
        return preconditions == other.preconditions
                && effects == other.effects
                && original_effect_vars == other.original_effect_vars
                && name == other.name
                && cost == other.cost
                && is_an_axiom == other.is_an_axiom;
    }

    bool operator!=(const RegressionOperator &other) const {
        return !(*this == other);
    }

    int get_original_index() const {
        return original_index;
    }
    
    int get_cost() const {
        return cost;
    }

    bool is_axiom() const {
        return is_an_axiom;
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

    const std::set<int> get_original_effect_vars() const {
        return original_effect_vars;
    }

    bool is_applicable(const PartialAssignment &assignment) const;

};

class RegressionOperatorProxy {
    const AbstractTask *task;
    const RegressionOperator op;
    bool is_an_axiom;
public:

    RegressionOperatorProxy(const AbstractTask &task, const RegressionOperator &op)
    : task(&task), op(op), is_an_axiom(false) {
    }
    ~RegressionOperatorProxy() = default;

    bool operator==(const RegressionOperatorProxy &other) const {
        assert(task == other.task);
        return op == other.op && is_an_axiom == other.is_an_axiom;
    }

    bool operator!=(const RegressionOperatorProxy &other) const {
        return !(*this == other);
    }

    RegressionConditionsProxy get_preconditions() const {
        return RegressionConditionsProxy(*task, op.preconditions);
    }

    RegressionEffectsProxy get_effects() const {
        return RegressionEffectsProxy(*task, op.effects);
    }

    std::set<int> get_original_effect_vars() const {
        return op.original_effect_vars;
    }

    int get_cost() const {
        return op.cost;
    }

    bool is_axiom() const {
        return is_an_axiom;
    }

    int get_id() const {
        return op.original_index;
    }

    std::string get_name() const {
        std::cout<<"ID"<<get_id() <<" A"<<is_an_axiom<<std::endl;
        std::cout << op.get_name() << std::endl;
        std::cout<< task->get_num_operators() << std::endl;
        std::cout<<"ASDF"<<std::endl;
        return "Regression" + task->get_operator_name(get_id(), op.is_an_axiom);
    }

    OperatorID get_global_operator_id() const {
        assert(!is_an_axiom);
        return task->get_global_operator_id(OperatorID(get_id()));
    }

    bool is_applicable(const PartialAssignment &assignment) const {
        return op.is_applicable(assignment);
    }
};

class RegressionOperatorsProxy {
    const AbstractTask *task;
    const std::vector<RegressionOperator> ops;

public:
    using ItemType = RegressionOperatorProxy;

    RegressionOperatorsProxy(const AbstractTask &task,
            const std::vector<RegressionOperator> & ops)
    : task(&task), ops(ops) {
    }
    ~RegressionOperatorsProxy() = default;

    std::size_t size() const {
        return ops.size();
    }

    bool empty() const {
        return size() == 0;
    }

    RegressionOperatorProxy operator[](std::size_t op_index) const {
        assert(op_index < size());
        return RegressionOperatorProxy(*task, ops[op_index]);
    }
};

class RegressionTaskProxy : public TaskProxy {
    const bool possess_mutexes;
    const std::vector<int> domain_sizes;
    const std::vector<RegressionOperator> operators;

public:

    explicit RegressionTaskProxy(const AbstractTask &task);
    ~RegressionTaskProxy() = default;

    RegressionOperatorsProxy get_regression_operators() const {
        return RegressionOperatorsProxy(*task, operators);
    }

    bool has_mutexes() {
        return possess_mutexes;
    }

    int test() const {
        return task->get_num_operators();
    }
    PartialAssignment get_goal_assignment() const {
        GoalsProxy gp = GoalsProxy(*task);
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

    std::pair<bool, State> convert_to_full_state(PartialAssignment &assignment,
            bool check_mutexes, utils::RandomNumberGenerator &rng) const;

    PartialAssignment create_partial_assignment(std::vector<int> &&values) const {
        return PartialAssignment(*task, std::move(values));
    }
    const causal_graph::CausalGraph &get_causal_graph() const;
};


#endif

