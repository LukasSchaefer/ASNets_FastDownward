#ifndef SEARCH_SPACE_H
#define SEARCH_SPACE_H

#include "global_state.h"
#include "operator_cost.h"
#include "per_state_information.h"
#include "search_node_info.h"
#include "state_id.h"

#include <vector>

class GlobalState;
class OperatorProxy;
class TaskProxy;


class SearchNode {
    const StateRegistry &state_registry;
    StateID state_id;
    SearchNodeInfo &info;
    OperatorCost cost_type;
public:
    SearchNode(const StateRegistry &state_registry,
               StateID state_id,
               SearchNodeInfo &info,
               OperatorCost cost_type);

    StateID get_state_id() const {
        return state_id;
    }
    
    StateID get_parent_id() const {
        return info.parent_state_id;
    }
    
    OperatorID get_creating_operator() const {
        return info.creating_operator;
    }
    
    GlobalState get_state() const;

    bool is_new() const;
    bool is_open() const;
    bool is_closed() const;
    bool is_dead_end() const;

    int get_g() const;
    int get_real_g() const;

    void open_initial();
    void open(const SearchNode &parent_node,
              const OperatorProxy &parent_op);
    void reopen(const SearchNode &parent_node,
                const OperatorProxy &parent_op);
    void update_parent(const SearchNode &parent_node,
                       const OperatorProxy &parent_op);
    void close();
    void mark_as_dead_end();

    void dump(const TaskProxy &task_proxy) const;
};


class SearchSpace {
    PerStateInformation<SearchNodeInfo> search_node_infos;

    StateRegistry &state_registry;
    OperatorCost cost_type;
public:
    SearchSpace(StateRegistry &state_registry, OperatorCost cost_type);

    SearchNode get_node(const GlobalState &state);
    
    const SearchNode get_node(const GlobalState &state) const;
    
    StateID get_parent_id(const GlobalState &state) const {
        return search_node_infos[state].parent_state_id;
    }
    
    OperatorID get_creating_operator(const GlobalState &state) const {
        return search_node_infos[state].creating_operator;
    }
    
    void trace_path(const GlobalState &goal_state,
                    std::vector<OperatorID> &path) const;
    void trace_path(const GlobalState &goal_state,
                    std::vector<StateID> &trajectory) const;
    void trace_path(const GlobalState &goal_state,
                    std::vector<OperatorID> &path,
                    std::vector<StateID> &trajectory) const;

    void dump(const TaskProxy &task_proxy) const;
    void print_statistics() const;
};

#endif
