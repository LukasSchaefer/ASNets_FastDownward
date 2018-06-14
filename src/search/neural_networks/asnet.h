#ifndef NEURAL_NETWORKS_ASNET_H
#define NEURAL_NETWORKS_ASNET_H

#include "protobuf_network.h"

#include "../policy.h"
#include "../option_parser.h"

#include <tuple>

namespace neural_networks {
/*Action Schema Network (https://arxiv.org/abs/1709.04271) class. This class takes an already
 trained computation graph and loads it. At every state, it feeds a specific input representation
 of the current state into the network and extract a policy output for the state.
 
 Input representation:
    input_prop_truth_values: binary value (0/1) indicating whether proposition is currently
                              true in the state
    input_prop_goal_values:  binary value (0/1) indicating whether proposition is part
                              of the goal
    input_act_applic_values: binary value (0/1) indicating whether action is currently
                              applicable
    (additional_input_features): optional additional input features per action. LM-Cut
                                 landmark-values were proposed in the original paper

    abbreviations:
        num_prop = number of propositions/ grounded predicates
        num_act = number of actions

    parts: proposition truth values | proposition goal values | action applicable values (| addition input features)
    size:         num_prop          |       num_prop          |        num_act           (| extra_input_size * num_act)

  Outputs:
    policy = probability distribution over all actions to choose a given action with
             one probability for every action in the same ordering as input values (lexicographically by action names)

  IMPORTANT: The order of the binary values for propositions and actions matter!
             Underlying ordering must match the one from network_models/asnets/problem_meta.py for 
             propositional_actions and grounded_predicates (= propositions) which is lexicographical
             ordering of their names
 */
class ASNet : public ProtobufNetwork {
    virtual std::vector<std::pair<int, int>> get_facts_lexicographically();
    virtual std::vector<int> get_operator_indeces_lexicographically();
protected:
    const int extra_input_size = 0;
    PolicyResult last_policy_output = std::pair<std::vector<OperatorID>, std::vector<float>>(std::vector<OperatorID>(), std::vector<float>());
    /* vector of entries of form (variable_index, value_index) for each fact in lexicographical ordering
       of their names */
    std::vector<std::pair<int, int>> facts_sorted;
    /* vector of operator indeces sorted by the corresponding operator names */
    std::vector<int> operator_indeces_sorted;
    
public:
    explicit ASNet(const Options &opts);
    ASNet(const ASNet& orig) = delete;
    virtual ~ASNet() override = default;
    
    virtual bool is_policy() override;
    virtual bool dead_ends_are_reliable() override;
    virtual PolicyResult get_policy() override;
    
    virtual void initialize() override;
    virtual void initialize_inputs() override;
    virtual void initialize_output_layers() override;
    
    virtual void fill_input(const State &state) override;
    virtual void extract_output() override;


private:

};
}
#endif /* NEURAL_NETWORKS_PROBLEM_NETWORK_H */
