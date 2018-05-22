#ifndef NEURAL_NETWORKS_STATE_GOAL_NETWORK_H
#define NEURAL_NETWORKS_STATE_GOAL_NETWORK_H

#include "protobuf_network.h"

#include "../heuristic.h"
#include "../option_parser.h"

#include <tuple>
#include <vector>

namespace neural_networks {
/**
 * This network takes as input a state and the goal description (the goal
 * description is given by telling for every predicate in the state 1 if it is
 * part of the goal and 0 otherwise).
 * The input state can be larger than the state Fast Downward works on. Via
 * command line arguments the predicates of the state and their default values
 * can be described.
 */
class StateGoalNetwork : public ProtobufNetwork {
private:
    std::vector<int> tmp_goal_idx;
protected:
    const std::vector<int> domain_sizes;
    const std::string tmp_state_input_layer_name;
    const std::string tmp_goal_input_layer_name;
    const std::string tmp_output_layer_name;
    
    const std::vector<std::string> state_atoms;
    const std::vector<int> state_defaults;
    /*Tells for all facts which are used as input of the network their index in
      the input tensor. Every tuple in the list has the three fields
      <VarID, Value, Idx in tensor>. Variables not used in the tensor have no
      tuples.
      The  tuples are sorted by their VarID and then their Value!*/
    const std::vector<std::tuple<int, int, int>> fact_mapping;
    
    const OutputType output_type;
    int last_h = Heuristic::NO_VALUE;
    
    std::vector<std::string> get_atoms_list(std::vector<std::string> atoms);
    std::vector<int> get_atoms_defaults(std::vector<int> defaults);
    std::vector<std::tuple<int, int, int>> get_fact_mapping();
    
public:
    explicit StateGoalNetwork(const Options &opts);
    StateGoalNetwork(const StateGoalNetwork& orig) = delete;
    virtual ~StateGoalNetwork() override = default;
    
    virtual bool is_heuristic() override;
    virtual int get_heuristic() override;
    
    virtual void initialize() override;
    virtual void initialize_inputs() override;
    virtual void initialize_output_layers() override;
    
    virtual void fill_input(const State &state) override;
    virtual void extract_output() override;


private:

};
}
#endif /* NEURAL_NETWORKS_STATE_GOAL_NETWORK_H */

