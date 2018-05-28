#ifndef NEURAL_NETWORKS_PROTOBUF_NETWORK_H
#define NEURAL_NETWORKS_PROTOBUF_NETWORK_H

#include "abstract_network.h"

#include "../option_parser.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include <vector>
#include <string>

namespace neural_networks {
/*Base class for all networks using tensorflow and are loaded from a protobuf file.*/
class ProtobufNetwork :  public AbstractNetwork{ 
protected:
    /*Task for which to use the network*/
    const std::shared_ptr<AbstractTask> task;
    TaskProxy task_proxy;
    
    /*Path to the trained network file*/
    const std::string path;
    /*Tensorflow session which is used to perform the evaluation within*/
    tensorflow::Session* session;
    /*At each evaluation fill in here the input. This variable is feed to the network*/
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    /*Names of the variables in the computation graph from which to extract the output*/
    std::vector<std::string> output_layers;
    /*After each evaluation the network output is stored here waiting for further processing.*/
    std::vector<tensorflow::Tensor> outputs;
    
    virtual void initialize_inputs() = 0;
    virtual void initialize_output_layers() = 0;
    
    virtual void fill_input(const State &state) = 0;
    virtual void extract_output() = 0;
    
    virtual void initialize() override;
    virtual void evaluate(const State& state) override;

public:
    explicit ProtobufNetwork(const Options &opts);
    ProtobufNetwork(const ProtobufNetwork& orig) = delete;
    virtual ~ProtobufNetwork() override;
    
    static void add_options_to_parser(options::OptionParser &parser);
private:

};
}
#endif /* NEURAL_NETWORKS_PROTOBUF_NETWORK_H */

