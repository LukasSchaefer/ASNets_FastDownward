#ifndef NEURAL_NETWORKS_PROTOBUF_NETWORK_H
#define NEURAL_NETWORKS_PROTOBUF_NETWORK_H

#include "abstract_network.h"

#include "../option_parser.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include <vector>
#include <string>

namespace neural_networks {
class ProtobufNetwork :  AbstractNetwork{
protected:
    const std::shared_ptr<AbstractTask> task;
    TaskProxy task_proxy;
    
    const std::string path;
    const std::string input_layer_name;
    const std::string output_layer_name;
    
    tensorflow::Session* session;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    std::vector<tensorflow::Tensor> outputs;

    virtual void initialize_inputs() = 0;
    virtual void fill_input(const State &state) = 0;
    virtual void extract_output() = 0;
    
    virtual void evaluate(const State& state) override;
    
    

public:
    explicit ProtobufNetwork(const Options &opts);
    ProtobufNetwork(const ProtobufNetwork& orig) = delete;
    virtual ~ProtobufNetwork() = default;
    
    static void add_options_to_parser(options::OptionParser &parser);
private:

};
}
#endif /* NEURAL_NETWORKS_PROTOBUF_NETWORK_H */

