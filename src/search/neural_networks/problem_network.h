#ifndef NEURAL_NETWORKS_PROBLEM_NETWORK_H
#define NEURAL_NETWORKS_PROBLEM_NETWORK_H

#include "protobuf_network.h"

#include "../heuristic.h"
#include "../option_parser.h"

namespace neural_networks {
class ProblemNetwork : public ProtobufNetwork {
protected:
    const std::vector<int> domain_sizes;
    const OutputType output_type;
    int last_h = Heuristic::NO_VALUE;
    
public:
    explicit ProblemNetwork(const Options &opts);
    ProblemNetwork(const ProblemNetwork& orig) = delete;
    virtual ~ProblemNetwork() override = default;
    
    virtual bool is_heuristic() override;
    virtual int get_heuristic() override;
    
    virtual void initialize() override;
    virtual void initialize_inputs() override;
    virtual void fill_input(const State &state) override;
    virtual void extract_output() override;


private:

};
}
#endif /* NEURAL_NETWORKS_PROBLEM_NETWORK_H */

