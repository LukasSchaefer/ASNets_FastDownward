#ifndef NEURAL_NETWORKS_ABSTRACT_NETWORK_H
#define NEURAL_NETWORKS_ABSTRACT_NETWORK_H

#include "../task_proxy.h"

#include "../algorithms/ordered_set.h"
#include "../utils/system.h"

namespace neural_networks {

enum OutputType {
    Classification, Regression, Undefined
};
OutputType get_output_type(std::string type) {
    if (type == "regression"){
        return OutputType::Regression;
    } else if (type == "classification") {
        return OutputType::Classification;
    } else {
        std::cerr << "Invalid network output type: " << type << std::endl;
        utils::exit_with(utils::ExitCode::UNSUPPORTED);
    }
}
    
class AbstractNetwork {
public:
    AbstractNetwork() = default;
    AbstractNetwork(const AbstractNetwork& orig) = delete;
    virtual ~AbstractNetwork() = default;
    
    virtual void evaluate(const State & state) = 0;
    
    virtual bool is_heuristic();
    virtual bool is_preferred();
    
    void verify_heuristic();
    void verify_preferred();
    
    virtual int get_heuristic();
    virtual ordered_set::OrderedSet<OperatorID>& get_preferred();
};
}

#endif /* NEURAL_NETWORKS_NEURAL_NETWORK_H */

