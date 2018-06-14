#ifndef NEURAL_NETWORKS_ABSTRACT_NETWORK_H
#define NEURAL_NETWORKS_ABSTRACT_NETWORK_H

#include "../task_proxy.h"

#include "../algorithms/ordered_set.h"
#include "../utils/system.h"

#include <tuple>
#include <vector>

namespace neural_networks {

enum OutputType {
    /*Different types of basic output a network can produce
     (not how we later interpret the output*/
    Classification, Regression, Undefined
};

/*Get for a given string the output type it is associated with*/
extern OutputType get_output_type(std::string type);
    
/* Base class for all networks.
 * add new is_PROPERTY, verify_PROPERTY, get_PROPERTY if adding new properties
 * a network can produce as output (those are now interpreted outputs types like
 * heuristic values or preferred operators)
   */
class AbstractNetwork {
public:
    AbstractNetwork() = default;
    AbstractNetwork(const AbstractNetwork& orig) = delete;
    virtual ~AbstractNetwork() = default;
    
    /*initialize network. Called AFTER construction and before using the network.*/
    virtual void initialize() {};
    /*Performs a single evaluation on the given state*/
    virtual void evaluate(const State & state) = 0;
    
    /*Tells if concrete network produces such an output*/
    virtual bool is_heuristic();
    virtual bool is_policy();
    virtual bool is_preferred();

    // tells if dead_end detection by network policy/ heuristic is reliable
    virtual bool dead_ends_are_reliable();
    
    /*Checks if network produce such an output and stops execution if not.*/
    void verify_heuristic();
    void verify_policy();
    void verify_preferred();
    
    /*Gets the output of the associated type from the networks last evaluation.
     (e.g. heuristic value) */
    virtual int get_heuristic();
    /*policy output consists of operators as ids and corresponding
      preferences (= probabilities) as floats */
    virtual std::pair<std::vector<OperatorID>, std::vector<float>> get_policy();
    virtual ordered_set::OrderedSet<OperatorID>& get_preferred();
};
}

#endif /* NEURAL_NETWORKS_NEURAL_NETWORK_H */

