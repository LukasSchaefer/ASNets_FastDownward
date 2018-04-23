#include "abstract_network.h"

#include "../heuristic.h"
#include "../utils/system.h"

#include <iostream>

using namespace std;
namespace neural_networks {

bool AbstractNetwork::is_heuristic(){
    return false;
}
bool AbstractNetwork::is_preferred(){
    return false;
}
void AbstractNetwork::verify_heuristic(){
    if (!is_heuristic()){
        cerr << "Network does not support heuristic estimates." << endl
         << "Terminating." <<endl; 
    utils::exit_with(utils::ExitCode::UNSUPPORTED);
    }
}
void AbstractNetwork::verify_preferred(){
    if (!is_preferred()){
        cerr << "Network does not support preferred operator estimates." << endl
         << "Terminating." <<endl; 
    utils::exit_with(utils::ExitCode::UNSUPPORTED);
    }
}
int AbstractNetwork::get_heuristic(){
    cerr << "Network does not support heuristic estimates." << endl
         << "Terminating." <<endl; 
    utils::exit_with(utils::ExitCode::UNSUPPORTED);
}
ordered_set::OrderedSet<OperatorID>& AbstractNetwork::get_preferred(){
    cerr << "Network does not support preferred operator estimates." << endl
         << "Terminating." <<endl; 
    utils::exit_with(utils::ExitCode::UNSUPPORTED);
}
}