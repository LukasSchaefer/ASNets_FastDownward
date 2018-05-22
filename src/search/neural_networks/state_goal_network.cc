#include "state_goal_network.h"

#include "../option_parser.h"
#include "../plugin.h"

#include <algorithm>
#include <memory>

using namespace std;
using namespace tensorflow;
namespace neural_networks {

vector<int> get_domain_sizes(const TaskProxy & task_proxy){
    vector<int> domain_sizes;
    domain_sizes.reserve(task_proxy.get_variables().size());
    for(const VariableProxy &variable_proxy : task_proxy.get_variables()) {
        domain_sizes.push_back(variable_proxy.get_domain_size());
    }
    return domain_sizes;
}

//Ask Someone for move command
vector<string> StateGoalNetwork::get_atoms_list(vector<string> atoms) {
    if (atoms.empty()){
        for (VariableProxy vp: task_proxy.get_variables()) {
            for (int idx = 0; idx < vp.get_domain_size(); ++idx) {
                FactProxy fp = vp.get_fact(idx);
                if (fp.get_name() == "<none of those>"
                    || fp.get_name().find("NegatedAtom") == 0) {
                    continue;
                }
                atoms.push_back(fp.get_name());
            }
        }
        sort(atoms.begin(), atoms.end());
    }
    return atoms;
}

vector<int> StateGoalNetwork::get_atoms_defaults(vector<int> defaults) {
    if (defaults.empty()){
        for (unsigned int i = 0; i < state_atoms.size(); ++i){
            defaults.push_back(0);
        }
    } else if (defaults.size() != state_atoms.size()){
        cerr << "The number of specified atoms does not agree with the "
                "number of given default values: " << state_atoms.size()
             << ", " << defaults.size() << endl;
    }
    return defaults;
}

vector<tuple<int, int, int>> StateGoalNetwork::get_fact_mapping(){
    //KEEP FACT_MAPPING SORTED BY VARIABLE, VALUE
    unordered_map<string, int> atom2idx;
    for (unsigned int atom_idx = 0; atom_idx < state_atoms.size(); ++atom_idx) {
        atom2idx[state_atoms[atom_idx]] = atom_idx;
    }
    //HACK defaults for goal input tensor need idx of goal atoms and this is the
    //cheapest way
    unordered_map<int, unordered_set<int>> goals;
    for (const FactProxy &fp: task_proxy.get_goals()){
        FactPair fact = fp.get_pair();
        goals[fact.var].insert(fact.value);
    }
    
    vector<tuple<int, int, int>> tmp_fact_mapping;
    
    for (unsigned int var = 0; var < domain_sizes.size(); ++var) {
        for (int value = 0; value < domain_sizes[var]; ++value) {
            std::string fact_name = task_proxy.get_fact(var, value).get_name();
            unordered_map<string, int>::iterator iter = atom2idx.find(fact_name);
            if (iter != atom2idx.end()) {
                tmp_fact_mapping.push_back(tuple<int, int, int> (var, value, iter->second));
                if (goals.find(var) != goals.end() && goals[var].find(value) != goals[var].end()){
                    tmp_goal_idx.push_back(get<2>(tmp_fact_mapping[tmp_fact_mapping.size() - 1]));
                }
            }
        }
    }
    return tmp_fact_mapping;
}

StateGoalNetwork::StateGoalNetwork(const Options& opts)
    : ProtobufNetwork(opts),
      domain_sizes(get_domain_sizes(task_proxy)),
      tmp_state_input_layer_name(opts.get<string>("state_layer")),
      tmp_goal_input_layer_name(opts.get<string>("goal_layer")),
      tmp_output_layer_name(opts.get<string>("output_layer")),
      state_atoms(get_atoms_list(opts.get_list<string>("atoms"))),
      state_defaults(get_atoms_defaults(opts.get_list<int>("defaults"))),
      fact_mapping(get_fact_mapping()),
      output_type(get_output_type(opts.get<string>("type"))) {
    if (output_type != OutputType::Classification
        && output_type != OutputType::Regression){
        cerr << "Invalid output type for network: " << output_type << endl;
        utils::exit_with(utils::ExitCode::UNSUPPORTED);
    }
    
    cout << "Network State Input: ";
    for (unsigned int i = 0 ; i < state_atoms.size() - 1; ++i){
        cout << state_atoms[i] << "(" << state_defaults[i] << "), ";
    }
    cout << state_atoms[state_atoms.size() - 1] << "(" 
         << state_defaults[state_atoms.size() - 1] << ")" << endl;
    
}

bool StateGoalNetwork::is_heuristic(){
    return true;
}
int StateGoalNetwork::get_heuristic(){
    return last_h;
}

void StateGoalNetwork::initialize() {
    ProtobufNetwork::initialize();
}


void StateGoalNetwork::initialize_inputs() {
    int number_atoms = state_atoms.size();
    Tensor state_tensor(DT_FLOAT, TensorShape({1, number_atoms}));
    Tensor goal_tensor(DT_FLOAT, TensorShape({1, number_atoms}));
    inputs = {
        {tmp_state_input_layer_name, state_tensor},
        {tmp_goal_input_layer_name, goal_tensor}
    };

    // Fill state tensor with default values
    auto t_matrix = inputs[0].second.matrix<float>();
    for (unsigned int idx = 0; idx < state_defaults.size(); ++idx) {
        t_matrix(0, idx) = state_defaults[idx];
    }
    // Fill goal tensor with default values
    t_matrix = inputs[1].second.matrix<float>();
    for (unsigned int idx = 0; idx < state_defaults.size(); ++idx) {
        t_matrix(0, idx) = 0;
    }
    for (int idx : tmp_goal_idx) {
        t_matrix(0, idx) = 1;
    }
}

void StateGoalNetwork::initialize_output_layers() {
    output_layers.push_back(tmp_output_layer_name);
}

void StateGoalNetwork::fill_input(const State& state){
    const vector<int> values = state.get_values();
    auto t_matrix = inputs[0].second.matrix<float>();
    for (const tuple<int, int, int> tup: fact_mapping){
        if (values[get<0>(tup)] == get<1>(tup)) {
            t_matrix(0, get<2>(tup)) = 1;
        } else {
            t_matrix(0, get<2>(tup)) = 0;
        }
    }
}

void StateGoalNetwork::extract_output(){
    auto output_c = outputs[0].flat<float>();
    if (output_type == OutputType::Regression) {
        last_h = round(output_c(0));
    } else if (output_type == OutputType::Classification) {
        int maxIdx = 0;
        float maxVal = output_c(0);
        for (int i = 0; i < output_c.size(); i++) {
            if (output_c(i) > maxVal) {
                maxIdx = i;
                maxVal = output_c(i);
            }
        }
        last_h = maxIdx;
    }
}

}

static shared_ptr<neural_networks::AbstractNetwork> _parse(OptionParser &parser) {
    neural_networks::ProtobufNetwork::add_options_to_parser(parser);
    parser.add_option<string>("type",
        "Type of network output (regression or classification)");
    parser.add_option<string>("state_layer", "Name of the input layer in "
        "the computation graph to insert the current state.");
    parser.add_option<string>("goal_layer", "Name of the input layer in"
        "the computation graph to insert the current goal.");
    parser.add_option<string>("output_layer", "Name of the output layer "
        "from which to extract the network output.");
    parser.add_list_option<string>("atoms", "(Optional) Description of the atoms"
        "in the input state of the network. Provide a list of atom names exactly"
        "as they are used by Fast Downward. The order of the list has to fit to"
        "the order the network expects them.", "[]");
    parser.add_list_option<int>("defaults", "(Optional) If not given, then"
        "all state atoms values are defaulted to 0. If given, then this needs to "
        "be of the same size as \"atoms\" or as the number of not atoms"
        " not pruned by Fast Downward (if only this is given and not \"atoms\"."
        " Provide 1 for atoms present by default and 0 for atoms absent. (The "
        "goal atoms default always to 0)", "[]");
    
    Options opts = parser.parse();
    shared_ptr<neural_networks::StateGoalNetwork> network;
    if (!parser.dry_run()) {       
        network = make_shared<neural_networks::StateGoalNetwork>(opts);
    }

    return network;
}

static PluginShared<neural_networks::AbstractNetwork> _plugin("sgnet", _parse);

