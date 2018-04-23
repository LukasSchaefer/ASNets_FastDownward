#include "problem_network.h"

#include "../option_parser.h"
#include "../plugin.h"

#include <memory>

using namespace std;
using namespace tensorflow;
namespace neural_networks {

vector<int> get_domain_sizes(TaskProxy & task_proxy){
    vector<int> domain_sizes;
    domain_sizes.reserve(task_proxy.get_variables().size());
    for(const VariableProxy &variable_proxy : task_proxy.get_variables()) {
        domain_sizes.push_back(variable_proxy.get_domain_size());
    }
    return domain_sizes;
}

ProblemNetwork::ProblemNetwork(const Options& opts)
    : ProtobufNetwork(opts),
      domain_sizes(get_domain_sizes(task_proxy)),
      output_type(get_output_type(opts.get<string>("type"))) {
    if (output_type != OutputType::Classification
        && output_type != OutputType::Regression){
        cerr << "Invalid output type for network: " << output_type << endl;
        utils::exit_with(utils::ExitCode::UNSUPPORTED);
    }
}

bool ProblemNetwork::is_heuristic(){
    return true;
}
int ProblemNetwork::get_heuristic(){
    return last_h;
}

void ProblemNetwork::initialize_inputs() {
    int input_size = 0;
    for (unsigned int i = 0; i < task_proxy.get_variables().size(); ++i) {
            input_size += task_proxy.get_variables()[i].get_domain_size();
    }
    Tensor tensor(DT_FLOAT, TensorShape({1, input_size}));
    inputs = {{input_layer_name, tensor},};
}


void ProblemNetwork::fill_input(const State& state){
    auto t_matrix = inputs[0].second.matrix<float>();
    int idx = 0;
    for (unsigned int i = 0; i < domain_sizes.size(); i++) {
        for (int j = 0; j < domain_sizes[i]; j++) {
            if (j == state[i]) {
                t_matrix(0, idx) = 1;
            } else {

                t_matrix(0, idx) = 0;
            }
            idx++;
        }
    }
}

void ProblemNetwork::extract_output(){
    auto output_c = outputs[0].flat<float>();
    if (output_type == OutputType::Regression) {
        last_h = round(output_c(0));
    } else (output_type == OutputType::Classification) {
        int maxIdx = 0;
        float maxVal = output_c(0);
        for (int i = 0; i < output_c.size(); i++) {
            //std::cout << i << "\t" << output_c(i) << std::endl;
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
    neural_networks::ProtobufNetwork::add_options_to_parser(parser)
    parser.add_option<string>("type",
        "Type of network output (regression or classification)");
    Options opts = parser.parse();

    shared_ptr<neural_networks::ProblemNetwork> network;
    if (!parser.dry_run()) {       
        network = make_shared<neural_networks::ProblemNetwork>(opts);
    }

    return network;
}

static PluginShared<neural_networks::AbstractNetwork> _plugin("probnet", _parse);
}
