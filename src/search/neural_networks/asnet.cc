#include "asnet.h"

#include "../plugin.h"
#include "../task_utils/regression_utils.h"
#include "../task_utils/successor_generator.h"


using namespace std;
using namespace tensorflow;

namespace neural_networks {

ASNet::ASNet(const Options& opts)
    : ProtobufNetwork(opts),
      extra_input_size(opts.get<int>("extra_input_size")),
      facts_sorted(lexicographical_access::get_facts_lexicographically(task_proxy)),
      operator_indeces_sorted(lexicographical_access::get_operator_indeces_lexicographically(task_proxy)) {
}


bool ASNet::is_policy() {
    return true;
}

bool ASNet::dead_ends_are_reliable() {
    return true;
}

PolicyResult ASNet::get_policy() {
    return last_policy_output;
}

void ASNet::initialize() {
    ProtobufNetwork::initialize();
}


void ASNet::initialize_inputs() {
    unsigned int num_prop = facts_sorted.size();
    Tensor prop_truth_tensor(DT_FLOAT, TensorShape({1, num_prop}));
    Tensor prop_goal_tensor(DT_FLOAT, TensorShape({1, num_prop}));
    unsigned int num_act = operator_indeces_sorted.size();
    Tensor act_applic_tensor(DT_FLOAT, TensorShape({1, num_act}));
    if (extra_input_size != 0) {
        Tensor extra_inputs(DT_FLOAT, TensorShape({1, extra_input_size * num_act}));
        inputs = {{"input_prop_truth_values", prop_truth_tensor}, {"input_prop_goal_values", prop_goal_tensor},
            {"input_act_applic_values", act_applic_tensor}, {"additional_input_features", extra_inputs}};
    } else {
        inputs = {{"input_prop_truth_values", prop_truth_tensor}, {"input_prop_goal_values", prop_goal_tensor},
            {"input_act_applic_values", act_applic_tensor}};
    }
}

void ASNet::initialize_output_layers() {
    output_layers.push_back("softmax_output_layer_1/truediv");
}


void ASNet::fill_input(const State& state){
    const std::vector<int> values = state.get_values();
    auto prop_truth_values = inputs[0].second.matrix<float>();
    auto prop_goal_values = inputs[1].second.matrix<float>();
    int idx = 0;
    for (std::pair<int, int> fact_pair : facts_sorted) {
        // check truth value of proposition/ fact and set value accordingly
        if (fact_pair.second == values[fact_pair.first]) {
            prop_truth_values(0, idx) = 1;
        } else {
            prop_truth_values(0, idx) = 0;
        }

        // check if proposition is a goal and set value accordingly
        if (std::find(g_goal.begin(), g_goal.end(), fact_pair) != g_goal.end()) {
            prop_goal_values(0, idx) = 1;
        } else {
            prop_goal_values(0, idx) = 0;
        }
        idx++;
    }

    // set action applicable values
    auto act_applic_values = inputs[2].second.matrix<float>();
    const auto operators = task_proxy.get_operators();

    // get all applicable actions
    vector<OperatorID> applicable_ops;
    g_successor_generator->generate_applicable_ops(state, applicable_ops);
    idx = 0;
    for (int operator_index : operator_indeces_sorted) {
        // check if operator is applicable
        OperatorProxy op_proxy = operators[operator_index];
        if (find(applicable_ops.begin(), applicable_ops.end(), op_proxy.get_global_operator_id()) != applicable_ops.end()) {
            act_applic_values(0, idx) = 1;
        } else {
            act_applic_values(0, idx) = 0;
        }
        idx++;
    }

    /*
    if (extra_input_size) {
        auto additional_input_features = inputs[3].second.matrix<float>();
        // TODO: Implement extra input features later on
    }
    */
}

void ASNet::extract_output() {
    auto output_c = outputs[0].flat<float>();
    std::vector<float> operator_preferences(output_c.size());
    // one output probability for each action
    assert(output_c.size() == (int) operator_indeces_sorted.size());
    float sum = 0;
    // operator preferences in sorted order
    for (unsigned index = 0; index < operator_indeces_sorted.size(); index++) {
	float val = output_c(index);
	operator_preferences[index] = val;
	sum += val;
    }
    /* should be a policy with probabilities summing up to 1.0
       potentially small rounding errors (Should be okay?) */
    assert(std::abs(sum - 1.0) < 0.01 && "Policy output probabilities sum was > 0.01 away from 1.0!");

    // match operator IDs to sorted order
    auto operators = task_proxy.get_operators();
    std::vector<OperatorID> operator_ids;
    for (unsigned int sorted_op_index = 0; sorted_op_index < operators.size(); sorted_op_index++) {
	int original_op_index = operator_indeces_sorted[sorted_op_index];
	OperatorID op_id = operators[original_op_index].get_global_operator_id();
        operator_ids.push_back(op_id);
    }

    last_policy_output = std::pair<std::vector<OperatorID>, std::vector<float>>(operator_ids, operator_preferences);
}

}

static shared_ptr<neural_networks::AbstractNetwork> _parse(OptionParser &parser) {
    neural_networks::ProtobufNetwork::add_options_to_parser(parser);
    parser.add_option<int>("extra_input_size",
        "extra input size (per action) of optional additional input features", "0");
    Options opts = parser.parse();

    shared_ptr<neural_networks::ASNet> asnet;
    if (!parser.dry_run()) {       
        asnet = make_shared<neural_networks::ASNet>(opts);
    }

    return asnet;
}

static PluginShared<neural_networks::AbstractNetwork> _plugin("asnet", _parse);
