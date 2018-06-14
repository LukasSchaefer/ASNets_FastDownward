#include "asnet.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_utils/regression_utils.h"

#include <memory>
#include <cmath>

using namespace std;
using namespace tensorflow;

namespace neural_networks {

/* sorts facts lexicographically and returns vector with tuple entries of
   (variable_index, value_index (in variable domain)) */
std::vector<std::pair<int, int>> ASNet::get_facts_lexicographically() {
    // compute necessary vector size (sum of all variable domain sizes)
    unsigned int num_facts = 0;
    for (int dom_size : g_variable_domain) {
        num_facts += dom_size;
    }

    // fill vector of triples consisting of (fact_name, variable_index, value_index)
    std::vector<std::tuple<std::string, int, int>> facts_with_names(num_facts);
    for (unsigned int var_index = 0; var_index < task_proxy.get_variables().size(); var_index++) {
        for (unsigned int val_index = 0; val_index < g_fact_names[var_index].size(); val_index++) {
            std::string fact_name = g_fact_names[var_index][val_index];
            std::tuple<std::string, int, int> fact_triple = std::make_tuple(fact_name, var_index, val_index);
            facts_with_names.push_back(fact_triple);
        }
    }

    // sort vector of triples (by first element -> lexicographically by fact_name)
    sort(facts_with_names.begin(), facts_with_names.end());

    // setup new vector where first element is dropped
    std::vector<std::pair<int, int>> facts_without_names(facts_with_names.size());
    for (unsigned int triple_index = 0; triple_index < facts_with_names.size(); triple_index++) {
        std::tuple<std::string, int, int> triple = facts_with_names[triple_index];
        // take only 2nd and 3rd value (variable_index, value_index)
        facts_without_names[triple_index] = std::pair<int, int>(std::get<1>(triple), std::get<2>(triple));
    }
    return facts_without_names;
}

/* sorts operators lexicographically and returns vector of operator indeces in lexicographical ordering
   of their names */
std::vector<int> ASNet::get_operator_indeces_lexicographically() {
    // fill vector of pairs consisting of (operator_name, operator_index)
    std::vector<std::pair<std::string, int>> operators_with_names(task_proxy.get_operators().size());
    auto operators = task_proxy.get_operators();
    for (unsigned int op_index = 0; op_index < operators.size(); op_index++) {
        auto op = operators[op_index];
        std::pair<std::string, int> op_pair = std::pair<std::string, int>(op.get_name(), op_index);
        operators_with_names[op_index] = op_pair;
    }
    // sort vector of pairs (by first element -> lexicographically by operator_name)
    sort(operators_with_names.begin(), operators_with_names.end());

    // setup new vector with only op_indeces
    std::vector<int> operator_indeces(operators_with_names.size());
    for (unsigned int pair_index = 0; pair_index < operators_with_names.size(); pair_index++) {
        // take only index (2nd) value
        operator_indeces[pair_index] = operators_with_names[pair_index].second;
    }
    return operator_indeces;
}

ASNet::ASNet(const Options& opts)
    : ProtobufNetwork(opts),
      extra_input_size(opts.get<int>("extra_input_size")),
      facts_sorted(get_facts_lexicographically()),
      operator_indeces_sorted(get_operator_indeces_lexicographically()) {
}


bool ASNet::is_policy() {
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
    output_layers.push_back("softmax_div_output");
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
    idx = 0;
    for (int operator_index : operator_indeces_sorted) {
        // check if operator is applicable
 	OperatorProxy op_proxy = operators[operator_index];
	RegressionOperator op = RegressionOperator(op_proxy);
        if (op.is_applicable(state)) {
            act_applic_values(0, idx) = 1;
        } else {
            act_applic_values(0, idx) = 0;
        }
        idx++;
    }

    if (extra_input_size) {
        auto additional_input_features = inputs[3].second.matrix<float>();
        // TODO: Implement extra input features later on
    }
}

void ASNet::extract_output() {
    auto output_c = outputs[0].flat<float>();
    std::vector<float> operator_preferences(output_c.size());
    // one output probability for each action
    assert(output_c.size() == operator_indeces_sorted.size());
    float sum = 0;
    for (unsigned index = 0; index < output_c.size(); index++) {
        float val = output_c(index);
        operator_preferences[index] = val;
        sum += val;
    }
    /* should be a policy with probabilities summing up to 1.0
       potentially small rounding errors (Should be okay?) */
    assert(std::abs(sum - 1.0) < 0.01 && "Policy output probabilities sum was > 0.01 away from 1.0!");

    /* match operator indeces as in g_operators to OperatorIDs in order of opeartor_indeces_sorted
       (output probabilities will be in the same order) */
    auto operators = task_proxy.get_operators();
    std::vector<OperatorID> operator_ids;
    int idx = 0;
    for (int operator_index : operator_indeces_sorted) {
        operator_ids.push_back(operators[operator_index].get_global_operator_id());
        idx++;
    }

    last_policy_output = std::pair<std::vector<OperatorID>, std::vector<float>>(operator_ids, operator_preferences);
}

}

static shared_ptr<neural_networks::AbstractNetwork> _parse(OptionParser &parser) {
    neural_networks::ProtobufNetwork::add_options_to_parser(parser);
    parser.add_option<int>("extra_input_size",
        "extra input size (per action) of optional additional input features", 0);
    Options opts = parser.parse();

    shared_ptr<neural_networks::ASNet> asnet;
    if (!parser.dry_run()) {       
        asnet = make_shared<neural_networks::ASNet>(opts);
    }

    return asnet;
}

static PluginShared<neural_networks::AbstractNetwork> _plugin("asnet", _parse);
