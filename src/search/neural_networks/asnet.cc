#include "asnet.h"

#include "../plugin.h"
#include "../task_utils/regression_utils.h"
#include "../task_utils/successor_generator.h"


using namespace std;
using namespace tensorflow;

namespace neural_networks {

vector<int> reverse_operator_sorted_vec(vector<int> operator_indeces_sorted) {
    vector<int> operator_indeces_sorted_reversed(operator_indeces_sorted.size());
    for (unsigned int sorted_index = 0; sorted_index < operator_indeces_sorted.size(); sorted_index++) {
        int unsorted_index = operator_indeces_sorted[sorted_index];
        operator_indeces_sorted_reversed[unsorted_index] = sorted_index;
    }
    return operator_indeces_sorted_reversed;
}

ASNet::ASNet(const Options& opts)
    : ProtobufNetwork(opts),
      additional_input_features(opts.get<string>("additional_input_features")),
      facts_sorted(lexicographical_access::get_facts_lexicographically(task_proxy)),
      operator_indeces_sorted(lexicographical_access::get_operator_indeces_lexicographically(task_proxy)),
      operator_indeces_sorted_reversed(reverse_operator_sorted_vec(operator_indeces_sorted)) {
    if (additional_input_features == "landmarks" || additional_input_features == "binary_landmarks") {
        landmark_generator = utils::make_unique_ptr<lm_cut_heuristic::LandmarkCutLandmarks>(task_proxy);
    }
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
    if (additional_input_features != "none") {
        int extra_input_size = 0;
        if (additional_input_features == "landmarks") {
            extra_input_size = 2;
        } else if (additional_input_features == "binary_landmarks") {
            extra_input_size = 3;
        }
        Tensor extra_inputs(DT_FLOAT, TensorShape({1, extra_input_size * num_act}));
        inputs = {{"input_prop_truth_values", prop_truth_tensor}, {"input_prop_goal_values", prop_goal_tensor},
            {"input_act_applic_values", act_applic_tensor}, {"additional_input_features", extra_inputs}};
    } else {
        inputs = {{"input_prop_truth_values", prop_truth_tensor}, {"input_prop_goal_values", prop_goal_tensor},
            {"input_act_applic_values", act_applic_tensor}};
    }
}

void ASNet::initialize_output_layers() {
    output_layers.push_back("softmax_output_layer/truediv");
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

    // additional input features
    if (additional_input_features != "none") {
        auto additional_input_tensor = inputs[3].second.matrix<float>();

        if (additional_input_features == "landmarks") {
            vector< vector<int> > cuts_ids;
            // compute LM-cut landmarks and collect cut op ids
            landmark_generator->compute_landmarks(
                state,
                nullptr,
                [&cuts_ids](vector<int> cut, int /*cut_cost*/) {cuts_ids.push_back(cut); });

            for (vector<int> cut : cuts_ids) {
                bool single_element = false;
                if (cut.size() == 1) {
                    single_element = true;
                }
                for (int unsorted_op_index : cut) {
                    int sorted_index = operator_indeces_sorted_reversed[unsorted_op_index];
                    // op was contained in contain -> increment counter
                    additional_input_tensor(0, 2 * sorted_index) += 1;
                    if (single_element) {
                        // op was only action in the cut -> increment 2nd counter
                        additional_input_tensor(0, 2 * sorted_index + 1) += 1;
                    }
                }
            }
        } else if (additional_input_features == "binary_landmarks") {
            vector< vector<int> > cuts_ids;
            // compute LM-cut landmarks and collect cut op ids
            landmark_generator->compute_landmarks(
                state,
                nullptr,
                [&cuts_ids](vector<int> cut, int /*cut_cost*/) {cuts_ids.push_back(cut); });

            unsigned long number_of_operators = operator_indeces_sorted.size();
            for (unsigned int op_id = 0; op_id < number_of_operators; op_id++) {
                /*
                * initialize third value of each action with 1 (at first it appeared
                * in no LM yet)
                */
                additional_input_tensor(0, op_id * 3 + 2) = 1;
            }

            for (vector<int> cut : cuts_ids) {
                bool single_element = false;
                if (cut.size() == 1) {
                    single_element = true;
                }
                bool two_or_more_elements = false;
                if (cut.size() >= 2) {
                    two_or_more_elements = true;
                }
                for (int unsorted_op_index : cut) {
                    int sorted_index = operator_indeces_sorted_reversed[unsorted_op_index];
                    if (single_element) {
                        // op was only action in the cut -> set first binary value
                        additional_input_tensor(0, 3 * sorted_index) = 1;
                    }
                    if (two_or_more_elements) {
                        // op included in a cut with at least 2 actions
                        additional_input_tensor(0, 3 * sorted_index + 1) = 1;
                    }
                    // op was contained in a cut
                    additional_input_tensor(0, 3 * sorted_index + 2) = 0;
                }
            }
        }
    }
}

void ASNet::extract_output() {
    auto output_c = outputs[0].flat<float>();
    std::vector<float> operator_preferences(output_c.size());
    // one output probability for each action
    assert(output_c.size() == (int) operator_indeces_sorted.size());
    
    // operator preferences in sorted order
    for (unsigned index = 0; index < operator_indeces_sorted.size(); index++) {
	operator_preferences[index] = output_c(index);
    }

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
    parser.add_option<std::string> ("additional_input_features",
        "Name of additional input features to be used for the network. These "
        "will be extracted and added to the samples", "none");
    Options opts = parser.parse();

    shared_ptr<neural_networks::ASNet> asnet;
    if (!parser.dry_run()) {       
        asnet = make_shared<neural_networks::ASNet>(opts);
    }

    return asnet;
}

static PluginShared<neural_networks::AbstractNetwork> _plugin("asnet", _parse);
