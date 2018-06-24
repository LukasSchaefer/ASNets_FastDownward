#include "lexicographical_access.h"


namespace lexicographical_access {

/* sorts facts lexicographically and returns vector with tuple entries of
   (variable_index, value_index (in variable domain)) */
std::vector<std::pair<int, int>> get_facts_lexicographically(TaskProxy task_proxy) {
    // compute necessary vector size (sum of all variable domain sizes)
    unsigned int num_facts = 0;
    for (int dom_size : g_variable_domain) {
        num_facts += dom_size;
    }

    // fill vector of triples consisting of (fact_name, variable_index, value_index)
    std::vector<std::tuple<std::string, int, int>> facts_with_names(num_facts);
    unsigned int index = 0;
    for (unsigned int var_index = 0; var_index < task_proxy.get_variables().size(); var_index++) {
        for (unsigned int val_index = 0; val_index < g_fact_names[var_index].size(); val_index++) {
            std::string fact_name = g_fact_names[var_index][val_index];
            std::tuple<std::string, int, int> fact_triple = std::make_tuple(fact_name, var_index, val_index);
            facts_with_names[index] = fact_triple;
            index++;
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
std::vector<int> get_operator_indeces_lexicographically(TaskProxy task_proxy) {
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
}