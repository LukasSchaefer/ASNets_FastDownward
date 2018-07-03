#ifndef TASK_UTILS_LEXICOGRAPHICAL_ACCESS_H
#define TASK_UTILS_LEXICOGRAPHICAL_ACCESS_H

#include "../globals.h"
#include "../task_proxy.h"

#include <vector>
#include <tuple>
#include <string.h>

namespace lexicographical_access {

    /* sorts facts lexicographically and returns vector with tuple entries of
    (variable_index, value_index (in variable domain)) */
    std::vector<std::pair<int, int>> get_facts_lexicographically(TaskProxy task_proxy);

    std::vector<int> get_operator_indeces_lexicographically(TaskProxy task_proxy);
}
#endif