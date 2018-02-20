#include "task_proxy.h"

#include "task_utils/causal_graph.h"

#include <iostream>

using namespace std;

void State::dump_pddl(std::ostream& out) const {
    for (FactProxy fact : (*this)) {
        string fact_name = fact.get_name();
        if (fact_name != "<none of those>")
            out << fact_name << endl;
    }
}

void State::dump_fdr(std::ostream& out) const {
    for (FactProxy fact : (*this)) {
        VariableProxy var = fact.get_variable();
        out << "  #" << var.get_id() << " [" << var.get_name() << "] -> "
             << fact.get_value() << endl;
    }
}

const causal_graph::CausalGraph &TaskProxy::get_causal_graph() const {
    return causal_graph::get_causal_graph(task);
}
