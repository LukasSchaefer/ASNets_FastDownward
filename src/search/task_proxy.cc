#include "task_proxy.h"

#include "task_utils/causal_graph.h"
#include "globals.h"

#include <iostream>

using namespace std;

const int PartialAssignment::UNASSIGNED = -1;

void PartialAssignment::dump_pddl(std::ostream& out) const {
    for (unsigned int var = 0; var < (*this).size(); ++var) {
        if ((*this).assigned(var)) {
            FactProxy fact = (*this)[var];
            string fact_name = fact.get_name();
            if (fact_name != "<none of those>")
                out << fact_name << endl;
        }
    }
}

void PartialAssignment::dump_fdr(std::ostream& out) const {
    for (unsigned int var = 0; var < (*this).size(); ++var) {
        if ((*this).assigned(var)) {
            FactProxy fact = (*this)[var];
            VariableProxy var = fact.get_variable();
            out << "  #" << var.get_id() << " [" << var.get_name() << "] -> "
                << fact.get_value() << endl;
        }
    }
}

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

