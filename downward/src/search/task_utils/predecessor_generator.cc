#include "predecessor_generator.h"

#include "regression_utils.h"

#include "../abstract_task.h"
#include "../global_state.h"
#include "../task_proxy.h"

#include "../task_utils/task_properties.h"

#include <set>

using namespace std;

namespace predecessor_generator {

PredecessorGenerator::PredecessorGenerator(const RegressionTaskProxy &task_proxy)
: rtp(task_proxy),
ops(task_proxy.get_regression_operators()) {
    task_properties::verify_no_axioms(task_proxy);
    task_properties::verify_no_conditional_effects(task_proxy);
}

void PredecessorGenerator::generate_applicable_ops(
    const PartialAssignment &assignment, vector<OperatorID> &applicable_ops) const {

    for (const RegressionOperatorProxy op : ops) {
        if (op.is_applicable(assignment)) {
            applicable_ops.push_back(op.get_global_operator_id());
        }
    }
}

void PredecessorGenerator::generate_applicable_ops(
    const GlobalState &state, vector<OperatorID> &applicable_ops) const {
    generate_applicable_ops(rtp.create_partial_assignment(state.get_values()),
        applicable_ops);
 
}
}
