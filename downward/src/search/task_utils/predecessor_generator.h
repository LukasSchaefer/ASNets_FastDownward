#ifndef TASK_UTILS_PREDECESSOR_GENERATOR_H
#define TASK_UTILS_PREDECESSOR_GENERATOR_H

#include "regression_utils.h"

#include <memory>
#include <vector>

class GlobalState;
class OperatorID;
class PartialAssignment;
class TaskProxy;
class OperatorsProxy;

/* TODO use the more efficient look up like successor generator
 */
namespace predecessor_generator {

class PredecessorGenerator {
    ops;
public:
    explicit PredecessorGenerator(const TaskProxy &task_proxy);
    /*
      We cannot use the default destructor (implicitly or explicitly)
      here because GeneratorBase is a forward declaration and the
      incomplete type cannot be destroyed.
    */
    ~PredecessorGenerator() = default;

    void generate_applicable_ops(
        const PartialAssignment &assignment, std::vector<OperatorID> &applicable_ops) const;
    // Transitional method, used until the search is switched to the new task interface.
    void generate_applicable_ops(
        const GlobalState &state, std::vector<OperatorID> &applicable_ops) const;
};
}

#endif
