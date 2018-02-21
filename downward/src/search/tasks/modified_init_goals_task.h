#ifndef TASKS_MODIFIED_INIT_GOALS_TASK_H
#define TASKS_MODIFIED_INIT_GOALS_TASK_H


#include "delegating_task.h"

#include <vector>

namespace extra_tasks {

    class ModifiedInitGoalsTask : public tasks::DelegatingTask {
        std::vector<int> initial_state;
        const std::vector<FactPair> goals;
    public:
        ModifiedInitGoalsTask(
            const std::shared_ptr<AbstractTask> &parent,
            std::vector<int> &&initial_state,
            std::vector<FactPair> &&goals);
        virtual ~ModifiedInitGoalsTask() = default;
    
        virtual int get_num_goals() const override;
        virtual FactPair get_goal_fact(int index) const override;
        virtual std::vector<int> get_initial_state_values() const override;

    };
}
#endif