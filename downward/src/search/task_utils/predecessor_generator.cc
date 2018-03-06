#include "successor_generator.h"
#include "predecessor_generator.h"

#include "../abstract_task.h"
#include "../global_state.h"
#include "../task_proxy.h"

#include "../task_utils/task_properties.h"

#include <set>

using namespace std;

namespace predecessor_generator {

PredecessorGenerator::PredecessorGenerator(const TaskProxy &task_proxy)
: ops(task_proxy.get_operators()) {
    task_properties::verify_no_axioms(task_proxy);
    task_properties::verify_no_conditional_effects(task_proxy);
}

void PredecessorGenerator::generate_applicable_ops(
    const PartialAssignment &assignment, vector<OperatorID> &applicable_ops) const { }

void PredecessorGenerator::generate_applicable_ops(
    const GlobalState &state, vector<OperatorID> &applicable_ops) const {
    /*
     * Original code from Bachelorthesis:
     * Evaluation of Regression Search and State Subsumption in Classical Planning. Andreas Thüring. University of Basel. 31.07.2015. Examiner: Prof. Dr. Malte Helmert
     */
    for (OperatorProxy &op : ops) {
        bool cnd_eff_g_interset = false;
        bool cnd_eff_g_disagree = false;
        bool cnd_pre_g_disagree = false;

        set<int> vars_with_effects;
        for (EffectProxy &effect : op.get_effects()) {
            //TODO conditional effects

            FactPair fact = effect.get_fact().get_pair()
                vars_with_effects.insert(fact.var);

            if (state[fact.var] != PartialAssignment::UNASSIGNED) {
                continue;
            }

            if (state[fact.var] == fact.val) {
                cnd_eff_g_interset = true;
            } else {
                cnd_eff_g_disagree = true;
                break;

            }


        }

        if (!cnd_eff_g_disagree && cnd_eff_g_interset) {
            for (FactProxy &fp : op.get_preconditions()) {
                FactPair fact = fp.get_pair();
                if (vars_with_effects.find(fact.var) != vars_with_effects.end()) {

                }
            }




            if (applicable) {
                for (auto condition : op.get_preconditions()) {
                    bool has_effect = false;
                    for (auto var : vals_with_effects) {
                        if (var == condition.var) {
                            has_effect = true;
                            break;
                        }
                    }
                    if (!has_effect) {
                        if (state[condition.var] != g_variable_domain[condition.var]
                            && state[condition.var] != condition.val) {
                            applicable = false;
                            break;
                        }
                    }
                }
            }
            if (applicable) {
                applicable_ops.push_back(&op);
            }
        }
    }
}
}
