#ifndef SEARCH_ENGINES_ASNET_SAMPLING_SEARCH_H
#define SEARCH_ENGINES_ASNET_SAMPLING_SEARCH_H

#include "../search_engine.h"
#include "../search_engines/policy_search.h"
#include "../policy.h"
#include "../state_id.h"

#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <tuple>
#include <vector>

class Evaluator;
class Policy;
class PruningMethod;

namespace options {
    class Options;
}

namespace asnet_sampling_search {
    using Trajectory = std::vector<StateID>;

    enum ASNetSampleType {
        TEACHER_TRAJECTORY, NETWORK_TRAJECTORY
    };

    class ASNetSamplingSearch : public SearchEngine {
    private:
        static std::hash<std::string> shash;
        std::vector<StateID> network_explored_states = std::vector<StateID>();

    protected:
        const std::string problem_hash;
        const std::string target_location;

        /* vector of entries of form (variable_index, value_index) for each fact in lexicographical ordering
           of their names */
        std::vector<std::pair<int, int>> facts_sorted;
        /* vector of operator indeces sorted by the corresponding operator names */
        std::vector<int> operator_indeces_sorted;

        // teacher policy used for extraction along reasonable (usually optimal) trajectories
        const Policy* teacher_policy;
        // limit for explored trajectories during network policy exploration
        const int exploration_trajectory_limit;

        Policy * network_policy;
        PolicySearch network_search;
        PolicySearch teacher_search;
    protected:

        std::ostringstream samples;

        /* Statistics*/
        int generated_samples = 0;

        std::string extract_modification_hash(State init, GoalsProxy goals) const;
        void goal_into_stream(ostringstream goal_stream) const;
        void state_into_stream(GlobalState &state, ostringstream state_stream) const;
        void applicable_values_into_stream(GlobalState &state, ostringstream applicable_stream) const;
        void network_probs_into_stream(GlobalState &state, ostringstream network_probs_stream) const;
        void action_opt_values_into_stream(GlobalState &state, ostringstream action_opts_stream) const;
        int extract_sample_entries_trajectory(
                const Plan &plan, const Trajectory &trajectory,
                const StateRegistry &sr, OperatorsProxy &ops,
                std::ostream &stream) const;
        std::string extract_exploration_sample_entries();
        std::string extract_teacher_sample_entries();
        void add_header_samples(std::ostream &stream) const;
        void save_plan_intermediate();


        virtual void initialize() override;
        virtual SearchStatus step() override;

    public:
        explicit ASNetSamplingSearch(const options::Options &opts);
        virtual ~ASNetSamplingSearch() = default;

        virtual void print_statistics() const override;

        static void add_sampling_options(options::OptionParser &parser);

    };

    extern std::shared_ptr<AbstractTask> modified_task;

    class Path {
        using Plan = std::vector<OperatorID>;
    protected:
        Trajectory trajectory;
        Plan plan;
    public:
        Path(StateID start);
        Path(const Path &) = delete;
        ~Path();

        void add(OperatorID op, StateID next);

        const Trajectory &get_trajectory() const;
        const Plan &get_plan() const;
    };
    extern std::vector<Path> paths;
}

#endif