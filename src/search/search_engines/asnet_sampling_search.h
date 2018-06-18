#ifndef SEARCH_ENGINES_ASNET_SAMPLING_SEARCH_H
#define SEARCH_ENGINES_ASNET_SAMPLING_SEARCH_H

#include "../search_engine.h"
#include "../policy.h"
#include "../state_id.h"
#include "../neural_networks/abstract_network.h"

#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <../ext/tree.hh>
#include <tuple>
#include <vector>

class Evaluator;
class Policy;
class PruningMethod;

namespace options {
    class Options;
    struct ParseNode;
    using ParseTree = tree<ParseNode>;
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
        options::ParseTree search_parse_tree;
        const std::string problem_hash;
        const std::string target_location;

        // limit for explored trajectories during network policy exploration
        const int exploration_trajectory_limit;
        /* if true -> sample even teacher trajectories/ paths which did not reach a goal
           otherwise only sample states along plans, so trajectories/ paths leading to a goal state */
        const bool use_non_goal_teacher_paths;

        /* vector of entries of form (variable_index, value_index) for each fact in lexicographical ordering
           of their names */
        std::vector<std::pair<int, int>> facts_sorted;
        /* vector of operator indeces sorted by the corresponding operator names */
        std::vector<int> operator_indeces_sorted;

        // network
        std::shared_ptr<neural_networks::AbstractNetwork> asnet;
        // Policy and corresponding Search of network
        Policy * network_policy;
        std::shared_ptr<SearchEngine> network_search;

        // teacher search used for sampling along reasonable (usually optimal) trajectories
        std::shared_ptr<SearchEngine> teacher_search;
    protected:

        std::ostringstream samples;

        /* Statistics*/
        int generated_samples = 0;

        options::ParseTree prepare_search_parse_tree(
                const std::string& unparsed_config) const;
        void goal_into_stream(std::ostringstream &goal_stream) const;
        void state_into_stream(const GlobalState &state,
                std::ostringstream &state_stream) const;
        std::vector<int> applicable_values_into_stream(
                const GlobalState &state, const OperatorsProxy &ops,
		std::ostringstream &applicable_stream) const;
        void network_probs_into_stream(
                const GlobalState &state, const OperatorsProxy &ops,
                std::ostringstream &network_probs_stream) const;
        void action_opt_values_into_stream(
                const GlobalState &state, std::vector<int> applicable_values,
                const OperatorsProxy &ops, std::ostringstream &action_opts_stream);
        void extract_sample_entries_trajectory(
                const Trajectory &trajectory, const StateRegistry &sr,
                const OperatorsProxy &ops, std::ostream &stream);
        std::string extract_exploration_sample_entries();
        std::string extract_teacher_sample_entries();
        void set_modified_task_with_new_initial_state(StateID state_id);
        std::shared_ptr<SearchEngine> get_new_teacher_search_with_modified_task() const;
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