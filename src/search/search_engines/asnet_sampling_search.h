#ifndef SEARCH_ENGINES_ASNET_SAMPLING_SEARCH_H
#define SEARCH_ENGINES_ASNET_SAMPLING_SEARCH_H

#include "../search_engine.h"
#include "../policy.h"
#include "../state_id.h"
#include "../neural_networks/abstract_network.h"

#include "../heuristics/lm_cut_landmarks.h"

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

    /*
     * exhaustive list of all supported additional input features (by name)
     * any additions need also to be considered/ added in extract_sample_entries_trajectory
     * to extract the corresponding values for the samples
     */
    const std::string arr[] = {"none", "landmarks", "binary_landmarks"};
    const std::vector<std::string> supported_additional_input_features(arr, arr + sizeof(arr)/sizeof(std::string));

    class ASNetSamplingSearch : public SearchEngine {
    private:
        static std::hash<std::string> shash;
        std::vector<StateID> network_explored_states = std::vector<StateID>();

    protected:
        options::ParseTree search_parse_tree;
        const std::string problem_hash;
        const std::string target_location;

        /* if true -> sample even teacher trajectories/ paths which did not reach a goal
           otherwise only sample states along plans, so trajectories/ paths leading to a goal state */
        const bool use_non_goal_teacher_paths;

        // if false -> only sample using the network search
        const bool use_teacher_search;

        // name of additional input features to be used
        const std::string additional_input_features;

        // LM-cut landmark generator for additional landmark features if used
        std::unique_ptr<lm_cut_heuristic::LandmarkCutLandmarks> landmark_generator;

        // the fact_goal_values should be stored (equal for all samples)
        std::vector<int> fact_goal_values;

        /* vector of entries of form (variable_index, value_index) for each fact in lexicographical ordering
           of their names */
        std::vector<std::pair<int, int>> facts_sorted;
        /* vector of (original) operator indeces sorted by the corresponding operator names */
        std::vector<int> operator_indeces_sorted;
        /*
         * vector where index corresponds to original operator index and entry is sorted operator
         * index
         */
        std::vector<int> operator_indeces_sorted_reversed;

        // search of network policy
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
                StateRegistry &sr, const OperatorsProxy &ops,
                std::ostringstream &action_opts_stream);
        void landmark_values_input_stream(
                const GlobalState &global_state, const TaskProxy &tp,
                std::ostringstream &add_input_features_stream) const;
        void binary_landmark_values_input_stream(
                const GlobalState &global_state, //const TaskProxy &tp,
                std::ostringstream &add_input_features_stream) const;
        void extract_sample_entries_trajectory(
                const Trajectory &trajectory, const TaskProxy &tp,
		StateRegistry &sr, const OperatorsProxy &ops,
		std::ostream &stream);
        std::string extract_exploration_sample_entries();
        std::string extract_teacher_sample_entries();
        void set_modified_task_with_new_initial_state(StateID state_id,
		const StateRegistry &sr);
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
