#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_H

#include "../open_list.h"
#include "../search_engine.h"

#include "../task_utils/sampling_technique.h"

#include <functional>
#include <memory>
#include <sstream>
#include <tree.hh>
#include <vector>

class Evaluator;
class Heuristic;
class PruningMethod;

namespace options {
    class Options;
    class ParseNode;
    using ParseTree = tree<ParseNode>;
}

namespace sampling_search {
    using Trajectory = std::vector<StateID>;

    enum SampleType {
        TRAJECTORY_SOLUTION, TRAJECTORY_OTHER, STATE_OTHER
    };

    class SamplingSearch : public SearchEngine {
    private:
        static std::hash<std::string> shash;

    protected:
        const options::ParseTree search_parse_tree;
        const std::string problem_hash;
        const std::string target_location;
        const std::string field_separator;

        const bool store_solution_trajectories;
        const bool expand_solution_trajectory;
        const bool store_other_trajectories;
        const bool store_all_states;
        const std::vector<std::string> use_heuristics;
        const int threshold_samples_for_disk;


        const std::vector<std::shared_ptr<sampling_technique::SamplingTechnique>> sampling_techniques;
    private:
        std::vector<std::shared_ptr<sampling_technique::SamplingTechnique>>::const_iterator current_technique;

    protected:

        std::shared_ptr<SearchEngine> engine;
        std::ostringstream samples;
        int samples_for_disk = 0;

        /* Statistics*/
        int generated_samples = 0;

        options::ParseTree prepare_search_parse_tree(
                const std::string& unparsed_config) const;
        std::vector<std::shared_ptr<sampling_technique::SamplingTechnique>> prepare_sampling_techniques(
                std::vector<std::shared_ptr<sampling_technique::SamplingTechnique>> input) const;
        void next_engine();
        std::string extract_modification_hash(State init, GoalsProxy goals) const;
        std::string extract_sample_entries();
        void extract_sample_entries_add_heuristics(const GlobalState &state,
                std::ostream &stream) const;
        int extract_sample_entries_trajectory(
                const Plan &plan, const Trajectory &trajectory, bool expand,
                const StateRegistry &sr, OperatorsProxy &ops,
                const std::string& modification_hash, SampleType trajectory_type,
                std::ostream &stream) const;
        void extract_sample_entries_state(
                StateID &sid, SampleType type,
                const std::string &goal_description,
                const StateRegistry &sr, const SearchSpace &ss, OperatorsProxy &ops,
                const std::string& modification_hash,
                std::ostream &stream) const;
        void save_plan_intermediate();


        virtual void initialize() override;
        virtual SearchStatus step() override;

    public:
        explicit SamplingSearch(const options::Options &opts);
        virtual ~SamplingSearch() = default;

        virtual void print_statistics() const override;
        virtual void save_plan_if_necessary() const override;


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
