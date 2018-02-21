#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_H

#include "../open_list.h"
#include "../search_engine.h"

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

    struct DataEntry {
    };
    
    
    class SamplingTechnique {
    private:
        const int count;
        int counter = 0;
    
    protected:
        virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) = 0;
        
    public:
        SamplingTechnique(const options::Options &opts);
        virtual ~SamplingTechnique();
        
        int get_count() const;
        int get_counter() const;
        bool empty() const;
        const std::shared_ptr<AbstractTask> next(
            const std::shared_ptr<AbstractTask> seed_task = g_root_task());
        
        static void add_options_to_parser(options::OptionParser &parser);
    };

    class TechniqueForwardNone : public SamplingTechnique {
    public:
        TechniqueForwardNone(const options::Options &opts);
        virtual ~TechniqueForwardNone() = default;
        
        virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task);
        
        const static std::string name;
    };
    class SamplingSearch : public SearchEngine {
    private:
        static std::hash<std::string> shash;


    protected:
        const options::ParseTree search_parse_tree;
        std::string problem_hash;
        std::string target_location;
        std::string field_separator;
        
        const std::vector<std::shared_ptr<SamplingTechnique>> sampling_techniques;

        std::shared_ptr<SearchEngine> engine;
        std::ostringstream samples;

        /* Statistics*/
        int generated_samples = 0;

        options::ParseTree get_search_parse_tree(const std::string& unparsed_config);
        void next_engine();
        std::string extract_modification_hash(State init, GoalsProxy goals);
        std::string extract_sample_entries();
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
}

#endif
