#ifndef SEARCH_ENGINES_SAMPLING_SEARCH_H
#define SEARCH_ENGINES_SAMPLING_SEARCH_H

#include "../open_list.h"
#include "../search_engine.h"

#include <memory>
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
    
    struct DataEntry{
        
    };
class SamplingSearch : public SearchEngine {
    


protected:
    const options::ParseTree search_parse_tree;
    std::shared_ptr<SearchEngine> engine;

    options::ParseTree get_search_parse_tree(const std::string& unparsed_config);
    void next_engine();
    
    
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit SamplingSearch(const options::Options &opts);
    virtual ~SamplingSearch() = default;

    virtual void print_statistics() const override;

    
    static void add_sampling_options(options::OptionParser &parser);
};
}

#endif
