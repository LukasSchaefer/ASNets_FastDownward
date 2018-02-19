#include "sampling_search.h"

#include "../evaluation_context.h"
#include "../globals.h"
#include "../heuristic.h"
#include "../open_list_factory.h"
#include "../option_parser.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../task_utils/successor_generator.h"
#include "sampling_search.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>

using namespace std;

namespace sampling_search {

    SamplingSearch::SamplingSearch(const Options &opts)
    : SearchEngine(opts),
    search_parse_tree(get_search_parse_tree(opts.get_unparsed_config())) {
    }

    options::ParseTree SamplingSearch::get_search_parse_tree(
            const std::string& unparsed_config) {
        options::ParseTree pt = options::generate_parse_tree(unparsed_config);
        return subtree(pt, options::first_child_of_root(pt));
    }

    void SamplingSearch::next_engine() {
        OptionParser engine_parser(search_parse_tree, false);
        engine = engine_parser.start_parsing<shared_ptr < SearchEngine >> ();
    }

    void SamplingSearch::initialize() {
        cout << "Sampling Manager"
                << endl;

    }

    SearchStatus SamplingSearch::step() {

        next_engine();


        engine->search();
        this->set_plan(engine->get_plan());
        this->solution_found = engine->found_solution();
        cout << "step4" << endl;

        return TIMEOUT;

        //return IN_PROGRESS;
    }

    void SamplingSearch::print_statistics() const {
        cout << "TODO NO STATISTICS CURRENTLY" << endl;
    }

    void SamplingSearch::add_sampling_options(OptionParser &parser) {
        parser.add_option<shared_ptr < SearchEngine >> ("ssearch",
                "Search engine to use for sampling");
    }


}
