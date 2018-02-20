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
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

using namespace std;

namespace sampling_search {

    SamplingSearch::SamplingSearch(const Options &opts)
    : SearchEngine(opts),
    search_parse_tree(get_search_parse_tree(opts.get_unparsed_config())),
    problem_hash(opts.get<string>("hash")),
    target_location(opts.get<string>("target")),
    field_separator(opts.get<string>("separator")) {
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

    std::string SamplingSearch::extract_modification_hash(State init, GoalsProxy goals) {

        ostringstream oss;
        init.dump_pddl(oss);
        goals.dump_pddl(oss);
        std::string merged = oss.str();
        return to_string(SamplingSearch::shash(merged));
    }

    std::string SamplingSearch::extract_sample_entries() {
        Plan plan = engine->get_plan();
        Trajectory trajectory = engine->get_trajectory();
        const StateRegistry &sr = engine->get_state_registry();
        const TaskProxy &tp = engine->get_task_proxy();
        OperatorsProxy ops = tp.get_operators();

        std::string modification_hash = extract_modification_hash(
                tp.get_initial_state(), tp.get_goals());


        ostringstream new_entries;

        for (int idx_goal = trajectory.size() - 1;
                idx_goal >= 1; idx_goal--) {

            int heuristic = 0;
            ostringstream pddl_goal;
            // TODO: Replace by partial assignments via Regression from Goal
            sr.lookup_state(trajectory[idx_goal]).dump_pddl(pddl_goal);

            for (int idx_init = idx_goal - 1;
                    idx_init >= 0; idx_init--) {

                heuristic += ops[plan[idx_init]].get_cost()


                        new_entries << problem_hash << field_separator
                        new_entries << modification_hash << field_separator;

                sr.lookup_state(trajectory[idx_init]).dump_pddl(new_entries);
                new_entries << field_separator
                        << pddl_goal.str() << field_separator
                        << heuristic << field_separator
                        << ops[plan[idx_init]].get_name() << field_separator;

                sr.lookup_state(trajectory[idx_init + 1]).dump_pddl(new_entries);
                new_entries << endl;

            }
        }
        return new_entries.str();
    }

    void SamplingSearch::initialize() {
        cout << "Sampling Manager"
                << endl;


    }

    SearchStatus SamplingSearch::step() {

        next_engine();

        engine->search();
        //not needed, just for printing something

        this->set_plan(engine->get_plan());
        this->set_trajectory(engine->get_trajectory());
        this->solution_found = engine->found_solution();

        samples << extract_sample_entries();

        return TIMEOUT;

        //return IN_PROGRESS;
    }

    void SamplingSearch::print_statistics() const {
        cout << "TODO NO STATISTICS CURRENTLY" << endl;
    }

    void SamplingSearch::save_plan_if_necessary() const {
        if (samples.str().compare("")) {
            ofstream outfile(g_plan_filename, ios::app);
            outfile << samples.str();
        }
    }

    void SamplingSearch::add_sampling_options(OptionParser &parser) {
        parser.add_option<shared_ptr < SearchEngine >> ("search",
                "Search engine to use for sampling");
        parser.add_option<std::string> ("target",
                "Place to save the sampled data (currently only appending files"
                "is supported", "None");
        parser.add_option<std::string> ("hash",
                "MD5 hash of the input problem. This can be used to "
                "differentiate which problems created which entries.", "None");
        parser.add_option<std::string> ("separator",
                "String to use to separate the different fields in a data sample",
                ";");

    }


}
