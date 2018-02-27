#include "sampling_search.h"

#include "../evaluation_context.h"
#include "../globals.h"
#include "../heuristic.h"
#include "../open_list_factory.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../task_utils/successor_generator.h"
#include "../task_utils/sampling.h"
#include "../utils/rng.h"
#include "../utils/rng_options.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>


using namespace std;

namespace sampling_search {

/* START DEFINITION SAMPLING_TECHNIQUE */
SamplingTechnique::SamplingTechnique(const options::Options& opts)
: count(opts.get<int>("count")),
rng(utils::parse_rng_from_options(opts)) { }

SamplingTechnique::SamplingTechnique(int count)
: count(count) { }

int SamplingTechnique::get_count() const {
    return count;
}

int SamplingTechnique::get_counter() const {
    return counter;
}

bool SamplingTechnique::empty() const {
    return (counter >= count);
}

const shared_ptr<AbstractTask> SamplingTechnique::next(
    const shared_ptr<AbstractTask> seed_task) {

    if (empty()) {
        return nullptr;
    } else {
        counter++;
        return create_next(seed_task);
    }
}

void SamplingTechnique::add_options_to_parser(options::OptionParser& parser) {
    parser.add_option<int>("count", "Number of times this sampling "
        "technique shall be used.");
    utils::add_rng_options(parser);
}

/* START DEFINITION TECHNIQUE_NULL */
const std::string TechniqueNull::name = "null";

const string &TechniqueNull::get_name() const {
    return name;
}

TechniqueNull::TechniqueNull()
: SamplingTechnique(0) { }

TechniqueNull::~TechniqueNull() { }

const shared_ptr<AbstractTask> TechniqueNull::create_next(
    const shared_ptr<AbstractTask> seed_task) const {
    // HACK to circumvent the WError + UnusedParameterWarning
    shared_ptr<AbstractTask> hack = seed_task;
    hack = nullptr;
    return hack;
}

/*
 I think it is not necessary for the user to parse TechniqueNull

 TechniqueNull::TechniqueNull(const options::Options& opts)
: SamplingTechnique(opts) { }
 
static shared_ptr<SamplingTechnique> _parse_technique_null(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    Options opts = parser.parse();

    shared_ptr<TechniqueNull> technique;
    if (!parser.dry_run()) {
        technique = make_shared<sampling_search::TechniqueNull>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_none_none(
    TechniqueNull::name, _parse_technique_null);
*/

/* START DEFINITION TECHNIQUE_NONE_NONE */
const std::string TechniqueNoneNone::name = "none_none";

const string &TechniqueNoneNone::get_name() const {
    return name;
}

TechniqueNoneNone::TechniqueNoneNone(const options::Options& opts)
: SamplingTechnique(opts) { }

TechniqueNoneNone::TechniqueNoneNone(int count)
: SamplingTechnique(count) { }

TechniqueNoneNone::~TechniqueNoneNone() { }

const shared_ptr<AbstractTask> TechniqueNoneNone::create_next(
    const shared_ptr<AbstractTask> seed_task) const {
    return seed_task;
}

static shared_ptr<SamplingTechnique> _parse_technique_none_none(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    Options opts = parser.parse();

    shared_ptr<TechniqueNoneNone> technique;
    if (!parser.dry_run()) {
        technique = make_shared<sampling_search::TechniqueNoneNone>(opts);
    }
    return technique;
}
static PluginShared<SamplingTechnique> _plugin_technique_none_none(
    TechniqueNoneNone::name, _parse_technique_none_none);

/* START DEFINITION TECHNIQUE_FORWARD_NONE */
const std::string TechniqueForwardNone::name = "forward_none";

const string &TechniqueForwardNone::get_name() const {
    return name;
}

TechniqueForwardNone::TechniqueForwardNone(const options::Options& opts)
: SamplingTechnique(opts) { }

TechniqueForwardNone::~TechniqueForwardNone() { }

const shared_ptr<AbstractTask> TechniqueForwardNone::create_next(
    const shared_ptr<AbstractTask> seed_task) const {

    TaskProxy task_proxy(*seed_task);
    const successor_generator::SuccessorGenerator successor_generator(task_proxy);
    const State initial_state = task_proxy.get_initial_state();
    
    State s = sampling::sample_state_with_random_forward_walk(task_proxy,
        successor_generator, initial_state, 10, *rng);
    cout << ">>>>>>>>>>>>>>>>NEW STATE "<<endl;
    s.dump_pddl();
    cout << endl;
    return seed_task;
}

static shared_ptr<SamplingTechnique> _parse_technique_forward_none(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    Options opts = parser.parse();

    shared_ptr<TechniqueForwardNone> technique;
    if (!parser.dry_run()) {
        technique = make_shared<sampling_search::TechniqueForwardNone>(opts);
    }
    return technique;
}
static PluginShared<SamplingTechnique> _plugin_technique_forward_none(
    TechniqueForwardNone::name, _parse_technique_forward_none);

/* START DEFINITION SAMPLING_SEARCH */
SamplingSearch::SamplingSearch(const Options &opts)
: SearchEngine(opts),
search_parse_tree(prepare_search_parse_tree(opts.get_unparsed_config())),
problem_hash(opts.get<string>("hash")),
target_location(opts.get<string>("target")),
field_separator(opts.get<string>("separator")),
sampling_techniques(prepare_sampling_techniques(
                    opts.get_list<shared_ptr<SamplingTechnique>>("techniques"))),
current_technique(sampling_techniques.begin()) { }

options::ParseTree SamplingSearch::prepare_search_parse_tree(
    const std::string& unparsed_config) const {
    options::ParseTree pt = options::generate_parse_tree(unparsed_config);
    return subtree(pt, options::first_child_of_root(pt));
}

vector<shared_ptr<SamplingTechnique>>
    SamplingSearch::prepare_sampling_techniques(
                vector<shared_ptr<SamplingTechnique>> input) const {
    if (input.empty()){
        input.push_back(make_shared<sampling_search::TechniqueNull>());
    }
    return input;
}

void SamplingSearch::next_engine() {
    OptionParser engine_parser(search_parse_tree, false);
    engine = engine_parser.start_parsing<shared_ptr < SearchEngine >> ();
}

std::string SamplingSearch::extract_modification_hash(
    State init, GoalsProxy goals) const {

    ostringstream oss;
    init.dump_pddl(oss);
    goals.dump_pddl(oss);
    std::string merged = oss.str();
    return to_string(SamplingSearch::shash(merged));
}

std::string SamplingSearch::extract_sample_entries() const {
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

            heuristic += ops[plan[idx_init]].get_cost();


            new_entries << problem_hash << field_separator;
            new_entries << modification_hash << field_separator;

            sr.lookup_state(trajectory[idx_init]).dump_pddl(new_entries);
            new_entries << field_separator;
            new_entries << pddl_goal.str() << field_separator;
            new_entries << heuristic << field_separator;
            new_entries << ops[plan[idx_init]].get_name() << field_separator;

            sr.lookup_state(trajectory[idx_init + 1]).dump_pddl(new_entries);
            new_entries << "~";

        }
    }
    string post = new_entries.str();
    replace(post.begin(), post.end(), '\n', '\t');
    replace(post.begin(), post.end(), '~', '\n');
    return post;
}

void SamplingSearch::initialize() {
    cout << "Sampling Manager...";
    
    
    cout << "done." << endl;
}

SearchStatus SamplingSearch::step() {

    if ((*current_technique)->empty()) {
        current_technique++;
        if (current_technique == sampling_techniques.end()) {
            return SOLVED;
        }
    }
    TaskProxy tp(*g_root_task());
    cout <<"ORIG"<<endl;
    tp.get_initial_state().dump_pddl();
    modified_task = (*current_technique)->next(g_root_task());
    next_engine();
    engine->search();
    
    //TODO this is for debugging and shows the last technique's run output
    this->set_plan(engine->get_plan());
    this->set_trajectory(engine->get_trajectory());
    this->solution_found = engine->found_solution();

    samples << extract_sample_entries();

    return IN_PROGRESS;

}

void SamplingSearch::print_statistics() const {
    int new_samples = 0;

    if (samples.str().compare("")) {
        string s = samples.str();
        for (unsigned int i = 0; i <= s.size(); i++) {
            if (s[i] == '\n') {
                new_samples++;
            }
        }
    }
    cout << "Generated Samples: " << (generated_samples + new_samples) << endl;
    cout << "Sampling Techniques used:" << endl;
    for (auto &st : sampling_techniques) {
        cout << '\t' << st->get_name();
        cout << ":\t" << st->get_counter();
        cout << "/" << st->get_count() << '\n';
    }
}

void SamplingSearch::save_plan_intermediate() {
    if (samples.str().compare("")) {
        string s = samples.str();

        ofstream outfile(g_plan_filename, ios::app);
        outfile << s;
        for (unsigned int i = 0; i <= s.size(); i++) {
            if (s[i] == '\n') {
                generated_samples++;
            }
        }
        samples.str("");
        samples.clear();
    }

}

void SamplingSearch::save_plan_if_necessary() const {
    if (samples.str().compare("")) {
        string s = samples.str();

        ofstream outfile(g_plan_filename, ios::app);
        outfile << s;
    }
}

void SamplingSearch::add_sampling_options(OptionParser &parser) {
    parser.add_option<shared_ptr < SearchEngine >> ("search",
        "Search engine to use for sampling");
    parser.add_option<std::string> ("target",
        "Place to save the sampled data (currently only appending files"
        "is supported", "None");
    parser.add_list_option<shared_ptr < SamplingTechnique >> ("techniques",
        "List of sampling technique definitions to use",
        "[]");
    parser.add_option<std::string> ("hash",
        "MD5 hash of the input problem. This can be used to "
        "differentiate which problems created which entries.", "None");
    parser.add_option<std::string> ("separator",
        "String to use to separate the different fields in a data sample",
        ";");
}



/*
 * Use modified_task as transformed task in heuristic (and now also sampling).
 * 
 * We use this to instanciate search engines with our desired modified tasks.
 */
std::shared_ptr<AbstractTask> modified_task = nullptr;

static shared_ptr<AbstractTask> _parse_sampling_transform(
    OptionParser &parser) {
    if (parser.dry_run())
        return nullptr;
    else
        return modified_task;
}
static PluginShared<AbstractTask> _plugin_sampling_transform(
    "sampling_transform", _parse_sampling_transform);

}
