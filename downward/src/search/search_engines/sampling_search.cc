#include "sampling_search.h"

#include "../evaluation_context.h"
#include "../globals.h"
#include "../heuristic.h"
#include "../open_list_factory.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../tasks/modified_init_goals_task.h"
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


SamplingSearch::SamplingSearch(const Options &opts)
: SearchEngine(opts),
search_parse_tree(prepare_search_parse_tree(opts.get_unparsed_config())),
problem_hash(opts.get<string>("hash")),
target_location(opts.get<string>("target")),
field_separator(opts.get<string>("separator")),
store_solution_trajectories(opts.get<bool>("store_solution_trajectory")),
expand_solution_trajectory(opts.get<bool>("expand_solution_trajectory")),
store_other_trajectories(opts.get<bool>("store_other_trajectories")),
store_all_states(opts.get<bool>("store_all_states")),
sampling_techniques(prepare_sampling_techniques(
opts.get_list<shared_ptr<sampling_technique::SamplingTechnique>>("techniques"))),
current_technique(sampling_techniques.begin()) { }

options::ParseTree SamplingSearch::prepare_search_parse_tree(
    const std::string& unparsed_config) const {
    options::ParseTree pt = options::generate_parse_tree(unparsed_config);
    return subtree(pt, options::first_child_of_root(pt));
}

vector<shared_ptr<sampling_technique::SamplingTechnique>>
SamplingSearch::prepare_sampling_techniques(
    vector<shared_ptr<sampling_technique::SamplingTechnique>> input) const {
    if (input.empty()) {
        input.push_back(make_shared<sampling_technique::TechniqueNull>());
    }
    return input;
}

void SamplingSearch::next_engine() {
    sampling_search::trajectories.clear();
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
    /* 
     Data set format (; represents the field separator which can be changed):
     <T>;<ProblemHash>;<ModificationHash>;<CurrentState>;<GoalPredicates>;
        <Operator>;<OtherState>;<HeuristicViaTrajectory>;<Heuristics>*
     
     All files exists in every sampling entry, although some might be empty.
     <T> := * if entry belongs to solution path, 
            + if entry belongs to a trajectory stored by the search algorithm,
            - if the entry belongs to an arbitrary visited state
     <ModificationHash> := md5 hash of input problem file (or 'NA' if missing)
     <ModificationHash> := hash of the modified states initial state + goals
     <CurrentState> := State of this entry
     <GoalPredicates> := goal predicates of the problem to solve
     
     The next two are either both filled or both empty:
     <Operator> :=  if <T> in {*,+}: operator chosen in the current state
                    if <T> in {-}: operator used to reach current state
     <OtherState> := if <T> in {*,+}: state resulting from applying operator
                     if <T> in {-}: parent state using operator to reach current
     
     <HeuristicViaTrajectory> := if the current state is in the solution
     path, then this heuristic value is the value of cost accumulated from goal
     to current state. Otherwise, this field is empty
     <Heuristics>* := List of heuristic values estimated for the current state
    */
    
    const GlobalState goal_state = engine->get_goal_state();
    const StateRegistry &sr = engine->get_state_registry();
    const SearchSpace &ss = engine->get_search_space();
    const TaskProxy &tp = engine->get_task_proxy();
    OperatorsProxy ops = tp.get_operators();

    std::string modification_hash = extract_modification_hash(
        tp.get_initial_state(), tp.get_goals());
    


    ostringstream new_entries;
    
    if (store_solution_trajectories && engine->found_solution()){
            Plan plan = engine->get_plan();
            
            Trajectory trajectory;
            ss.trace_path(goal_state, trajectory);
            
            for (int idx_goal = trajectory.size() - 1; idx_goal >= 1; idx_goal--) {

        int heuristic = 0;
        ostringstream pddl_goal;
        // TODO: Replace by partial assignments via Regression from Goal
        sr.lookup_state(trajectory[idx_goal]).dump_pddl(pddl_goal);

        for (int idx_init = idx_goal - 1; idx_init >= 0; idx_init--) {

            heuristic += ops[plan[idx_init]].get_cost();

            if (idx_goal == (int)(trajectory.size()-1) && idx_init == 0){
                new_entries <<">>>>>>>>>>>>>>>>>>>>>>>>>";
            }
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

    }
    
    if (store_other_trajectories){
        for (const Trajectory& trajectory: sampling_search::trajectories){
            
        }
    }
    
    if (store_all_states){
        
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
    cout << "ORIG" << endl;
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
    parser.add_list_option<shared_ptr < sampling_technique::SamplingTechnique >> ("techniques",
        "List of sampling technique definitions to use",
        "[]");
    parser.add_option<bool> ("store_solution_trajectory",
        "Stores for every state on the solution path which operator was chosen"
        "next to reach the goal, the next state reached, and the heuristic "
        "values estimated for the state.", "true");
    parser.add_option<bool> ("expand_solution_trajectory",
        "Stores for every state on the solution path which operator was chosen"
        "next to reach the goal, the next state reached, and the heuristic "
        "values estimated for the state.", "true");
    parser.add_option<bool> ("store_other_trajectories",
        "Stores for every state on the other trajectories (has only an effect,"
        "if the used search engine stores other trajectories) which operator "
        "was chosen next to reach the goal, the next state reached, and the "
        "heuristic values estimated for the state.", "true");
    parser.add_option<bool> ("store_all_states",
        "Stores for every state visited the operator chosen to reach it,"
        ", its parent state, and the heuristic "
        "values estimated for the state.", "true");
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
std::shared_ptr<AbstractTask> modified_task = g_root_task();

static shared_ptr<AbstractTask> _parse_sampling_transform(
    OptionParser &parser) {
    if (parser.dry_run()){
        return nullptr;
    } else {
        return modified_task;
    }
}
static PluginShared<AbstractTask> _plugin_sampling_transform(
    "sampling_transform", _parse_sampling_transform);



/* Global variable for search algorithms to store arbitrary trajectories 
   (the storing of the solution trajectory is independent of this variable).
   The variable will be cleaned for every new search engine and is checked for
   storage afte reach search engine execution. */
std::vector<Trajectory> trajectories();
   
}
