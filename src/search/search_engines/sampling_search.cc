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

#include "../task_utils/sampling.h"
#include "../utils/rng.h"

#include "../task_utils/regression_utils.h"
#include "../task_utils/predecessor_generator.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <stdlib.h>

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
use_heuristics(opts.get_list<string>("use_registered_heuristics")),
threshold_samples_for_disk(opts.get<int>("sample_cache_size")),
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
    sampling_search::paths.clear();
    g_reset_registered_heuristics();
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

void SamplingSearch::extract_sample_entries_add_heuristics(
    const GlobalState &state, std::ostream& stream) const {
    for (const string &heuristic : use_heuristics) {
        EvaluationContext eval_context(state);

        stream << field_separator
            << heuristic << "=" << g_registered_heuristics[heuristic]
            ->compute_result(eval_context).get_h_value();
    }
}

int SamplingSearch::extract_sample_entries_trajectory(
    const Plan &plan, const Trajectory &trajectory, bool expand,
    const StateRegistry &sr, OperatorsProxy &ops,
    const string& meta,
    ostream &stream) const {

    int counter = 0;
    
    int min_idx_goal = (expand) ? (1) : (trajectory.size() - 1);
    for (int idx_goal = trajectory.size() - 1; idx_goal >= min_idx_goal;
        idx_goal--) {

        int heuristic = 0;
        ostringstream pddl_goal;
        // TODO: Replace by partial assignments via Regression from Goal
        sr.lookup_state(trajectory[idx_goal]).dump_pddl(pddl_goal);

        for (int idx_init = idx_goal - 1; idx_init >= 0; idx_init--) {

            heuristic += ops[plan[idx_init]].get_cost();

            if ((!expand) && (idx_init != 0)) {
                continue;
            }

            GlobalState init_state = sr.lookup_state(trajectory[idx_init]);

            stream << meta;
            init_state.dump_pddl(stream);
            stream << field_separator
                << pddl_goal.str() << field_separator
                << ops[plan[idx_init]].get_name() << field_separator;
            sr.lookup_state(trajectory[idx_init + 1]).dump_pddl(stream);
            stream << field_separator
                << heuristic;

            // the heuristics calculate their values only w.r.t. the true goal
            if (idx_goal == (int) trajectory.size() - 1) {
                extract_sample_entries_add_heuristics(init_state, stream);
            }

            stream << "~";
            counter++;
        }
    }
    return counter;
}

void SamplingSearch::extract_sample_entries_state(
    StateID &sid, const string &goal_description,
    const StateRegistry &sr, const SearchSpace &ss, OperatorsProxy &ops,
    const string& meta,
    ostream &stream) const {

    const GlobalState state = sr.lookup_state(sid);
    OperatorID oid = ss.get_creating_operator(state);
    StateID pid = ss.get_parent_id(state);

    stream << meta;
    state.dump_pddl(stream);
    stream << field_separator
        << goal_description << field_separator;

    if (oid != OperatorID::no_operator) {
        stream << ops[oid].get_name();
    }
    stream << field_separator;
    if (pid != StateID::no_state) {
        sr.lookup_state(ss.get_parent_id(state)).dump_pddl(stream);
    }
    stream << field_separator;

    extract_sample_entries_add_heuristics(state, stream);

    stream << "~";
}

std::string SamplingSearch::extract_sample_entries() {
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
     
     <HeuristicViaTrajectory> := cost from the current state to the end of the
            trajectory. If the trajectory is the solution path, then this is
            the cost to the goal. Otherwise, this is the cost to some arbitrary
            trajectory end. If the entry does not belong to a trajectory, then
            this field is empty.
     <Heuristics>* := List of heuristic values estimated for the current state
     */
    const StateRegistry &sr = engine->get_state_registry();
    const SearchSpace &ss = engine->get_search_space();
    const TaskProxy &tp = engine->get_task_proxy();
    OperatorsProxy ops = tp.get_operators();
    const GoalsProxy gps = tp.get_goals();

    int count = 0;
    ostringstream meta_info;
    meta_info << "<Meta problem_hash=\"" << this->problem_hash 
              << "\" modification_hash=\"" 
              << extract_modification_hash(tp.get_initial_state(), tp.get_goals())
              << "\" format=\"FD\" type=\"";
    ostringstream new_entries;

    if (store_solution_trajectories && engine->found_solution()) {
        const string meta_info_opt_traj = meta_info.str() + "O\">";
        const GlobalState goal_state = engine->get_goal_state();
        Plan plan = engine->get_plan();
        Trajectory trajectory;
        ss.trace_path(goal_state, trajectory);

        count += extract_sample_entries_trajectory(plan, trajectory,
            expand_solution_trajectory, sr, ops, meta_info_opt_traj,
            new_entries);
    }

    if (store_other_trajectories) {
        const string meta_info_traj = meta_info.str() + "T\">";
        for (const Path& path : sampling_search::paths) {
            count += extract_sample_entries_trajectory(
                path.get_plan(), path.get_trajectory(),
                false, sr, ops, meta_info_traj,
                new_entries);
        }
    }

    if (store_all_states) {
        const string meta_info_traj = meta_info.str() + "S\">";
        ostringstream pddl_goal;
        // TODO: Replace by partial assignments via Regression from Goal
        gps.dump_pddl(pddl_goal);

        for (StateRegistry::const_iterator iter = sr.begin();
            iter != sr.end(); ++iter) {
            StateID state = *iter;
            extract_sample_entries_state(state, pddl_goal.str(),
                sr, ss, ops, meta_info_traj, new_entries);
            count++;
        }
    }

    string post = new_entries.str();
    replace(post.begin(), post.end(), '\n', '\t');
    replace(post.begin(), post.end(), '~', '\n');

    samples_for_disk += count;
    return post;
}

void SamplingSearch::initialize() {

    cout << "Initializing Sampling Manager...";
    add_header_samples(samples);

    cout << "done." << endl;
}

SearchStatus SamplingSearch::step() {

    if ((*current_technique)->empty()) {
        current_technique++;
        if (current_technique == sampling_techniques.end()) {
            save_plan_intermediate();
            return SOLVED;
        }
    }

    modified_task = (*current_technique)->next(task);
    next_engine();
    engine->search();
    samples << extract_sample_entries();

    if (samples_for_disk > threshold_samples_for_disk) {
        save_plan_intermediate();
        samples_for_disk = 0;
    }
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

void SamplingSearch::add_header_samples(ostream &stream) const {
    stream << "# Everything in a line after '#' is a comment" << endl;
    stream << "# Entry format:<Meta problem_hash=\"<HASH>\" modification_hash=\"<HASH>\" format=\"<FORMAT>\" type=\"<TYPE>\">; <CurrentState>; <GoalPredicates>; <Operator>; <OtherState>; <HeuristicViaTrajectory>; <Heuristics>*" << endl;
    stream << "# <HASH> := hash value to identify where the sample comes from" << endl;
    stream << "# <FORMAT> := format of the sample predicates. FD is the format FastDownward produces (only speaking about present not pruned atoms)." << endl;
    stream << "# <TYPE> := tells which type of entry this is (state of optimal(O)/any trajectory(T), arbitrary state of search space(S))" << endl;
    stream << "# <Operator> :=  if <T> in {*,+}: operator chosen in the current state. if <T> in {-}: operator used to reach current state" << endl;
    stream << "# <OtherState> := if <T> in {*,+}: state resulting from applying operator. if <T> in {-}: parent state using operator to reach current" << endl;
}

void SamplingSearch::save_plan_intermediate() {
    if (samples.str().compare("")) {
        string s = samples.str();

        ofstream outfile(g_plan_filename, ios::app);
        outfile << s;
        //number of entries is a bit more complicated, because we filter
        //comments out.
        bool comment = false;
        int line_length = 0;
        for (unsigned int i = 0; i <= s.size(); i++) {
            if (s[i] == '\n') {
                if (line_length > 0)
                    generated_samples++;
                comment = false;
                line_length = 0;
            } else if (s[i] == '#') {
                comment = true;
            } else if (!comment) {
                line_length++;
            }
        }
        samples.str("");
        samples.clear();
    }
}

void SamplingSearch::save_plan_if_necessary() const {
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
        "values estimated for the state.", "false");
    parser.add_option<int> ("sample_cache_size",
        "If more than sample_cache_size samples are cached, then the entries"
        " are written to disk and the cache is emptied. When sampling "
        "finishes, all remaining cached samples are written to disk. If running"
    "out of memory, the current cache is lost.", "5000");
    parser.add_list_option<std::string> ("use_registered_heuristics",
        "Stores for every state visited the operator chosen to reach it,"
        ", its parent state, and the heuristic "
        "values estimated for the state.", "[]");
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
    if (parser.dry_run()) {
        return nullptr;
    } else {

        return modified_task;
    }
}
static PluginShared<AbstractTask> _plugin_sampling_transform(
    "sampling_transform", _parse_sampling_transform);

/* Global variable for search algorithms to store arbitrary paths (plans [
   operator ids] and trajectories [state ids]).
   (the storing of the solution trajectory is independent of this variable).
   The variable will be cleaned for every new search engine and is checked for
   storage afte reach search engine execution. */
Path::Path(StateID start) {

    trajectory.push_back(start);
}

Path::~Path() { }

void Path::add(OperatorID op, StateID next) {

    plan.push_back(op);
    trajectory.push_back(next);
}

const Path::Plan &Path::get_plan() const {

    return plan;
}

const Trajectory &Path::get_trajectory() const {
    return trajectory;
}


std::vector<Path> paths;

}
