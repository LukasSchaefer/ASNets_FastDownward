#include <algorithm>

#include "sampling_technique.h"

#include "sampling.h"
#include "successor_generator.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../plugin.h"
#include "../state_registry.h"

using namespace std;



namespace sampling_technique {

/* START DEFINITION SAMPLING_TECHNIQUE */
SamplingTechnique::SamplingTechnique(const options::Options& opts)
: count(opts.get<int>("count")),
  evals(opts.get_list<Evaluator *>("evals")),
  rng(utils::parse_rng_from_options(opts)){
    for (Evaluator *e: evals) {
        if (!e->dead_ends_are_reliable()) {
            cout << "Warning: A given dead end detection evaluator is not safe." <<endl;
        }
    }
}

SamplingTechnique::SamplingTechnique(int count, mt19937 &mt)
: count(count),
rng(make_shared<utils::RandomNumberGenerator>(mt)){ }

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
        while(true) {
            shared_ptr<AbstractTask> next_task = create_next(seed_task);
            if(check_solvability(next_task)) {
                return next_task;
            }
        }
    }
}


bool SamplingTechnique::check_solvability(shared_ptr<AbstractTask> task) const {
    if (evals.empty()) {
        return true;
    }
    StateRegistry state_registry (
        *task, *g_state_packer, *g_axiom_evaluator, task->get_initial_state_values());
    EvaluationContext eval_context (state_registry.get_initial_state());
    const GlobalState &gs = state_registry.get_initial_state();
    gs.dump_pddl();
    for (Evaluator * e: evals){
        EvaluationResult eval_result = e->compute_result(eval_context);
        if (eval_result.is_infinite()){
            return false;
        }
    }
    return true;
}

void SamplingTechnique::add_options_to_parser(options::OptionParser& parser) {
    parser.add_option<int>("count", "Number of times this sampling "
        "technique shall be used.");
    parser.add_list_option<Evaluator *>("evals", "evaluators for dead-end "
    "detection (use only save ones to not reject a non dead end). If any of the "
    "evaluators detects a dead end, the assignment is rejected. \nATTENTON: "
    "The evaluators are initialize on the original task. Thus, evaluators which "
    "require to be initialized on the correct initial state and goal do not "
    "work!", "[]");
    utils::add_rng_options(parser);
}

vector<int> SamplingTechnique::extractInitialState(const State& state) {
    vector<int> values;
    values.reserve(state.size());
    for (int val : state.get_values()) {
        values.push_back(val);
    }
    return values;
}

vector<FactPair> SamplingTechnique::extractGoalFacts(
    const GoalsProxy& goals_proxy) {
    vector<FactPair> goals;
    goals.reserve(goals_proxy.size());
    for (unsigned int i = 0; i < goals_proxy.size(); i++) {
        goals.push_back(goals_proxy[i].get_pair());
    }
    return goals;
}

vector<FactPair> SamplingTechnique::extractGoalFacts(const State& state) {
    vector<FactPair> goals;
    goals.reserve(state.size());
    int var = 0;
    for (int val : state.get_values()) {
        goals.emplace_back(var, val);
        var++;
    }
    return goals;
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


/* PARSING TECHNIQUE_NONE_NONE*/
static shared_ptr<SamplingTechnique> _parse_technique_none_none(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    Options opts = parser.parse();

    shared_ptr<TechniqueNoneNone> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueNoneNone>(opts);
    }
    return technique;
}
static PluginShared<SamplingTechnique> _plugin_technique_none_none(
    TechniqueNoneNone::name, _parse_technique_none_none);


/* START DEFINITION TECHNIQUE_IFORWARD_NONE */
const std::string TechniqueIForwardNone::name = "iforward_none";

const string &TechniqueIForwardNone::get_name() const {
    return name;
}

TechniqueIForwardNone::TechniqueIForwardNone(const options::Options& opts)
: SamplingTechnique(opts),
steps(opts.get<shared_ptr<utils::DiscreteDistribution>>("distribution")){ }

TechniqueIForwardNone::~TechniqueIForwardNone() { }

const shared_ptr<AbstractTask> TechniqueIForwardNone::create_next(
    const shared_ptr<AbstractTask> seed_task) const {

    TaskProxy task_proxy(*seed_task);
    const successor_generator::SuccessorGenerator successor_generator(task_proxy);
    State new_init = sampling::sample_state_with_random_forward_walk(task_proxy,
        successor_generator, task_proxy.get_initial_state(), steps->next(), *rng);

    return make_shared<extra_tasks::ModifiedInitGoalsTask>(seed_task,
        extractInitialState(new_init),
        extractGoalFacts(task_proxy.get_goals()));
}

/* PARSING TECHNIQUE_IFORWARD_NONE*/
static shared_ptr<SamplingTechnique> _parse_technique_iforward_none(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    parser.add_option<shared_ptr<utils::DiscreteDistribution>>("distribution",
        "Discrete random distribution to determine the random walk length used"
        " by this technique.");
        
    Options opts = parser.parse();

    shared_ptr<TechniqueIForwardNone> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueIForwardNone>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_iforward_none(
    TechniqueIForwardNone::name, _parse_technique_iforward_none);


/* START DEFINITION TECHNIQUE_IFORWARD_IFORWARD */
const std::string TechniqueIForwardIForward::name = "iforward_iforward";

const string &TechniqueIForwardIForward::get_name() const {
    return name;
}

TechniqueIForwardIForward::TechniqueIForwardIForward(const options::Options& opts)
: SamplingTechnique(opts),
isteps(opts.get<shared_ptr<utils::DiscreteDistribution>>("dist_init")),
gsteps(opts.get<shared_ptr<utils::DiscreteDistribution>>("dist_goal")) { }

TechniqueIForwardIForward::~TechniqueIForwardIForward() { }

const shared_ptr<AbstractTask> TechniqueIForwardIForward::create_next(
    const shared_ptr<AbstractTask> seed_task) const {

    TaskProxy task_proxy(*seed_task);
    const successor_generator::SuccessorGenerator successor_generator(task_proxy);
    State new_init = sampling::sample_state_with_random_forward_walk(task_proxy,
        successor_generator, task_proxy.get_initial_state(), isteps->next(), *rng);

    State new_goal = sampling::sample_state_with_random_forward_walk(task_proxy,
        successor_generator, task_proxy.get_initial_state(), gsteps->next(), *rng);
    
    return make_shared<extra_tasks::ModifiedInitGoalsTask>(seed_task,
        extractInitialState(new_init),
        extractGoalFacts(new_goal));
}


/* PARSING TECHNIQUE_IFORWARD_IFORWARD*/

static shared_ptr<SamplingTechnique> _parse_technique_iforward_iforward(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    parser.add_option<shared_ptr<utils::DiscreteDistribution>>("dist_init",
        "Discrete random distribution to determine the random walk length used"
        " by this technique for the initial state.");
    parser.add_option<shared_ptr<utils::DiscreteDistribution>>("dist_goal",
        "Discrete random distribution to determine the random walk length used"
        " by this technique for the goal state.");
        
    Options opts = parser.parse();

    shared_ptr<TechniqueIForwardIForward> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueIForwardIForward>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_iforward_iforward(
    TechniqueIForwardIForward::name, _parse_technique_iforward_iforward);


/* START DEFINITION TECHNIQUE_NONE_GBACKWARDS */
const std::string TechniqueNoneGBackward::name = "none_gbackward";

const string &TechniqueNoneGBackward::get_name() const {
    return name;
}

TechniqueNoneGBackward::TechniqueNoneGBackward(const options::Options& opts)
: SamplingTechnique(opts),
steps(opts.get<shared_ptr<utils::DiscreteDistribution>>("distribution")){ }

TechniqueNoneGBackward::~TechniqueNoneGBackward() { }

const shared_ptr<AbstractTask> TechniqueNoneGBackward::create_next(
    const shared_ptr<AbstractTask> seed_task) const {

    RegressionTaskProxy task_proxy(*seed_task);
    const predecessor_generator::PredecessorGenerator predecessor_generator(task_proxy);
    PartialAssignment new_goal = sampling::sample_partial_assignment_with_random_backward_walk(
        task_proxy, predecessor_generator, task_proxy.get_goal_assignment(), steps->next(), *rng);

    return make_shared<extra_tasks::ModifiedInitGoalsTask>(seed_task,
        extractInitialState(task_proxy.get_initial_state()),
        new_goal.get_assigned_facts());
}

/* PARSING TECHNIQUE_NONE_GBACKWARDS*/
static shared_ptr<SamplingTechnique> _parse_technique_none_gbackward(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    parser.add_option<shared_ptr<utils::DiscreteDistribution>>("distribution",
        "Discrete random distribution to determine the random walk length used"
        " by this technique.");
        
    Options opts = parser.parse();

    shared_ptr<TechniqueNoneGBackward> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueNoneGBackward>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_none_gbackward(
    TechniqueNoneGBackward::name, _parse_technique_none_gbackward);



/* START DEFINITION TECHNIQUE_UNIFORM_NONE */
const std::string TechniqueUniformNone::name = "uniform_none";

const string &TechniqueUniformNone::get_name() const {
    return name;
}

TechniqueUniformNone::TechniqueUniformNone(const options::Options& opts)
: SamplingTechnique(opts) { }

TechniqueUniformNone::~TechniqueUniformNone() { }

const shared_ptr<AbstractTask> TechniqueUniformNone::create_next(
    const shared_ptr<AbstractTask> seed_task) const {
    
    TaskProxy seed_task_proxy(*seed_task);
    
    vector<int> init;
    for (int var = 0; var < seed_task->get_num_variables(); ++var) {
        init.push_back((*rng)(seed_task->get_variable_domain_size(var)));
    }
    return make_shared<extra_tasks::ModifiedInitGoalsTask>(seed_task,
        move(init), extractGoalFacts(seed_task_proxy.get_goals()));
    
}

/* PARSING TECHNIQUE_UNIFORM_NONE*/
static shared_ptr<SamplingTechnique> _parse_technique_uniform_none(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);   
    

    Options opts = parser.parse();

    shared_ptr<TechniqueUniformNone> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueUniformNone>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_uniform_none(
    TechniqueUniformNone::name, _parse_technique_uniform_none);


/* START DEFINITION TECHNIQUE_UNIFORM_UNIFORM */
const std::string TechniqueUniformUniform::name = "uniform_uniform";

const string &TechniqueUniformUniform::get_name() const {
    return name;
}

TechniqueUniformUniform::TechniqueUniformUniform(const options::Options& opts)
: SamplingTechnique(opts) { }

TechniqueUniformUniform::~TechniqueUniformUniform() { }

const shared_ptr<AbstractTask> TechniqueUniformUniform::create_next(
    const shared_ptr<AbstractTask> seed_task) const {
    
    TaskProxy seed_task_proxy(*seed_task);
    
    vector<int> init;
    for (int var = 0; var < seed_task->get_num_variables(); ++var) {
        init.push_back((*rng)(seed_task->get_variable_domain_size(var)));
    }
    return make_shared<extra_tasks::ModifiedInitGoalsTask>(seed_task,
        move(init), extractGoalFacts(seed_task_proxy.get_goals()));
    
}

/* PARSING TECHNIQUE_UNIFORM_NONE*/
static shared_ptr<SamplingTechnique> _parse_technique_uniform_uniform(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);   
    
    Options opts = parser.parse();

    shared_ptr<TechniqueUniformUniform> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueUniformUniform>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_uniform_uniform(
    TechniqueUniformUniform::name, _parse_technique_uniform_uniform);


}

