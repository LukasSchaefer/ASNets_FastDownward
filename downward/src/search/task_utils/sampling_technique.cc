#include "sampling_technique.h"

#include "sampling.h"
#include "successor_generator.h"

#include "../plugin.h"

using namespace std;



namespace sampling_technique {

/* START DEFINITION SAMPLING_TECHNIQUE */
SamplingTechnique::SamplingTechnique(const options::Options& opts)
: count(opts.get<int>("count")),
rng(utils::parse_rng_from_options(opts)){ }

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
        return create_next(seed_task);
    }
}

void SamplingTechnique::add_options_to_parser(options::OptionParser& parser) {
    parser.add_option<int>("count", "Number of times this sampling "
        "technique shall be used.");
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


/* START DEFINITION TECHNIQUE_FORWARD_NONE */
const std::string TechniqueForwardNone::name = "forward_none";

const string &TechniqueForwardNone::get_name() const {
    return name;
}

TechniqueForwardNone::TechniqueForwardNone(const options::Options& opts)
: SamplingTechnique(opts),
steps(opts.get<shared_ptr<utils::DiscreteDistribution>>("distribution")){ }

TechniqueForwardNone::~TechniqueForwardNone() { }

const shared_ptr<AbstractTask> TechniqueForwardNone::create_next(
    const shared_ptr<AbstractTask> seed_task) const {

    TaskProxy task_proxy(*seed_task);
    const successor_generator::SuccessorGenerator successor_generator(task_proxy);
    State new_init = sampling::sample_state_with_random_forward_walk(task_proxy,
        successor_generator, task_proxy.get_initial_state(), steps->next(), *rng);

    return make_shared<extra_tasks::ModifiedInitGoalsTask>(seed_task,
        extractInitialState(new_init),
        extractGoalFacts(task_proxy.get_goals()));
}
/* START DEFINITION TECHNIQUE_NONE_BACKWARDS */
const std::string TechniqueNoneBackward::name = "none_backward";

const string &TechniqueNoneBackward::get_name() const {
    return name;
}

TechniqueNoneBackward::TechniqueNoneBackward(const options::Options& opts)
: SamplingTechnique(opts),
steps(opts.get<shared_ptr<utils::DiscreteDistribution>>("distribution")){ }

TechniqueNoneBackward::~TechniqueNoneBackward() { }

const shared_ptr<AbstractTask> TechniqueNoneBackward::create_next(
    const shared_ptr<AbstractTask> seed_task) const {

    RegressionTaskProxy task_proxy(*seed_task);
    const predecessor_generator::PredecessorGenerator predecessor_generator(task_proxy);
    PartialAssignment new_goal = sampling::sample_partial_assignment_with_random_backward_walk(
        task_proxy, predecessor_generator, task_proxy.get_goal_assignment(), steps->next(), *rng);

    return make_shared<extra_tasks::ModifiedInitGoalsTask>(seed_task,
        extractInitialState(task_proxy.get_initial_state()),
        new_goal.get_assigned_facts());
}

/* SHARED PLUGINS FOR PARSING */
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

static shared_ptr<SamplingTechnique> _parse_technique_forward_none(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    parser.add_option<shared_ptr<utils::DiscreteDistribution>>("distribution",
        "Discrete random distribution to determine the random walk length used"
        " by this technique.");
        
        
    Options opts = parser.parse();

    shared_ptr<TechniqueForwardNone> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueForwardNone>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_forward_none(
    TechniqueForwardNone::name, _parse_technique_forward_none);


static shared_ptr<SamplingTechnique> _parse_technique_none_backward(
    OptionParser &parser) {
    SamplingTechnique::add_options_to_parser(parser);
    parser.add_option<shared_ptr<utils::DiscreteDistribution>>("distribution",
        "Discrete random distribution to determine the random walk length used"
        " by this technique.");
        
    Options opts = parser.parse();

    shared_ptr<TechniqueNoneBackward> technique;
    if (!parser.dry_run()) {
        technique = make_shared<TechniqueNoneBackward>(opts);
    }
    return technique;
}

static PluginShared<SamplingTechnique> _plugin_technique_none_backward(
    TechniqueNoneBackward::name, _parse_technique_none_backward);

}