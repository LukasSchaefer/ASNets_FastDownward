#ifndef TASK_UTILS_SAMPLING_TECHNIQUE_H
#define TASK_UTILS_SAMPLING_TECHNIQUE_H

#include "regression_utils.h"
#include "sampling.h"

#include "../abstract_task.h"
#include "../globals.h"
#include "../option_parser.h"
#include "../task_proxy.h"
#include "../evaluator.h"

#include "../tasks/modified_goals_task.h"
#include "../tasks/modified_init_goals_task.h"
#include "../utils/rng.h"
#include "../utils/rng_options.h"
#include "../utils/distribution.h"

#include <memory>
#include <random>

class PartialAssignment;

namespace sampling_technique {

class SamplingTechnique {
private:
    const int count;
    int counter = 0;

protected:
    const std::vector<Evaluator *> evals;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const = 0;
    
    bool check_solvability(std::shared_ptr<AbstractTask> task) const;

public:
    SamplingTechnique(const options::Options &opts);
    SamplingTechnique(int count, std::mt19937 &mt = utils::get_global_mt19937());
    virtual ~SamplingTechnique() = default;

    int get_count() const;
    int get_counter() const;
    bool empty() const;
    const std::shared_ptr<AbstractTask> next(
            const std::shared_ptr<AbstractTask> seed_task = g_root_task());

    virtual void initialize() {
    };
    virtual const std::string &get_name() const = 0;

    static void add_options_to_parser(options::OptionParser &parser);
    static std::vector<int> extractInitialState(const State& state);
    static std::vector<FactPair> extractGoalFacts(
            const GoalsProxy &goals_proxy);
    static std::vector<FactPair> extractGoalFacts(
            const State &state);
};

class TechniqueNull : public SamplingTechnique {
protected:
    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const override;
public:
    //TechniqueNull(const options::Options &opts);
    TechniqueNull();
    virtual ~TechniqueNull() override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};

class TechniqueNoneNone : public SamplingTechnique {
protected:
    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const override;
public:
    TechniqueNoneNone(const options::Options &opts);
    TechniqueNoneNone(int count);
    virtual ~TechniqueNoneNone() override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};

class TechniqueIForwardIForward : public SamplingTechnique {
protected:
    std::shared_ptr<utils::DiscreteDistribution> isteps;
    std::shared_ptr<utils::DiscreteDistribution> gsteps;

    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const override;

public:
    TechniqueIForwardIForward(const options::Options &opts);
    virtual ~TechniqueIForwardIForward() override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};

class TechniqueIForwardNone : public SamplingTechnique {
protected:
    std::shared_ptr<utils::DiscreteDistribution> steps;

    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const override;

public:
    TechniqueIForwardNone(const options::Options &opts);
    virtual ~TechniqueIForwardNone() override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};

class TechniqueNoneGBackward : public SamplingTechnique {
protected:
    std::shared_ptr<utils::DiscreteDistribution> steps;

    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const override;

public:
    TechniqueNoneGBackward(const options::Options &opts);
    virtual ~TechniqueNoneGBackward() override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};


class TechniqueUniformNone : public SamplingTechnique {
protected:   
    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const override;

public:
    TechniqueUniformNone(const options::Options &opts);
    virtual ~TechniqueUniformNone() override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};


class TechniqueUniformUniform : public SamplingTechnique {
protected:   
    virtual const std::shared_ptr<AbstractTask> create_next(
            const std::shared_ptr<AbstractTask> seed_task) const override;

public:
    TechniqueUniformUniform(const options::Options &opts);
    virtual ~TechniqueUniformUniform() override;

    virtual const std::string &get_name() const override;
    const static std::string name;
};
}

#endif

