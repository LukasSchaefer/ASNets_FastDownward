#ifndef UTILS_DISTRIBUTION_H
#define UTILS_DISTRIBUTION_H

#include "rng.h"
#include "rng_options.h"

#include "../options/options.h"
#include "../options/option_parser.h"

#include <memory>
#include <random>

namespace utils {

template<typename T> class Distribution {
protected:
    std::mt19937 rng;
public:

    Distribution(std::mt19937 &rng);
    Distribution(const options::Options &opts);
    virtual ~Distribution() = default;
    
    virtual T next() = 0;
};

class DiscreteDistribution : public Distribution<int>{
public:
    DiscreteDistribution(std::mt19937 &rng) : Distribution(rng){}
    DiscreteDistribution(const options::Options &opts) : Distribution(opts){}
    virtual ~DiscreteDistribution() {};
};

class UniformIntDistribution : public DiscreteDistribution {
    std::uniform_int_distribution<int> dist;

public:
    UniformIntDistribution(int min, int max, std::mt19937 rng);
    UniformIntDistribution(const options::Options &opts);
    virtual ~UniformIntDistribution() override;

    virtual int next() override;
};
}


#endif /* DISTRIBUTION_H */

