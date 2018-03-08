#include "distribution.h"

#include "../plugin.h"

#include <memory>

using namespace std;
namespace utils {

template<typename T>
Distribution<T>::Distribution(std::mt19937& rng)
: rng(rng) { }

template<typename T>
Distribution<T>::Distribution(const options::Options& opts) {
    int seed = opts.get<int>("random_seed");
    if (seed == -1) {
        rng = utils::get_global_mt19937();
    } else {
        rng.seed(seed);
    }
}

UniformIntDistribution::UniformIntDistribution(int min, int max, std::mt19937 rng)
: DiscreteDistribution(rng),
dist(std::uniform_int_distribution<int>(min, max - 1)) { }

UniformIntDistribution::UniformIntDistribution(const options::Options& opts)
: DiscreteDistribution(opts),
dist(std::uniform_int_distribution<int>(opts.get<int>("min"), opts.get<int>("max") - 1)) { }

UniformIntDistribution::~UniformIntDistribution() { }

int UniformIntDistribution::next() {
    return dist(rng);
}

void add_general_parser_options(options::OptionParser &parser) {
    parser.add_option<int>(
        "random_seed",
        "Set to -1 (default) to use the global random number generator. "
        "Set to any other value to use a local random number generator with "
        "the given seed.",
        "-1",
        options::Bounds("-1", "infinity"));
}

static shared_ptr<DiscreteDistribution> _parse_uniform_int_distribution(
    options::OptionParser &parser) {

    parser.add_option<int>("min",
        "Minimum value (included) to return from the distribution");
    parser.add_option<int>("max",
        "Maximum value (excluded) to return from the distribution");

    add_general_parser_options(parser);

    options::Options opts = parser.parse();

    shared_ptr<UniformIntDistribution> distribution;
    if (!parser.dry_run()) {
        distribution = make_shared<UniformIntDistribution>(opts);
    }
    return distribution;
}

static PluginShared<DiscreteDistribution> _plugin_uniform_int_distribution(
    "uniform_int_dist", _parse_uniform_int_distribution);




}