#include "sampling_search.h"
#include "search_common.h"

#include "../option_parser.h"
#include "../plugin.h"

using namespace std;

namespace plugin_sampling {
static shared_ptr<SearchEngine> _parse(OptionParser &parser) {
    parser.document_synopsis("Sampling Search Manager", "");
    
    
    sampling_search::SamplingSearch::add_sampling_options(parser);
    SearchEngine::add_options_to_parser(parser);
    
    
    Options opts = parser.parse();

    shared_ptr<sampling_search::SamplingSearch> engine;
    if (!parser.dry_run()) {
        engine = make_shared<sampling_search::SamplingSearch>(opts);
    }

    return engine;
}

static PluginShared<SearchEngine> _plugin("sampling", _parse);
}
