#include "protobuf_network.h"

#include "../utils/system.h"

using namespace std;
using namespace tensorflow;
namespace neural_networks {
ProtobufNetwork::ProtobufNetwork(const Options &opts)
    : AbstractNetwork(),
      task(opts.get<shared_ptr<AbstractTask>>("transform")),
      task_proxy(*task),
      path(opts.get<string>("path")),
      input_layer_name(opts.get<string>("input_layer")),
      output_layer_name(opts.get<string>("output_layer")) {

        // Initialize a tensorflow session
        Status status = NewSession(SessionOptions(), &session);
        if (!status.ok()) {
            std::cout << "Tensorflow session error: " 
                      << status.ToString() << "\n";
            utils::exit_with(utils::ExitCode::CRITICAL_ERROR);
        }

        GraphDef graph_def;
        status = ReadBinaryProto(Env::Default(), path, &graph_def);
        if (!status.ok()) {
            std::cout << "Protobuf loading error: "
                      << status.ToString() << "\n";
            utils::exit_with(utils::ExitCode::CRITICAL_ERROR);
        }

        // Add the graph to the session
        status = session->Create(graph_def);
        if (!status.ok()) {
            std::cout << "Session graph creation error: " 
                      << status.ToString() << "\n";
            utils::exit_with(utils::ExitCode::CRITICAL_ERROR);
        }
        
        initialize_inputs();
    }

ProtobufNetwork::~ProtobufNetwork(){
    session->Close();
}


void ProtobufNetwork::evaluate(const State& state){
    fill_input(state);
    
    Status status = session->Run(inputs,{output_layer_name},
    {
    }, &outputs);


    if (!status.ok()) {
        std::cout << "Network evaluation error: " << status.ToString() << "\n";
        utils::exit_with(utils::ExitCode::CRITICAL_ERROR);
    }

    extract_output();
}

void ProtobufNetwork::add_options_to_parser(options::OptionParser& parser) {
    parser.add_option<shared_ptr<AbstractTask>>(
        "transform",
        "Optional task transformation for the network."
        " Currently, adapt_costs(), sampling_transform(), and no_transform() are "
        "available.",
        "no_transform()");
    parser.add_option<string>("path", "Path to networks protobuf file.");
    parser.add_option<string>("input_layer",
        "Name of the computation graphs input layer.");
    parser.add_option<string>("output_layer",
        "Name of the computation graphs output layer.");
}
}