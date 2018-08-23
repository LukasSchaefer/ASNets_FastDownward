# Action Schema Networks in Neural Fast Downward

This is the repository to the BSc thesis of Lukas Schäfer (luki.schaefer96@gmail.com) about *Domain-Dependent Policy Learning using Neural Networks in Classical Planning*. The contributions of the thesis involve the implementation and adaption of [Action Schema Networks](https://github.com/qxcv/asnets) (short ASNets), which are a neural network architecture suited for automatic planning proposed last year by [Sam Toyer et al.](https://arxiv.org/abs/1709.04271), in the [Fast-Downward planning system](http://www.fast-downward.org/) for application in classical, deterministic planning. Our work is therefore strongly influenced by the provided foundation of Sam Toyer et al.

The repository is build upon the Neural Fast-Downward adaption of Fast-Downward. More information on this work and especially its setup can be found at the end of this README.

### Dependencies

Additional to the build and setup of the Neural Fast-Downward repository (for more information see the end of the corresponding second half of this README), multiple python libraries are necessary for the training and creation of the networks:

- numpy
- matplotlib
- tensorflow (Version 1.8.0)
- keras (Version 2.1.6)

These can all be installed using pip(3).

In the following, we will first provide a general overview of the implemented key features in the repository and provide general usage information.

## Overview

### Network Definition

ASNets were implemented using Keras and all modules were implemented as custom keras layers in *network\_models/asnets/custom\_keras\_layers*. The network itself is constructed by the ASNet\_Model\_Builder class in *network\_models/asnets/asnet_keras_model.py*.

The necessary relations between grounded actions and propositions besides additional pruning in the planning task and information is computed and accessible via the *network\_models/asnets/problem\_meta.py* class.

### Fast-Downward Extensions

#### Policies

As ASNets compute policies by definition, we added policies as a general evaluator concept to Fast-Downward built upon the already existing concept of preferred operators. Generally a policy, i.e. a probability distribution over actions for a state, is represented by OperatorIDs (preferred operators) and probabilities (preferences) for each operator.

#### Policy-Search

To be able to evaluate policies and exploit them during search we added a simple policy search simply following the most probable action in each state according to the provided policy. It should be noted that the search fails whenever a previously encountered state is reached as this would lead to a diverging cycle of the search.

#### Sampling Search

During the training of the networks we sample states in a planning task and store them in a sampling file in a specific format. The implemented sampling search executes network searches as well as an arbitrary teacher search to collect states from the state-space of the given planning task. Each stored state is represented as follows:

	<HASH>; <FACT_GOAL_VALUES>; <FACT_VALUES>; <ACTION_APPLICABLE_VALUES>; <ACTION_OPT_VALUES>(; <ADDITIONAL_FEATURES>)
with the following ";" seperated fields:

- \<HASH>: hash-value indicating the problem instance
- \<FACT\_GOAL\_VALUES>: binary value for every fact indicating whether the fact is part of the goal. Values are ordered lexicographically by fact-names in a "," separated list form
- \<FACT\_VALUES>: binary values (0,1) indicating whether a fact is true. Values are ordered lexicographically by fact-names and are all "," separated in a list and are given for every fact (e.g. [0,1,1,0]) 
- \<ACTION\_APPLICABLE\_VALUES>: binary values indicating whether an action is applicable in the current state. Ordering again is lexicographically by action-names and the values are in a "," separated list form for all actions.
- \<ACTION\_OPT\_VALUES>: binary value for each action indicating whether the action starts a found plan according to the teacher-search. Again ordered lexicographically by action-names in a "," separated list.
- \<ADDITIONAL\_FEATURES>: additional input features given for each action in lexicographical order. The supported features are listed in supported\_additional\_input\_features

### Training

Our training cycle is a slight modification of the one proposed by Sam Toyer for probabilistic planning.
It is the main functionality of the *fast-asnet.py* file. It is executed with

	python3 fast-asnet.py -t -d <TRAIN_DIRECTORY> <ADDITIONAL_OPTIONS>

where the *-t* flag indicates that training should be executed and *\<TRAIN\_DIRECTORY\>* is the directory containing the .pddl problem files as well as the domain named as *domain.pddl*. For all options and the option to evaluate the network see the *--help* options of *fast-asnet.py*.

In the end the training will store the final weights obtained in the training directory in a HDF5 file. This file can be used to load the weights in a Keras ASNet model of the same configuration for the same domain.

### Evaluation

To evaluate an ASNet on planning problems, the simplest method is to use the *-e* option of the *fast-asnet.py* file. For more information see the *--help* options of *fast-asnet.py*.

It is also possible to directly execute the network policy search using *fast-downward.py*. Therefore, one has to build the respective Keras ASNet model first using *build\_and\_store\_asnet\_as\_pb.py* (or *build\_networks\_for\_eval.py* if the model for all problem instances of potentially even multiple domains should be created).

Afterwards, the network search can be called as follows:

	./fast-downward.py --build <BUILD> <path/to/problem.pddl> --search "policysearch(p=np(network=asnet(path=<path/to/network\_model.pb>)))"

## Tables \& Statistics

For the thesis, multiple scripts to create PGFPlots for Latex were created which contain information regarding the training or evaluation alike. Note that the scripts are explicitly written for our four baseline planners and three ASNet configurations. However, most scripts should be easily modifiable to represent any similar experiments.

### Training Graphs

For the training procedure the following information can be tracked, collected and represented as PGFPlot graphs:

- Loss value development
- Success-rate during training (success-rate is the percentage of solved problems during the network exploration of the sampling executed in each problem epoch during training)
- time distribution of the training: how much time is spent
	1. to create the ASNet network files
	2. during sampling
	3. in optimizer gradient-descent steps
- Action probabilities in the network policy (for all chosen actions during the network exploration of the sampling)

The graphs can be obtained in the following way.

1. train the ASNets for the domain using
	*fast-asnet.py -t -d \<path/to/domain/training\_dir\> --print\_all <further\_options> \> training.log*. Note the *--print\_all* option and lead the training prints into a log-file. This contains all the necessary output used later to create the graphs.
2. create a summary log-file containing all the important information via *evaluation/training\_parser.py \<path/to/training.log\> domain\_name > training\_sum.log*
3. store all these summary log-files (note they have to be named *training\_sum.log*!) in a directory hierarchy with a folder containing a folder for each domain (and nothing else). These domain-folders contain a directory for each used training configuration which contain the *training\_sum.log* file for the respective domain and configuration.
4. use *evaluation/training\_graphs\_generator.py \<path/to/training\_data\_dir\> \<path/to/save\_dir\>* where *training\_data\_dir* contains the domain folders etc. as described in 3. and the created graphs should all be stored in a hierarchy in *save\_dir*.

All scripts used to create the graphs are stored in *evaluation/evaluation\_data\_scripts* and can be modified.

### Evaluation Graphs

For the evaluation, a coverage table indicating the coverage of each planner for every domain can be created. Additionally, a table for each problem evaluated can be created indicating information regarding the performance of each planner like plan-length, planning-time and for the networks the network-creation time.

The information for these tables is extracted out of HTML reports created by a [LAB](https://bitbucket.org/jendrikseipp/lab/src/default/) evaluation.

The exact LAB experiments used can be found in *evaluation/lab\_files* besides our modification of *lab/downward/experiment.py* in the LAB repository. This was necessary to evaluate using our networks. Sadly, for each configuration it is necessary to modify the *experiment.py* to represent the configuration used (conf-name in line 347) and the respective path in line 355.

After the LAB experiments were run for each ASNet configuration and the baselines then execute the following:

1. *evaluation/evaluation\_data\_scripts/baselines\_evaluation\_report\_data.py \<path/to/baseline\_report.html\> \<path/to/evaluation\_tables/save\_dir\>*: This will create the tables containing the baseline planner data already. To add the network information, also execute the following steps.
2. execute *evaluation/evaluation\_data\_scripts/network\_evaluation\_report\_data.py \<path/to/network\_reports\_directory\> \<path/to/protobuf\_networks\_directory\> \<path/to/evaluation\_tables/save\_dir\>* where the *network\_reports\_directory* contains a *asnet\_eval\_report\_confx.html* file for each configuration. The \<protobuf\_networks\_directory\> contains a hierarchy similar to the *training\_data\_dir* where each domain has a subdirectory. These again contain a folder for each network configuration used containing a *prob\_name.log* file for each problem used during the evaluation. Note that these are automatically created by *build\_networks\_for\_eval.py* containing the network creation time.






# Neural Fast-Downward

This is an adaption of the [Fast-Downward planning system](http://www.fast-downward.org/) for experiments with neural networks and Fast-Downward. This version is in an very early development phase.
For questions, help or request converning this framework contact Patrick Ferber (patrick.ferber@unibas.ch)

Fast Downward is a domain-independent planning system.
For documentation and contact information see http://www.fast-downward.org/.

The following directories are not part of Fast Downward as covered by this
license:

* ./src/search/ext

For the rest, the following license applies:

```
Fast Downward is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Fast Downward is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
```


## == Setup ==

Those steps describe how you can set up the framework. Those steps may not be the only possible way, but were the easiest way for me. Especially in step 1 you might chose to use different versions from different sources.

0. Install required tools

    1. APT requirements
    
            sudo apt-get install python3-numpy python3-dev python3-wheel curl cmake
    
    2. Install bazel (this is the recommended compiler)
    
	    	echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee     /etc/apt/sources.list.d/bazel.list
	    	curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
	    	sudo apt-get update
	    	sudo apt-get install bazel -y
	    	sudo apt-get upgrade bazel -y
    
1. Obtain Tensorflow, Protobuf, and Eigen3

    1. Download the current Tensorflow source code and compile it
    
        	mkdir tensorflow
	        cd tensorflow
    	    git clone https://github.com/tensorflow/tensorflow
        	cd tensorflow
        	./configure
        	bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
        	
    2. Download the matching Protobuf and Eigen3 versions. Attentions: Their current versions from their webpages might not be compatible with the version of Tensorflow.
    
    	    tensorflow/contrib/makefile/download_dependencies.sh
    	    
    3. Build Protobuf
    
  	      	cd tensorflow/contrib/makefile/downloads/protobuf/
 	      	mkdir /tmp/proto
 	       	./autogen.sh
 	      	./configure --prefix=/tmp/proto/
 	     	make
  	     	make install
  	      
    4. Build Eigen
    
        	mkdir /tmp/eigen
        	cd ../eigen
    	    mkdir build_dir
    	    cd build_dir
     	   	cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../
        	make install
        	
    5. Create Library/Include directory structure. For the compilation of the framework CMake requires to know where to find the library files of all three packages and where to find their include files. You can store them anywhere but need to tell their positions CMake via environment variables.
    First we copy the required files into their target directories. I assume the current working directory is the first 'tensorflow' directory we created. Attention: Between different Tensorflow versions their directory structure has already changed. It might be necessary for you to adapt the steps below.
    
			cd ..
			mkdir protobuf
			cd protobuf
			mkdir include
			mkdir lib
			cd ..
			mkdir eigen
			cd eigen
			mkdir include
			mkdir lib
			cd ..
			cd tensorflow
			mkdir lib
			mkdir include
			cp tensorflow/bazel-bin/tensorflow/*.so lib
			cp -r tensorflow/bazel-genfiles/* include/
			cp -r tensorflow/third_party /include
			cp -r tensorflow/tensorflow/contrib/makefile/downloads/nsync include
			cp /tmp/proto/lib/libprotobuf.* ../protobuf/lib/
			cp -r /tmp/proto/include/* ../protobuf/include/
			cp -r /tmp/eigen/include/eigen3/* ../eigen/include

    Next, we set up the environment variables telling CMake where to find the 'lib' and 'include' directories.
    Fast-Downward is structured in a list of plugins. Every plugin defines on which other plugins
    it depends, which source code belongs to it, ON WHICH PACKAGES IT DEPENDS, and other stuff.
    The packages can be external dependencies, like Tensorflow, Protobuf, Eigen3. CMake will search
    in the default directories for those dependencies (their include files and their library
    files) and additionally takes hints from environment
    variables.
    
    For a Package PACKAGE, define one or up to all environment variables of:
    
	- PATH_PACKAGE32			= path where the include/ and lib/ dirs for the 32 bit build are stored
	- PATH_PACKAGE64			= path where the include/ and lib/ dirs for the 64 bit build are stored
	- PATH_PACKAGE_RELEASE32	= path where the include/ and lib/ dirs for the 32 bit release build are stored
	- PATH_PACKAGE_RELEASE64	= path where the include/ and lib/ dirs for the 64 bit release build are stored
	- PATH_PACKAGE_DEBUG32		= path where the include/ and lib/ dirs for the 32 bit debug build are stored
	- PATH_PACKAGE_DEBUG64		= path where the include/ and lib/ dirs for the 64 bit debug build are stored
	
	On building configuration <TYPE><BITS> it will check the variables in the following order:
	
    - PATH_PACKAGE_<TYPE><BITS>
	- PATH_PACKAGE_<BIT>
	- PATH_PACKAGE

	
2.  Compile Neural Fast Downward

    1. Compile with a build version which works for you. If Tensorflow could only be build with 64bit, build Fast-Downward as 64 bit (release64/debug64), if it could only be compiled as dynamic library, build a dynamic build (suffixed with dynamic), if both is the case use 'release64dynamic' or 'debug64dynamic'. If other restrictions apply use/define another build configuration.

3.  Have fun running your experiments

    If your compilation was successful, you can run some of the examples (currently only one) in the 'examples' directory to test if it is working.


## == Usage ==

For the usage I refer you to the documentation of [Fast Downward](http://www.fast-downward.org). Here are listed the relevant differences:

1. Case Sensitive Command Line

    Fast Downward converts the whole commandline input into lower case. This can be problematic when providing paths to your stored networks, node names in a computation graph or other case sensitive input. The new parsing allows inserting commands via FIRST_PART\COMMANDNAME\SECOND_PART into the commandline strings. Commands can modify the further parsing or the configuration. Currently implemented are:
   
    - \lower_case\	= parses the following chars of the configuration as lower case (DEFAULT)
    - \real_case\	= parses the following chars of the configuration in their original case
    - \lower_case\	= parses the following chars of the configuration as upper case
    - \default\		= resets all parsing modifications to their default value
    
    An example use-case is providing a file path. This could look like:
    
          ./fast-downward.py PROBLEM --search "eager_greedy(nh(network=mynetwork(path=\real_case\pathToMyTrainedNetwork\default\)))"


2. SearchEngine Transform Task

   SearchEngines can now also receive a task transformation (like previously heuristics) to run on a modified task. To use this provide the additional parameter 'transform', e.g.
    ./fast-downward.py PROBLEM --search "eager_greedy(ff(), transform=adapt_costs())"


3.  Register Heuristics

    Defined heuristics can be registered (happens on construction if the argument is provided and 
    automatically unregister on destruction or when manually unregistered or register is resetted) globally.
    This does not mean, they are invoked for every state, but this allows different components to
    access the heuristic without passing the reference everywhere around (this is used in the sampling
    search)


## == Currently implemented Features ==

1.  Structure for Networks

    The base class for networks is AbstractNetwork. Networks can produce arbitrary output.
    This makes is quite difficult how to design the inheritance structures such that a
    network which outputs data of type A and B has functions to access its A and B data
    AND let other classes use the network on possible A, B or A&B.
    
    Example:
    
       Let's say we have a base network class B and interaces HEURISTIC_VALUE, PREFERRED_ACTIONS
       (aka abstract classes with multiple inheritance) provide the getters for our 
       network class N subclass of B. Now a network heuristic could accept an object of type
       HEURISTIC_VALUE, evaluate it and access its heuristic value, but it could not access its
       preferred actions (and vice versa), because it is not possible to define objects which
       are of multiple class types (here interfaces HEURISTIC_VALUE and PREFERRED_ACTIONS).
    
    Therefore, the current approach says, every network (which shall be used by some arbitrary
    components) is a decendant of AbstractNetwork. For every type of output ANY IMPLEMENTED
    network can produce AbstractNetwork has the following methods:
    
    - bool is_OUTPUTTYPE()	- tells if the network produces output for this type
    - void verify_OUTPUTTYPE() - checks if network produces output for this type and stops
                                  execution if not. This shall be used by code which uses
                                  some networks in their initialization to check that the
                                  given network supports the needed outputs (e.g. the
                                  network heuristic checks that the given network produces a
                                  heuristic value)
    - TYPE get_OUTPUTTYPE()  - provides access to the last output of the given type the
                                  network has produced if able. If the network is not able
                                  to produce this kind of output, stop execution.

    Now components working with networks can simply accept AbstractNetwork objects and then
    use the verify method to check that all needed outputs can be produced by the given
    network. If you write a network which will only ever be used by ONE component, then
    you might ignore this and give this component directly an object of your network class.
    

2.  Sampling Search

    This is a search which samples states for a problem. The base problem is the one given
    to Fast-Downward. For each SamplingTechnique defined (as argument for the SamplingSearch)
    it samples a new derived problem of the base problem (or the base problem directly, if
    the SamplingTechnique does not perform any modifications). Then it starts for every
    derived problem a new search (what is run can be configured). The results of the search
    are sampled. There are currently three different kind of samplings which can be
    performed for a sampling run:

       - solution path		: sample the states and actions on the goal trajectory
       - other paths		: the search performed can add more paths to the variable paths in sampling_search.h (global variable) to store
       - all states		: samples all encountered states

    The format for every state in a path is as follows:
    
       state; action; successor state; heuristic value of state; (registered_heuristic = VALUE)*

    The format for states (from all states) is:
    
       state; parent action; predecessor state; heuristic value of state; (registered_heuristic = VALUE)*

    Additionally has every entry some Meta information <Meta ....> in the beginning of each line.
    The meta tag may contain:
    
       - problem_hash		= a hash of the base problem description(original pddl file)
       - modification_hash	= a hash of the modification done to obtain the derived problem
       - type				= 	O:entry from an optimal path, T:entry from other path, S: entry for some state
       - format				= format of the entry:
	   - FD					= Format used by FD (unchangable predicates are pruned)



## == Adding new Frameworks ==

If you want to add other frameworks (external dependencies), you have to declare them
in the CMake configuration. The easiest way is adding a package in the plugin description
for the Fast-Downward plugin which needs the dependency (dependencies can be multiple time
required and will be loaded only once). The packages are loaded via 'find_package(PACKAGE)'
of CMake, therefore, common packages SHOULD (never tested) need no further configuration
(e.g. boost), for other packages, you need to add a file 'FindPACKAGE.cmake' in 
'./src/cmake_modules' describing where to find the include/ and lib/ directories and doing
other needed stuff. Take a look at the other 'findPACKAGE.cmake' files there.


== Known Problems ==

1. Fast-Downward:

    1. During parsing Fast-Downward converts the arguments to lower case, this invalidates
    given file paths (to the trained networks) and other possible case-sensitive arguments (e.g. variable names within the computational graph of a Protobuf network). I am working on modifying this parsing.



2. Tensorflow

    1. Since I used Tensorflow, the compiled library has at least once changed its structure, needing a new configuration for the compilation (adding tensorflow_framework.so to the libraries and adding an include for 'nsync' files). The described procedure works with my version (around February 2018).
    
