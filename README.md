# Neural Fast Downward

This is an adaption of the [Fast Downward planning system](http://www.fast-downward.org/) for experiments with neural networks and Fast Downward. This version is in an very early development phase.
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
    
		echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
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
	-   PATH_PACKAGE32		= path where the include/ and lib/ dirs for the 32 bit build are stored
	-   PATH_PACKAGE64		= path where the include/ and lib/ dirs for the 64 bit build are stored
	-   PATH_PACKAGE_RELEASE32	= path where the include/ and lib/ dirs for the 32 bit release build are stored
	-   PATH_PACKAGE_RELEASE64	= path where the include/ and lib/ dirs for the 64 bit release build are stored
	-   PATH_PACKAGE_DEBUG32	= path where the include/ and lib/ dirs for the 32 bit debug build are stored
	-   PATH_PACKAGE_DEBUG64	= path where the include/ and lib/ dirs for the 64 bit debug build are stored
	
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
   Fast Downward converts the whole commandline input into lower case. This can be problematic when providing paths to your stored networks, node names in a computation graph or other case sensitive input. The new parsing allows inserting commands via FIRST_PART\COMMANDNAME\SECOND_PART into the commandline strings. Commands
   can modify the further parsing or the configuration. Currently implemented are:
      \lower_case\	= parses the following chars of the configuration as lower case (DEFAULT)
      \real_case\	= parses the following chars of the configuration in their original case
      \lower_case\	= parses the following chars of the configuration as upper case
      \default\		= resets all parsing modifications to their default value
    
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
       bool is_OUTPUTTYPE()	- tells if the network produces output for this type
       void verify_OUTPUTTYPE() - checks if network produces output for this type and stops
                                  execution if not. This shall be used by code which uses
                                  some networks in their initialization to check that the
                                  given network supports the needed outputs (e.g. the
                                  network heuristic checks that the given network produces a
                                  heuristic value)
      TYPE get_OUTPUTTYPE()     - provides access to the last output of the given type the
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
       - other paths		: the search performed can add more paths to the variable
				  paths in sampling_search.h (global variable) to store
       - all states		: samples all encountered states

    The format for every state in a path is as follows:
       state; action; successor state; heuristic value of state; (registered_heuristic = VALUE)*

    The format for states (from all states) is:
       state; parent action; predecessor state; heuristic value of state; (registered_heuristic = VALUE)*

    Additionally has every entry some Meta information <Meta ....> in the beginning of each line.
    The meta tag may contain:
       problem_hash		= a hash of the base problem description(original pddl file)
       modification_hash	= a hash of the modification done to obtain the derived problem
       type			= O:entry from an optimal path, T:entry from other path,
				  S: entry for some state
       format			= format of the entry:
		FD		= Format used by FD (unchangable predicates are pruned)



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
    
