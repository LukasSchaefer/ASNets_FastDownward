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



This version is an adaption by Patrick Ferber (patrick.ferber@unibas.ch) for experiments 
with neural networks and Fast-Downward. This version is still in a very early development phase.

== Setup ==
Setup steps (TODO extend)

0. Install bazel (this is the recommended compiler), and the requirements for Fast-Downward

1. Obtain Tensorflow, Protobuf, and Eigen3
1.1 Download the source code for all three projects named above in a custom directory
1.2 Compile Tensorflow as C++ library. The result should contain a directory which contains
    a subdirectory lib/ for the library files and include/ for files to include
1.3 Compile Protobuf. There was a time, when the current Protobuf version was not compatible
    with the current Tensorflow version. This is because Tensorflow downloads a different version
    which it use during compilation. Somewhere within the Tensorflow directories or the bazel cache
    is the corret downloaded version './include/contrib/makefile/downloads/protobuf' looks as it
    could be compiled. Again the result should be having in the same directory a lib/ and an 
    include/ directory.
1.4 Store somewhere Eigen3. It should have the structure 'include/eigen3/unsupported' within it.

REMARK: The last time, I did this, Tensorflow could only be compiled with 64 bit and as dynamic
        library. If this is still the same, do not worry, this is sufficient.


2.  Tell CMake where to find Tensorflow, Protobuf, Eigen3.
2.1 Fast-Downward is structured in a list of plugins. Every plugin defines on which other plugins
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
	
	On building configuration <TYPE><BITS> it will look into the variables in the following order:
		 - PATH_PACKAGE_<TYPE><BITS>
		 - PATH_PACKAGE_<BIT>
		 - PATH_PACKAGE
	
    
3.  Compile Neural Fast-Downward
3.1 Compile with a build version which works for you. If Tensorflow could only be build with 64bit,
    build Fast-Downward as 64 bit (release64/debug64), if it could only be compiled as dynamic
    library, build a dynamic build (suffixed with dynamic), if both is the case use 'release64dynamic'
    or 'debug64dynamic'. If other restrictions apply use/define another build configuration.

4.  Have fun running your experiments
4.1 If your compilation was successful, you can run some of the examples (currently only one) in the
    'examples' directory to test if it is working.
    REMARK: Actually, the example is currently not running, because of issue 1.1.


== Usage ==
The command line usage of this fork is nearly the same as for the Fast-Downward master.

1.  Commandline Argument Cases:
    In Fast-Downward the search engine configuration is converted to lower case which makes
    providing case sensitive string difficult or impossible.
    The new parsing allows inserting commands via FIRST_PART\COMMANDNAME\SECOND_PART. Commands
    can modify the further parsing or the configuration. Currently implemented are:
      \lower_case\	= parses the following chars of the configuration as lower case (DEFAULT)
      \real_case\	= parses the following chars of the configuration in their original case
      \lower_case\	= parses the following chars of the configuration as upper case
      \default\		= resets all parsing modifications to their default value
    
    An example use-case is providing a file path. This could look like:
      ./fast-downward.py PROBLEM --search "eager_greedy(nh(network=mynetwork(path=\real_case\pathToMyTrainedNetwork\default\)))"


2.  SearchEngine Transform Task
    SearchEngines can now also receive a task transformation (like previously heuristics) to run on a
    modified task. To use this provide the additional parameter 'transform', e.g.
      ./fast-downward.py PROBLEM --search "eager_greedy(ff(), transform=adapt_costs())"


3.  Register Heuristics
    Defined heuristics can be registered (happens on construction if the argument is provided and 
    automatically unregister on destruction or when manually unregistered or register is resetted) globally.
    This does not mean, they are invoked for every state, but this allows different components to
    access the heuristic without passing the reference everywhere around (this is used in the sampling
    search)


== Currently implemented Features ==
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



== Adding new Frameworks ==
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
1.1  During parsing Fast-Downward converts the arguments to lower case, this invalidates
     given file paths (to the trained networks) and other possible case-sensitive arguments
     (e.g. variable names within the computational graph of a Protobuf network).
     I am working on modifying this parsing.



2. Tensorflow
2.1  Since I used Tensorflow, the compiled library has at least once changed its structure,
     needing a new configuration for the compilation (adding tensorflow_framework.so to the
     libraries and adding an include for 'nsync' files). The described procedure works with
     my version (around February 2018).
    
