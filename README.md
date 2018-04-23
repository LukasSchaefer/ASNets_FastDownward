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
    it depends, which source code belongs to it, ONE WHICH PACKAGES IT DEPENDS, and other stuff.
    The packages can be external dependencies, like Tensorflow, Protobuf, Eigen3. CMake will search
    in the default directories for those dependencies and additionally takes hints from environment
    variables.
    For a Package PACKAGE, define one or up to all environment variables of:
	-   PATH_PACKAGE32		= path where the include/ and lib/ dirs for the 32 bit build are stored
	-   PATH_PACKAGE64		= path where the include/ and lib/ dirs for the 64 bit build are stored
	-   PATH_PACKAGE_RELEASE32	= path where the include/ and lib/ dirs for the 32 bit release build are stored
	-   PATH_PACKAGE_RELEASE64	= path where the include/ and lib/ dirs for the 64 bit release build are stored
	-   PATH_PACKAGE_DEBUG32	= path where the include/ and lib/ dirs for the 32 bit debug build are stored
	-   PATH_PACKAGE_DEBUG64	= path where the include/ and lib/ dirs for the 64 bit debug build are stored
    
3.  Compile Neural Fast-Downward
3.1 Compile with a build version which works for you. If Tensorflow could only be build with 64bit,
    build Fast-Downward as 64 bit (release64/debug64), if it could only be compiled as dynamic
    library, build a dynamic build (suffixed with dynamic), if both is the case use 'release64dynamic'
    or 'debug64dynamic'. If other restrictions apply use/define another build configuration.

4.  Have fun running your experiments
4.1 If your compilation was successful, you can run some of the examples (currently only one) in the
    'examples' directory to test if it is working.
    REMARK: Actually, the example is currently not running, because of issue 1.1.



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
    
