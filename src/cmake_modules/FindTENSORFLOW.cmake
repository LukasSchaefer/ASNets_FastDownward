# Find the Tensorflow package. This includes the libraries and include
# files.
#
# Attention: Tensorflow is still changing alot in how to build its
# libraries (luckily getting easier) and what is the output of the
# builds. It might be that for your version, this script has to be
# adapted.
# Last time I checked the script tensorflow did not allow compiling
# a static library, thus, we have to use a hack to find the shared
# library even in static linking mode.
#
#
# This code defines the following variables:
#
#  TENSORFLOW_FOUND          - TRUE if all components are found.
#  TENSORFLOW_INCLUDE_DIRS   - Full paths to all include dirs.
#  TENSORFLOW_LIBRARIES      - Full paths to all libraries.
#
# Example Usages:
#  find_package(TENSORFLOW)
#
# The location of TENSORFLOW can be specified using the environment variable
# or cmake parameter PATH_TENSORFLOW. If different installations
# for 32-/64-bit versions and release/debug versions are available,
# they can be specified with
#   PATH_TENSORFLOW32
#   PATH_TENSORFLOW64
#   PATH_TENSORFLOW_RELEASE32
#   PATH_TENSORFLOW_RELEASE64
#   PATH_TENSORFLOW_DEBUG32
#   PATH_TENSORFLOW_DEBUG64
# More specific paths are preferred over less specific ones when searching
# for libraries and 64 bit over 32 bit.
#
# Note that the standard FIND_PACKAGE features are supported
# (QUIET, REQUIRED, etc.).

foreach(BITWIDTH 32 64)
    foreach(BUILDMODE "RELEASE" "DEBUG")
        set(TENSORFLOW_HINT_PATHS_${BUILDMODE}${BITWIDTH}
            ${PATH_TENSORFLOW_${BUILDMODE}${BITWIDTH}}
            $ENV{PATH_TENSORFLOW_${BUILDMODE}${BITWIDTH}}
            ${PATH_TENSORFLOW${BITWIDTH}}
            $ENV{PATH_TENSORFLOW${BITWIDTH}}
            ${PATH_TENSORFLOW}
            $ENV{PATH_TENSORFLOW}
        )
    endforeach()
endforeach()

if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
    set(TENSORFLOW_HINT_PATHS_RELEASE ${TENSORFLOW_HINT_PATHS_RELEASE32})
    set(TENSORFLOW_HINT_PATHS_DEBUG ${TENSORFLOW_HINT_PATHS_DEBUG32})
elseif(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(TENSORFLOW_HINT_PATHS_RELEASE ${TENSORFLOW_HINT_PATHS_RELEASE64})
    set(TENSORFLOW_HINT_PATHS_DEBUG ${TENSORFLOW_HINT_PATHS_DEBUG64})
else()
    message(WARNING "Bitwidth could not be detected, preferring 64-bit version of TENSORFLOW")
    set(TENSORFLOW_HINT_PATHS_RELEASE
        ${TENSORFLOW_HINT_PATHS_RELEASE64}
        ${TENSORFLOW_HINT_PATHS_RELEASE32}
    )
    set(TENSORFLOW_HINT_PATHS_DEBUG
        ${TENSORFLOW_HINT_PATHS_DEBUG64}
        ${TENSORFLOW_HINT_PATHS_DEBUG32}
    )
endif()


find_path(TENSORFLOW_INCLUDE
    NAMES core
    HINTS ${TENSORFLOW_HINT_PATHS_RELEASE} ${TENSORFLOW_HINT_PATHS_DEBUG}
    PATH_SUFFIXES include
)

# A new version of tensorflow required this directory too
find_path(TENSORFLOW_NSYNC
    NAMES nsync.h
    HINTS ${TENSORFLOW_HINT_PATHS_RELEASE} ${TENSORFLOW_HINT_PATHS_DEBUG}
    PATH_SUFFIXES include/nsync/public
)

set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE} ${TENSORFLOW_NSYNC})

# Find tensorflow_{cc, framework} libraries.
# 1. HACK: Currently TF only allows shared library compilations. We have
# to add .so to the library search suffixes and remove afterwards
set(TMP_LIB_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} ".so")

find_library(TENSORFLOW_CC_LIBRARY_RELEASE
    NAMES tensorflow_cc
    HINTS ${TENSORFLOW_HINT_PATHS_RELEASE}
    PATH_SUFFIXES lib
)

find_library(TENSORFLOW_FRAMEWORK_LIBRARY_RELEASE
    NAMES tensorflow_framework
    HINTS ${TENSORFLOW_HINT_PATHS_RELEASE}
    PATH_SUFFIXES lib
)

find_library(TENSORFLOW_CC_LIBRARY_DEBUG
    NAMES tensorflow_cc
    HINTS ${TENSORFLOW_HINT_PATHS_DEBUG}
    PATH_SUFFIXES lib
)

find_library(TENSORFLOW_FRAMEWORK_LIBRARY_DEBUG
    NAMES tensorflow_framework
    HINTS ${TENSORFLOW_HINT_PATHS_DEBUG}
    PATH_SUFFIXES lib
)


# 1. HACK: If the framework library is missing skip it, because older TF
# versions do not have it.
set(TENSORFLOW_LIBRARIES
    optimized ${TENSORFLOW_CC_LIBRARY_RELEASE}
    debug ${TENSORFLOW_CC_LIBRARY_DEBUG})

if(TENSORFLOW_FRAMEWORK_LIBRARY_RELEASE)
    list(APPEND TENSORFLOW_LIBRARIES optimized ${TENSORFLOW_FRAMEWORK_LIBRARY_RELEASE})
endif()
if(TENSORFLOW_FRAMEWORK_LIBRARY_DEBUG)
    list(APPEND TENSORFLOW_LIBRARIES debug ${TENSORFLOW_FRAMEWORK_LIBRARY_DEBUG})
endif()


# 1. HACK: undo added lib search suffixes
set(CMAKE_FIND_LIBRARY_SUFFIXES ${TMP_LIB_SUFFIXES})


# Check for consistency and handle arguments like QUIET, REQUIRED, etc.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TENSORFLOW
    REQUIRED_VARS TENSORFLOW_INCLUDE_DIRS TENSORFLOW_LIBRARIES
)

# Do not show internal variables in cmake GUIs like ccmake.
mark_as_advanced(TENSORFLOW_INCLUDE_DIRS
                 TENSORFLOW_LIBRARY_RELEASE TENSORFLOW_LIBRARY_DEBUG
                 TENSORFLOW_LIBRARIES
                 TENSORFLOW_HINT_PATHS_RELEASE TENSORFLOW_HINT_PATHS_DEBUG)
