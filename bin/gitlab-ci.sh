#!/bin/bash

echo "$ $0 $@"

if [ "$#" -lt 1 ]; then
    echo "CMake project directory argument required"
    exit 1
fi

project_dir=$1
shift

cmake_args="-DTA_BUILD_UNITTEST=TRUE"

for arg in $@; do
    cmake_args+=" -D$arg"
done

cmake_build_target="cmake --build . --target "

set -e
set -x

# to run OpenMPI in docker as root
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_btl_vader_single_copy_mechanism="none"

cmake $project_dir $cmake_args

$cmake_build_target tiledarray
$cmake_build_target examples
$cmake_build_target ta_test
$cmake_build_target check

