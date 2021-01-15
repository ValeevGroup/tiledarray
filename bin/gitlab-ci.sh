#!/bin/bash

echo "$ $0 $@"

if [ "$#" -lt 1 ]; then
    echo "CMake project directory argument required"
    exit 1
fi

project_dir=$1
shift

targets=""
cmake_args=""

for arg in "$@"; do
    #echo $arg
    case $arg in
         *=*) cmake_args+=" \"-D$arg\"" ;;
         *)   targets+=" $arg" ;;
    esac
done

echo "CMake args: $cmake_args"
echo "Build targets: $targets"
echo ""

set -e
set -x

# to run OpenMPI in docker as root
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_btl_vader_single_copy_mechanism="none"

eval "cmake $project_dir $cmake_args"

for target in $targets; do
    echo "Building target $target"
    eval "cmake --build . --target $target"
done
