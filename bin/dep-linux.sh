#! /bin/sh

# Exit on error
set -ev

# Add repository for a newer version GCC
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y

# Update package list
sudo apt-get update -qq

# Install updated gcc compilers
sudo apt-get install -qq -y gcc-$GCC_VERSION g++-$GCC_VERSION
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
fi

# Print compiler information
$CC --version
$CXX --version

# Install packages
sudo apt-get install -qq -y cmake libblas-dev liblapack-dev mpich2 libtbb-dev libeigen3-dev libboost1.48-dev
