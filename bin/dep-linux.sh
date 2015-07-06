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

# Install CMAKE
curl -O http://www.cmake.org/files/v2.8/cmake-2.8.8.tar.gz
tar -xzf cmake-2.8.8.tar.gz
cd ./cmake-2.8.8
./configure --prefix=/usr/local
make -j2
sudo make install

# Install packages
sudo apt-get install -qq -y libblas-dev liblapack-dev mpich2 libtbb-dev libeigen3-dev libboost1.48-dev
