#!/bin/sh

# this script builds a TiledArray/CUDA docker image

# to run bash in the image: docker run --privileged -i -t tiledarray-cuda-dev:latest -- bash -l
# see docker.md for further instructions
# locations:
#   - source dir: /usr/local/src/tiledarray
#   - build dir: /usr/local/src/tiledarray/build
#   - installed headers dir: /usr/local/include/tiledarray
#   - installed libraries dir: /usr/local/lib, e.g. /usr/local/lib/libtiledarray.a

export CMAKE_VERSION=3.17.0

##############################################################
# make Dockerfile, if missing
cat > Dockerfile << END
FROM nvidia/cuda:latest

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# update the OS
RUN apt-get update && apt-get upgrade -y -o Dpkg::Options::="--force-confold"

# build TiledArray
# 1. basic prereqs
RUN apt-get update && apt-get install -y python3 ninja-build liblapacke-dev liblapack-dev mpich libboost-dev libeigen3-dev git wget libboost-serialization-dev libunwind-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# 2. recent cmake
RUN CMAKE_URL="https://cmake.org/files/v${CMAKE_VERSION%.[0-9]}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz" && wget --no-check-certificate -O - \$CMAKE_URL | tar --strip-components=1 -xz -C /usr/local
ENV CMAKE=/usr/local/bin/cmake
# 3. download and build TiledArray
RUN cd /usr/local/src && git clone --depth=1 https://github.com/ValeevGroup/tiledarray.git && cd /usr/local/src/tiledarray && mkdir build && cd build && \$CMAKE .. -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && \$CMAKE --build . --target tiledarray && \$CMAKE --build . --target examples && \$CMAKE --build . --target install

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
END

function clean_up {
  rm -f Dockerfile
  exit
}

trap clean_up SIGHUP SIGINT SIGTERM

##############################################################
# build a cuda-dev image
docker build -t tiledarray-cuda-dev .

##############################################################
# extra admin tasks, uncomment as needed
# docker save tiledarray-cuda-dev | bzip2 > tiledarray-cuda-dev.docker.tar.bz2

##############################################################
# done
clean_up
