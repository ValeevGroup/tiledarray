#!/bin/bash

# this script builds a 'Bionic' env docker image used by Travis-CI for TiledArray project
#
# to run bash in the image: docker run -it tiledarray-travis-debug bash -l
# see docker-travis.md for further instructions
# N.B. relevant locations:
#   - source dir: /home/travis/build/ValeevGroup/tiledarray (TRAVIS_BUILD_DIR env in Travis jobs)
#   - build dir: /home/travis/_build
#   - install dir: /home/travis/_install

# this is where in the container file system Travis-CI "starts"
export TRAVIS_BUILD_TOPDIR=/home/travis/build
export DIRNAME=`dirname $0`
export ABSDIRNAME=`pwd $DIRNAME`

##############################################################
# make a script to download all prereqs and clone TiledArray repo
setup=setup.sh
cat > $setup << END
#!/bin/sh
curl -sSL "http://apt.llvm.org/llvm-snapshot.gpg.key" | apt-key add -
echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal main" | tee -a /etc/apt/sources.list > /dev/null
apt-add-repository -y "ppa:ubuntu-toolchain-r/test"
apt-get -yq update >> ~/apt-get-update.log
apt-get -yq --no-install-suggests --no-install-recommends --force-yes install g++-7 g++-8 g++-9 gfortran-7 gfortran-8 gfortran-9 libblas-dev liblapack-dev liblapacke-dev libtbb-dev clang-8 clang-9 cmake cmake-data libclang1-9 graphviz fonts-liberation \
python3 python3-pip python3-pytest python3-numpy
mkdir -p ${TRAVIS_BUILD_TOPDIR}
cd ${TRAVIS_BUILD_TOPDIR}
git clone https://github.com/ValeevGroup/tiledarray.git ${TRAVIS_BUILD_TOPDIR}/ValeevGroup/tiledarray
END
chmod +x $setup

##############################################################
# make a script to build all extra prereqs once in the container
build=build.sh
cat > $build << END
#!/bin/sh
cd /home/travis/_build
export BUILD_PREFIX=/home/travis/_build
export INSTALL_PREFIX=/home/travis/_install
export TRAVIS_BUILD_DIR=${TRAVIS_BUILD_TOPDIR}/ValeevGroup/tiledarray
export TRAVIS_EVENT_TYPE=cron
export TRAVIS_OS_NAME=linux
\${TRAVIS_BUILD_DIR}/bin/build-\$TRAVIS_OS_NAME.sh
END
chmod +x $build

##############################################################
# make Dockerfile
cat > Dockerfile << END
# Travis default 'Focal' image
FROM travisci/ci-ubuntu-2004:packer-1609444725-e5de6974

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# create source, build, and install dirs
RUN mkdir -p /home/travis/_build
RUN mkdir -p /home/travis/_install

# install prereqs
ADD $setup /home/travis/_build/$setup
RUN /home/travis/_build/$setup

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# copy travis scripts
ADD $build /home/travis/_build/$build

# for further info ...
RUN echo "\e[92mDone! For info on how to use the image refer to $ABSDIRNAME/docker-travis.md\e[0m"

END

function clean_up {
  rm -f $setup $build Dockerfile
  exit
}

trap clean_up SIGHUP SIGINT SIGTERM

##############################################################
# build a dev image
docker build -t tiledarray-travis-debug .

##############################################################
# extra admin tasks, uncomment as needed

##############################################################
# done
clean_up
