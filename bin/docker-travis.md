# Intro
These notes describe how to build TiledArray within the latest Travis-CI Docker image. This is useful for debugging Travis-CI jobs on your local machine.
# Using
These notes assume that Docker is installed on your machine and that you start at the top of the TiledArray source tree.

## Create/build Docker Travis image
1. Create a Travis-CI docker image: `cd bin; ./docker-travis-build.sh`
2. Run a container using the newly created image: `docker run -it tiledarray-travis-debug bash -l`
3. `cd /home/travis/_build`
4. Configure the job to use the appropriate compiler, compiler version, and debug/release build type:
  * `export BUILD_TYPE=B`, where `B` is `Debug` or `Release`.
  * If want to use GNU C++ compiler (gcc):
    * `export GCC_VERSION=VVV` where `VVV` should be the GCC version to be used. The currently valid values are `7`, `8` and `9`.
    * `export CXX=g++`
  * If want to use Clang C++ compiler (clang++):
    * `export GCC_VERSION=8`
    * `export CLANG_VERSION=VVV` where `VVV` should be the Clang version to be used. The currently valid values are `7`, `8`, and `9`.
    * `export CXX=clang++`
    * `apt-get update && apt-get install libc++-${CLANG_VERSION}-dev libc++abi-${CLANG_VERSION}-dev`
5. Build prerequisites (MPICH, MADNESS, ScaLAPACK), TiledArray, and run tests: `./build.sh`

## Notes
* According to [Travis-CI docs](https://docs.travis-ci.com/user/reference/overview/) you want to configure your Docker to run containers with 2 cores and 7.5 GB of RAM to best match the production environment.
* If you plan to use this container multiple times it might make sense to take a snapshot at this point to avoid having to recompile the prerequisites each and every time. Store it as a separate image, e.g. `docker commit container_id tiledarray-travis-debug:clang-debug`, where `container_id` can be found in the output of `docker ps`. Next time to start debugging you will need to pull updates to the TiledArray source (do `cd /home/travis/build/ValeevGroup/tiledarray && git pull`), then execute step 2 with the new image name, execute step 3, and go directly to step 6.
* To install `gdb` execute `apt-get update && apt-get install gdb`. Also, it appears that to be able to attach `gdb` or any other debugger to a running process you must run the Docker container in privileged mode as `docker run --privileged -it tiledarray-travis-debug:clang-debug bash -l`.
* To debug parallel jobs you want to launch jobs in a gdb in an xterm. To run xterm you need to ssh into the container. To start an ssh server in the container do this:
  * Connect sshd's port of the container (22) to an unprivileged port (say, 2222) of the host: `docker run -p 127.0.0.1:2222:22 --privileged -it tiledarray-travis-debug:clang-debug bash -l`
  * Generate host keys: `ssh-keygen -A`
  * Create a root password: `passwd` and follow prompts. No need to be fancy: security is not a concern here, but `passwd` will not accept an empty password. N.B. This is easier than setting up a pubkey login, so don't bother with that.
  * Edit `/etc/ssh/sshd_config` and allow root to log in by ensuring that `PermitRootLogin` and `PasswordAuthentication` are set to `yes`.
  * Start ssh server: `/etc/init.d/ssh start`
  * (optional) To launch gdb in xterm windows: `apt-get update && apt-get install install`
  * You should be able to log in from an xterm on the host side: `ssh -Y -p 2222 root@localhost`
