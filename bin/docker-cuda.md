# Intro
These notes describe how to build TiledArray with CUDA support enabled within the latest nvidia/cuda Docker image (https://hub.docker.com/r/nvidia/cuda/). This is useful for experimentation and/or provisioning computational results (e.g. for creating supplementary info for a journal article). If you want to use Docker to run/debug Travis-CI jobs, see [docker-travis.md](docker-travis.md)

# Using
These notes assume that Docker 19.03 and NVIDIA Container Toolkit (https://github.com/NVIDIA/nvidia-docker) are installed on your machine and that you start at the top of the TiledArray source tree.

## Create/build Docker TA/CUDA image
1. Create a Docker image: `bin/docker-cuda-build.sh`
2. Run a container using the newly created image: `docker run --privileged -i -t --rm tiledarray-cuda-dev:latest bash -l`

## Notes
- Important locations:
  - source: `/usr/local/src/tiledarray`
  - build: `/usr/local/src/tiledarray/build`
  - install: `/usr/local`
