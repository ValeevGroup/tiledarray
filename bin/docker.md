# Intro
These notes describe how to build TiledArray within the latest phusion (https://github.com/phusion/baseimage-docker) Docker image. This is useful for experimentation and/or provisioning computational results (e.g. for creating supplementary info for a journal article).

# Using
These notes assume that Docker is installed on your machine and that you start at the top of the TiledArray source tree.

## Create/build Docker TA image
1. Create a Docker image: `bin/docker-build.sh`
2. Run a container using the newly created image: `docker run --privileged -i -t --rm tiledarray-dev:latest /sbin/my_init -- bash -l`

## Notes
- Important locations:
  - source: `/usr/local/src/tiledarray`
  - build: `/usr/local/src/tiledarray/build`
  - install: `/usr/local`
