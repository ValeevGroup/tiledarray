# Managing Travis Builds {#Travis-CI-Administration-Notes}

## Basic Facts
* Travis CI configuration is in file `.travis.yml`, and build scripts are in `bin/build-*linux.sh`. Only Linux builds are currently supported.
* `BUILD_TYPE=Debug` jobs build and install MADNESS separately, before building TiledArray' `BUILD_TYPE=Release` jobs build MADNESS as a step of the TiledArray build.
* MPICH and (`BUILD_TYPE=Debug` only) MADNESS installation directories are _cached_. **Build scripts only verify the presence of installed directories, and do not update them if their configuration (e.g. static vs. shared, or code version) has changed. _Thus it is admin's responsibility to manually wipe out the cache on a per-branch basis_.** It is the easiest to do via the Travis-CI web interface (click on 'More Options' menu at the top right, select 'Caches', etc.).
* Rebuilding cache of prerequisites may take more time than the job limit (50 mins at the moment), so rebuilding cache can take several attempts. Since Travis-CI does not support forced cache updates (see e.g. https://github.com/travis-ci/travis-ci/issues/6410) if the job looks like it's going to time out we report success to Travis just so that it will store cache. __Thus jobs that timed out will be falsely reported as successful (rather than errored)!__ When rebuilding cache it may be necessary to manually restart some build jobs to make sure that cache rebuild is complete (or, just to be sure, restart the whole __build__ one time just to be sure all caches have been rebuilt). Again: this is only relevant when rebuilding caches (i.e. <5% of the time), otherwise there should be no need to restart jobs manually.

# Debugging Travis-CI jobs

## Local debugging

Follow the instructions contained in [docker-travis.md](https://github.com/ValeevGroup/tiledarray/blob/master/bin/docker-travis.md) .
