To run CI in docker use `docker-run-ci` and give build parameters, eg:

```
$ ./ci/docker-run-ci TA_PYTHON=OFF ENABLE_CUDA=ON all check
Removing previous build container: andrey.tiledarray.build

Running new build of /home/andrey/github/tiledarray on andrey.tiledarray.build
* Use CTRL-p CTRL-q to dettach
* To reattach use: docker start -a -i andrey.tiledarray.build
...
```

This builds targets `all check` on current Git branch in a docker container named `${USER}.$(basename $PWD).build` 

