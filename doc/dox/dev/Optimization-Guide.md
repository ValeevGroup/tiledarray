#  Optimizing TiledArray {#Optimization-Guide}
The overall efficiency of a TiledArray-based program is controlled by a number factors. Some of these are determined by what the program does, e.g. tensor contractions vs. vector operations, or the extents and shapes of `DistArray`s. Some factors, such as tile size used to represent dense tensors, can be controlled and tuned by the developer. Lastly, some factors depend on the runtime parameters and hardware traits, such as the number and size of MADNESS communication buffers, various runtime parameters of the MPI library, etc.

# Efficiency vs. Traits of `DistArray`s

# Runtime parameters

## MADWorld Runtime

These parameters control the behavior of the MADNESS parallel runtime:
* `MAD_NUM_THREADS` -- Sets the total number of threads to be used
by MADWorld tasks.  When running with just one process all threads are
devoted to computation (1 main thread with the remainder in the thread
pool).  When running with multiple MPI processes, one of the threads
is devoted to communication. [Default = number of cores reported by ]
* `MAD_BUFFER_SIZE` -- [Default=1.5MB]
* `MAD_RECV_BUFFERS` -- [Default=128]

## MPI

## CUDA

In addition to [the environment variables that control the CUDA runtime behavior](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars), several environment variables control specifically the execution of TiledArray on CUDA devices:
* `TA_CUDA_NUM_STREAMS` -- The number of [CUDA streams](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf) used to execute tasks on each device. Each stream can be viewed as a thread in a threadpool, with tasks in a given stream executing in order, but each stream executing independently of others. For small tasks this may need to be increased. [Default=3]
* `CUDA_VISIBLE_DEVICES` -- This CUDA runtime environment variable is queried by TiledArray to determine whether CUDA devices on a multi-GPU node have been pre-mapped to MPI ranks.
  * By default (i.e. when # of MPI ranks on a node <= # of _available_ CUDA devices) TiledArray will map 1 device (in the order of increasing rank) to each MPI rank.
  * If # of available CUDA devices < # of MPI ranks on a node _and_ `CUDA_VISIBLE_DEVICES` is set TiledArray will assume that the user mapped the devices to the MPI ranks appropriately (e.g. using a resource manager like `jsrun`) and only checks that each rank has access to 1 CUDA device.
