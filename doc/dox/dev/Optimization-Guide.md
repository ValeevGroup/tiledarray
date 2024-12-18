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

## GPU/Device compute runtimes

In addition to the environment variables that control the runtime behavior of [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) and [HIP/ROCm](https://rocm.docs.amd.com/en/latest/search.html?q=environment+variables), several environment variables control specifically the execution of TiledArray on compute devices:
* `TA_DEVICE_NUM_STREAMS` -- The number of [compute streams](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf) used to execute tasks on each device. Each stream can be viewed as a thread in a threadpool, with tasks in a given stream executing in order, but each stream executing independently of others. For small tasks this may need to be increased. In addition stream for compute tasks TiledArray also creates 2 dedicated streams for data transfers to/from each device. [Default=3]
* `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` -- These runtime environment variables are can be used to map CUDA/HIP devices, respectively, on a multi-device node to MPI ranks. It is usually the responsibility of the resource manager to control this mapping, thus normally it should not be needed. By default TiledArray will assign compute devices on a multidevice node round robin to each MPI rank.
