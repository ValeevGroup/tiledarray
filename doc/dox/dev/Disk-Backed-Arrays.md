# Disk-Backed Arrays {#Disk-Backed-Arrays}

A `TiledArray::DiskArray<Tile, Policy>` is an array whose tiles are kept in scratch files and read when they are needed. It trades I/O for memory: an array that is alive but rarely used occupies (almost) no memory while it is on disk.

A disk-backed array is an ordinary `DistArray` whose tile type is `TiledArray::DiskTile<Tile>`, a [lazy tile](@ref User-Defined-Lazy-Tiles) that evaluates to `Tile`. It can appear wherever an expression accepts an array argument, and it can be the target of an assignment, whole or by block.

The rule that governs where the data is: **a `DiskTile` is on disk while it is an element of an array, on the process that owns that element, and in memory while it is on its own** (just made from a tile, or in transit to or from another process).

## When to use one

* For large arrays that are read occasionally: integrals or intermediates built once and consumed by a few later steps.
* For arrays assembled piece by piece, e.g. by a loop over batches, that would not fit in memory whole.
* Not for arrays that are updated in place or read by nearly every operation. A tile on disk is never modified, only replaced, and every use of a tile is a read from disk.

## Moving an array to disk and back

```cpp
TA::TArrayD a = ...;

// copy to disk: an ordinary assignment to a disk-backed array
TA::DiskArray<TA::Tensor<double>, TA::DensePolicy> d;
d("i,j") = a("i,j");
a = TA::TArrayD{};  // release the memory (once the tiles are written)

// use it as an argument: tiles are read as the expression needs them
TA::TArrayD c;
c("i,k") = d("i,j") * b("j,k");

// bring it back, whole ...
TA::TArrayD a2;
a2("i,j") = d("i,j");
// ... or by tile block, which reads only the tiles of the block
TA::TArrayD blk;
blk("i,j") = d("i,j").block({0, 0}, {2, 2});
```

The assignment does not modify its argument; the memory is released when the original array (and any other reference to its tiles) is dropped. At no point are two copies of the array in memory: each tile is written as the assignment produces it (see "Memory" below).

Nested (tensor-of-tensors) arrays are supported, with the usual nested annotations (`d("i,j;a,b")`).

## Assigning to a disk-backed array

A disk-backed array can be the result of an expression:

```cpp
TA::DiskArray<TA::Tensor<double>, TA::SparsePolicy> d;
d("i,k") = a("i,j") * b("j,k");  // each tile goes to disk as it is produced
```

and it can be assembled by assigning to its blocks, starting from an initialized array with an all-zero (sparse) shape:

```cpp
TA::Tensor<float> zero(trange.tiles_range(), 0.0f);
TA::DiskArray<TA::Tensor<double>, TA::SparsePolicy> d(
    world, trange, TA::SparseShape<float>(zero, trange, true));
for (auto [lo, hi] : batches) {   // tile bounds of each batch
  TA::TSpArrayD part = compute_batch(lo, hi);   // in memory
  d("i,j,k").block(lo, hi, TA::preserve_lobound) = part("i,j,k");
}
```

Each assignment extends the shape by the batch's nonzero tiles and writes only the batch's tiles; tiles assigned earlier stay where they are. The array is readable between batches. Assigning to a block again replaces its tiles, and the space of the replaced tiles is reused once nothing refers to them any more (after the next `fence`). Assembling by block requires a sparse policy: a dense array would need every tile to exist beforehand.

Batches that must be accumulated into the same tiles cannot be assembled on disk this way: assignment replaces, it does not add.

## Building an array tile by tile

To produce an array from tiles you generate yourself, set them: a tile goes to disk when it is made an element of an array.

```cpp
TA::DiskArray<TA::Tensor<double>, TA::SparsePolicy> d(
    world, trange, shape /*, pmap */);
for (auto it = d.pmap()->begin(); it != d.pmap()->end(); ++it)
  if (!d.is_zero(*it))
    // written at once; not kept in memory
    d.set(*it, TA::DiskTile<TA::Tensor<double>>(make_tile(*it)));
world.gop.fence();
```

Each process should set the tiles it owns: a tile set on another process is sent to its owner, which writes it. `set` may be called from several threads (for different tiles). With a sparse policy the shape is fixed when the array is constructed.

## Where the data lives

Each array has one scratch file on each process that owns some of its tiles, holding the tiles of that array owned by that process (and, until the next `fence`, tiles that an assignment has just replaced). The file is created when the process stores its first tile and is released when the array, and every copy of its tiles, is gone. An array that replaces another by block assignment continues the same file. Space of tiles that are gone is reused by later writes. A tile that is already on disk in another array's file and is made an element of this one (`e.set(i, d.find(i))`) is copied.

Scratch files are placed in the scratch directory, which should be on a filesystem local to the node. The scratch directory is, in order of precedence:

1. the directory given to `TiledArray::set_disk_scratch_dir()`;
2. the value of the `TA_DISK_SCRATCH_DIR` environment variable;
3. the system temporary directory.

A scratch directory that does not exist or cannot be written to is reported early: `set_disk_scratch_dir()` throws, and `TiledArray::initialize()` throws if `TA_DISK_SCRATCH_DIR` names one. Both check on the calling process only, so in a multi-node run every process should make the call (as every process calls `initialize()`).

**Check where the system temporary directory lives before relying on the default.** On many Linux systems, and on most compute nodes, `/tmp` is a memory-backed filesystem (tmpfs): a "disk-backed" array placed there still occupies memory, just not the process's own, and nothing is gained. Point the scratch directory at a node-local disk. It should not be on a network filesystem either: besides being slow, some (NFS) keep a renamed copy of an unlinked open file, which would survive a crash.

Each disk-backed array holds one open file per process for as long as it lives, so the number of disk-backed arrays alive at once is bounded by the process's limit on open files.

A scratch file belongs to the process that created it: it has no name (it is created without one where the operating system can do that, and is otherwise unlinked as soon as it is created), it is not inherited by a child process, and its space is reclaimed when the array is destroyed or when the process ends, however it ends. Nothing needs to be cleaned up after a crash. A consequence is that a scratch file is not visible in the directory; use `disk_io_stats()` to see how much has been written.

This is housekeeping, not protection: another process of the same user, or a privileged one, can still reach an open file of this process through the operating system (on Linux, `/proc/<pid>/fd`). The data is not encrypted, and the space of tiles that are gone is not erased. Do not put data on a scratch disk that its other users must not see.

No shared filesystem is assumed or needed. When a process needs a tile owned by another process, the owner reads the tile and sends its data; the tile that arrives is held in memory by the receiver. A contraction therefore reads each tile of a disk-backed argument once on its owner, and at most once more to send it.

## Memory

A tile is in memory from the moment an expression evaluates it until the expression is done with it. A contraction evaluates the tiles of an argument a batch at a time (one SUMMA iteration), so the argument is never in memory as a whole; how many batches are in flight at once is controlled by the `TA_SUMMA_MAX_DEPTH` environment variable. Smaller tiles along the contracted indices give finer batches.

Writing or reading a tile temporarily needs a buffer of the size of the tile in addition to the tile itself.

## Failures

Running out of disk space, or any other I/O error, raises a `TiledArray::Exception`. When that happens while tiles are being stored by an assignment, the exception is raised inside the runtime's tasks and the program terminates; it cannot be caught and recovered from. The same holds for a tile set with a future (`array.set(i, future)`): the tile is stored when the future is assigned, and a failure then terminates the program rather than leave the array with a tile that never arrives. Make sure the scratch directory has room for the arrays placed there.

Not every write error is reported when the write is made: the operating system may find out only later, when it moves the data to the device, and tell no one. Each tile is therefore stored with a checksum, kept in memory, and verified every time the tile is read, whether to evaluate it or to send it to another process. A tile that does not come back as it was written raises a `TiledArray::Exception` (which, inside a task, terminates the program) instead of yielding wrong numbers.

## Accounting

`TiledArray::disk_io_stats()` returns the number of tiles and bytes written and read by the calling process, and the time spent doing so; `TiledArray::reset_disk_io_stats()` zeroes the counters.

## Limitations

* A tile on disk is immutable: it can be replaced (by assignment), not updated in place or accumulated into.
* There is no restart: scratch files do not outlive the process.
* An evaluated tile is entirely in memory; a tile is the unit of I/O.
* A tile is in memory from when it is produced until it has been stored, and again whenever it is evaluated.
* Assignment from an expression whose lazy tiles evaluate asynchronously (to a `madness::Future`) is not supported.
