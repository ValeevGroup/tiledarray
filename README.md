### Synopsis
TiledArray is a scalable, block-sparse tensor library that is designed to aid in rapid composition of high-performance tensor expressions, appearing for example in many-body quantum mechanics. It allows users to compose tensor expressions of arbitrary complexity in native C++ code that closely resembles the standard mathematical notation. The library is designed to scale from a single multicore computer to a massive cluster.

TiledArray is a work in progress.

### Design Goals
* Simple implementation of tensor algebra in C++
* Dense, and structured/unstructured sparse tensors
* Intra- and inter-node scalability
* General purpose

### Example Code
The following are example expressions written in C++ with TiledArray. TiledArray use the [Einstein summation convention](http://en.wikipedia.org/wiki/Einstein_notation) when evaluating tensor expressions.

* Matrix-matrix multiplication

    C("m,n") = 2.0 * A("m,k") * B("k,n");

* Matrix-vector multiplication

    C("n") = A("k") * B("k,n");

* Complex tensor expression
 
    E("m,n") = 2.0 * A("m,k") * B("k,n") + C("k,n") * D("k,m");

### Developers
TiledArray is developed by the [Valeev Group](http://www.files.chem.vt.edu/chem-dept/valeev/) at [Virginia Tech](http://www.vt.edu/).

### Acknowledgements
Development of TiledArray is made possible by past and present contributions from the National Science Foundation (CHE and OCI divisions), the Alfred P. Sloan Foundation, and the Camille and Henry Dreyfus Foundation.
