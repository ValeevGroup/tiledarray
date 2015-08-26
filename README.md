[![Build Status](https://travis-ci.org/ValeevGroup/tiledarray.svg?branch=master)](https://travis-ci.org/ValeevGroup/tiledarray)

### Synopsis
TiledArray is a scalable, block-sparse tensor library that is designed to aid in rapid composition of high-performance tensor expressions, appearing for example in many-body quantum mechanics. It allows users to compose tensor expressions of arbitrary complexity in native C++ code that closely resembles the standard mathematical notation. The library is designed to scale from a single multicore computer to a massive cluster.

TiledArray is built on top of MADNESS parallel runtime (MADWorld), part of [MADNESS numerical calculus framework](https://github.com/m-a-d-n-e-s-s/madness).

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
TiledArray is developed by the [Valeev Group](http://research.valeyev.net) at [Virginia Tech](http://www.vt.edu).

### How to Cite

Cite TiledArray as
> "TiledArray: A general-purpose scalable block-sparse tensor library", Justus A. Calvin and Edward F. Valeev, https://github.com/valeevgroup/tiledarray .

Inner workings of TiledArray are partially described in the following publications:
* Justus A. Calvin and Edward F. Valeev, "Task-Based Algorithm for Matrix Multiplication: A Step Towards Block-Sparse Tensor Computing." http://arxiv.org/abs/1504.05046 .

The MADNESS parallel runtime is described in the following publication:
* Robert J. Harrison, Gregory Beylkin, Florian A. Bischoff, Justus A. Calvin, George I. Fann, Jacob Fosso-Tande, Diego Galindo, Jeff R. Hammond, Rebecca Hartman-Baker, Judith C. Hill, Jun Jia, Jakob S. Kottmann, M-J. Yvonne Ou, Junchen Pei, Laura E. Ratcliff, Matthew G. Reuter, Adam C. Richie-Halford, Nichols A. Romero, Hideo Sekino, William A. Shelton, Bryan E. Sundahl, W. Scott Thornton, Edward F. Valeev, Álvaro Vázquez-Mayagoitia, Nicholas Vence and Yukina Yokoi, "madness: A Multiresolution, Adaptive Numerical Environment for Scientific Simulation.", http://arxiv.org/abs/1507.01888 .

### Acknowledgements
Development of TiledArray is made possible by past and present contributions from the National Science Foundation (awards CHE-0847295, CHE-0741927, OCI-1047696, CHE-1362655, ACI-1047696), the Alfred P. Sloan Foundation, and the Camille and Henry Dreyfus Foundation.
