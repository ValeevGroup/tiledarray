# Using TiledArray with Eigen {#Tutorial-TiledArray-and-Eigen}

TiledArray has a set of functions that allows you to convert `Array` object to and from [Eigen](http://eigen.tuxfamily.org/) matrix objects. The purpose of these functions is to allow users to quickly prototype TiledArray algorithms with inputs from or outputs to Eigen matrices.

Because Eigen is a non-distributed library, these functions are not appropriate for production code in distributed computing environments. Therefore, these functions will throw an exception if the number of MPI processes is greater than one. If you require conversion of Eigen matrices in distributed computing environments, you need to write your own algorithm to collect or distribute matrix data. See the Submatrix Copy section for more information. 

# Conversion Functions

## `eigen_to_array()`
Convert an Eigen matrix into a DistArray object.

### Signature

```.cpp
    template<typename A , typename Derived >
    A
    TiledArray::eigen_to_array(madness::World & world,
                               const typename A::trange_type & trange,
                               const Eigen::MatrixBase< Derived > & matrix)
```

### Description
This function will copy the content of `matrix` into a DistArray object that is tiled according to the `trange` object. The copy operation is done in parallel, and this function will block until all elements of `matrix` have been copied into the result array tiles.

#### Template Parameters
* `A` A DistArray type
* `Derived` The Eigen matrix derived type

#### Parameters
* `world` The world where the array will live
* `trange` The tiled range of the new array
* `matrix` The Eigen matrix to be copied

#### Returns
A DistArray object (of type `A`) that contains a copy of `matrix`

#### Exceptions
`TiledArray::Exception` When world size is greater than 1

### Usage

```.cpp
    Eigen::MatrixXd matrix(100, 100);
    // Fill matrix with data ...
    
    // Create a tiled range for the new array object
    std::vector<std::size_t> blocks;
    for(std::size_t i = 0ul; i <= 100ul; i += 10ul)
      blocks.push_back(i);
    std::array<TiledArray::TiledRange1, 2> blocks2 =
        {{ TiledArray::TiledRange1(blocks.begin(), blocks.end()),
           TiledArray::TiledRange1(blocks.begin(), blocks.end()) }};
    TiledArray::TiledRange trange(blocks2.begin(), blocks2.end());
    
    // Create a DistArray from an Eigen matrix.
    TiledArray::TArray<double> array = eigen_to_array<TiledArray::TArray<double> >(world, trange, matrix);
```

### Notes

This function will only work in non-distributed environments. If you need to convert an Eigen matrix to a DistArray object, you must implement it yourself. However, you may use eigen_submatrix_to_tensor to make writing such an algorithm easier.

## `array_to_eigen()`
Convert a DistArray object into an Eigen matrix object.

### Signature

```.cpp
    template<typename T , unsigned int DIM, typename Tile >
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    TiledArray::array_to_eigen(const Array< T, DIM, Tile > & array)
```

### Description
This function will block until all tiles of array have been set and copied into the new Eigen matrix. Usage:

### Template Parameters
* `Tile` The tile type of the array
* `Policy` The policy type of the array

#### Parameters
* `array` The array to be converted

#### Exceptions
`TiledArray::Exception` When world size is greater than 1

#### Usage

```
    TiledArray::TArray<double> array(world, trange);
    // Set tiles of array ...
    Eigen::MatrixXd m = array_to_eigen(array);
```

### Note
This function will only work in non-distributed environments. If you need to convert an Array object to an Eigen matrix, you must implement it yourself. However, you may use eigen_submatrix_to_tensor to make writing such an algorithm easier.

# Eigen Interface

Eigen includes a map class which allows external libraries to work with Eigen. TiledArray provides a set of functions for wrapping `Tensor` object with Eigen::Map. These map objects may be used as normal Eigen matrices or vectors.

## `eigen_map()`

Construct an `m x n` `Eigen::Map` object for a given `Tensor` object.

### Signature

```
    template <typename T, typename A>
    inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(const Tensor<T, A>& tensor, const std::size_t m, const std::size_t n)

    template <typename T, typename A>
    inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(Tensor<T, A>& tensor, const std::size_t m, const std::size_t n)
```

### Description
This function will wrap a `tensor` in an Eigen map object. This object may be used in expressions with other Eigen objects. See [Eigen documentation](http://eigen.tuxfamily.org/dox/TutorialMapClass.html) for more details on Map objects.

#### Template Parameters
* `T` The tensor element type
* `A` The tensor allocator type

#### Parameters
* `tensor` The tensor object
* `m` The number of rows in the result matrix
* `n` The number of columns in the result matrix

#### Returns
An m x n Eigen matrix map for `tensor`

#### Exceptions
`TiledArray::Exception` When `m * n` is not equal to  `tensor` size.

### Usage

```
    // Construct a tensor object
    std::array<std::size_t, 2> size = {{ 10, 10 }};
    TiledArray::Tensor<int> tensor(TiledArray::Range(size), 1);
    
    // Construct an Eigen matrix
    Eigen::MatrixXi matrix(10, 10);
    matrix.fill(1);

    // Intialize an Eigen Matrix with a TiledArray tensor.
    Eigen::MatrixXi x = matrix + TiledArray::eigen_map(tensor, 10, 10);
```

### Note
The dimensions `m` and `n` do not have to match the dimensions of the tensor object, but they size of the tensor must be equal to the size of the resulting Eigen map object.

## eigen_map()

Construct a `Eigen::Map` object for a given `Tensor` object.

### Signature

```
    template <typename T, typename A>
    inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(const Tensor<T, A>& tensor)

    template <typename T, typename A>
    inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(Tensor<T, A>& tensor)
```

### Description
This function will wrap a `tensor` in an Eigen map object. This object may be used in expressions with other Eigen objects. See [http://eigen.tuxfamily.org/dox/TutorialMapClass.html Eigen documentation] for more details on Map objects.

#### Template Parameters
* `T` The tensor element type
* `A` The tensor allocator type

#### Parameters
* `tensor` The tensor object

#### Returns
An  Eigen matrix map for `tensor` where the number of rows and columns of the matrix match  the dimension sizes of the tensor object.

#### Exceptions
`TiledArray::Exception` When `tensor` dimensions are not equal to 2 or 1.

#### Usage

```
    // Construct a tensor object
    std::array<std::size_t, 2> size = {{ 10, 10 }};
    TiledArray::Tensor<int> tensor(TiledArray::Range(size), 1);
    
    // Construct an Eigen matrix
    Eigen::MatrixXi matrix(10, 10);
    matrix.fill(1);
    
    // Intialize an Eigen Matrix with a TiledArray tensor.
    Eigen::MatrixXi x = matrix + TiledArray::eigen_map(tensor);
```
