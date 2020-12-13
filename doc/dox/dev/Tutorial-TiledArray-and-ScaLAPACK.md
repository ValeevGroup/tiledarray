# Using TiledArray with ScaLAPACK {#Tutorial-TiledArray-and-ScaLAPACK}

TiledArray includes a set of functions that allows you to convert `DistArray`
objects to and from 2D block-cyclic distributions, such as those used by the
[ScaLAPACK](http://www.netlib.org/scalapack/) library. These conversions and
subsequent linear algebra bindings are facilitated through the
[blacspp]() and [scalapackpp]() libraries with provide modern C++ wrappers
around the BLACS and ScaLAPACK libraries.


As the target of the 2D block-cyclic conversions is for compatibility with 
ScaLAPACK, the conversion utilities are only to be used with rank 2 `Tensor`s.
Further, the internal `value_type` of the `DistArray` must be ScaLAPACK compatible,
i.e. `float`, `double`, `std::complex<float>`, `std::complex<double>`. If either
of these conditions are not met, the conversion utilities will throw an error.

## The ScaLAPACKMatrix class

TiledArray provides a helper class which enables the conversion of `DistArray`
objects to 2D block-cyclic distribution formats. It extends the concept of 
a `madness::WorldObject`.

### Public Class API 

```.cpp
    template <typename T,
              typename = scalapackpp::detail::enable_if_scalapack_supported_t<T>>
    class ScaLAPACKMatrix {

      /**
       *  \brief Construct and allocate memory for a ScaLAPACK matrix.
       *
       *  @param[in] world MADNESS World context
       *  @param[in] grid  BLACS grid context
       *  @param[in] M     Number of rows in distributed matrix
       *  @param[in] N     Number of columns in distributed matrix
       *  @param[in] MB    Block-cyclic row distribution factor
       *  @param[in] NB    Block-cyclic column distribution factor
       */
      ScaLAPACKMatrix(madness::World& world, const blacspp::Grid& grid, size_t M,
                      size_t N, size_t MB, size_t NB);

      /**
       *  \brief Construct a ScaLAPACK metrix from a TArray
       *
       *  @param[in] array Array to redistribute
       *  @param[in] grid  BLACS grid context
       *  @param[in] MB    Block-cyclic row distribution factor
       *  @param[in] NB    Block-cyclic column distribution factor
       */
      ScaLAPACKMatrix(const TArray<T>& array, const blacspp::Grid& grid, size_t MB,
                      size_t NB);
    };
```


### Converting between `DistArray<T>` Object and `ScaLAPACKMatrix<T>`

```.cpp
  TArray<T> tensor(...);
  // Populate tensor

  // Get MPI context for the Tensor
  auto&    tensor_world = tensor.world();
  MPI_Comm tensor_comm  = tensor_world.mpi.comm().Get_mpi_comm();

  // Create BLACS Grid context
  blacspp::Grid grid = blacspp::Grid::square_grid(tensor_comm);

  // Set row and column blocking factors
  size_t MB = 128;
  size_t NB = 128;

  // Convert to ScaLAPACK format
  ScaLAPACKMatrix<T> matrix( tensor, grid, MB, NB );

  // Extract matrix dimensions
  auto [M, N]       = matrix.dims();                  // Global
  auto [Mloc, Nloc] = matrix.dist().get_local_dims(); // Local

  // Form DESC array for ScaLAPACK
  auto desc = matrix.dist().descinit_noerror(M, N, Mloc);

  // Do useful linear algebra, e.g. perform a Hermitianeigenvalue decomposition
  // NOTE: scalapackpp will not check symmetry of the input tensor

  // IMPORTANT: Stage ScaLAPACK execution
  tensor_world.gop().fence();

  // EVP only defined for square problems
  assert( M == N   ); 
  assert( MB == NB );

  // Allocate space for eigensystem
  ScaLAPACKMatrix<T> evec( tensor_world, grid, M, M, MB, MB );
  std::vector<scalapackpp::detail::real_t<T>> eval( M );

  // See scalapackpp documentation for details
  auto info = scalapackpp::hereig(
    scalapackpp::VectorFlag::Vectors, // Compute Eigenvectors
    blacspp::Triangle::Lower,         // Only reference the lower triangle
    M, matrix.local_mat().data(), 1, 1, desc, eval.data(),
    evec.local_mat().data(), 1, 1, desc 
  );

  assert( info == 0 );

  // IMPORTANT: Stage ScaLAPACK execution
  tensor_world.gop().fence();

  // Convert Eigenvectors back to DistArray
  auto evec_tensor = evec.tensor_from_matrix( tensor.trange() );

```
