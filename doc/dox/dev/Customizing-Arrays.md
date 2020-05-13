# Customizing DistArray {#Customizing-Arrays}

* [User Defined Tiles](#wiki-user-defined-tiles)
  * [Lazy Tiles](#wiki-lazy-tiles)
  * [Data Tiles](#wiki-data-tiles)
* [User Defined Shapes](#wiki-ser-defined-shapes)
* [User Defined Process Map](#wiki-user-defined-process-map)

# User Defined Tiles

The default tile type of `TiledArray::DistArray` is `TiledArray::Tensor`. However, TiledArray supports using user-defined types as tiles.
There are few scenarios where one would like to provide a non-standard type as a tile; for example,
user wants to provide a more efficient implementation of certain operations on tiles.
There are two modes of user-defined types that can be used as tiles: types that store the data elements explicitly (“data tiles”)
and types that generate a data tile as needed (“lazy evaluation tiles”).

## User-Defined Data Tiles

Any user-defined tensor type can play a role of a data tile provided it matches the same concept as `TiledArray::Tensor`. For brevity, instead of an actual concept spec here is an example of a custom tile type that meets the concept spec.

```
class MyTensor {
public:
  // Typedefs
  typedef MyTensor eval_type;           // The type used when evaluating expressions
  typedef TiledArray::Range range_type; // Tensor range type
  typedef ... value_type;               // Element type
  typedef ... numeric_type;             // The scalar type that is compatible with value_type
  typedef ... size_type;                // Size type

public:

  // Default constructors (may be uninitialized)
  MyTensor();

  // Shallow copy constructor; see MyTensor::clone() for deep copy
  MyTensor(const MyTensor& other);

  // Shallow assignment operator; see MyTensor::clone() for deep copy
  MyTensor& operator=(const MyTensor& other);

  // Deep copy
  MyTensor clone() const;

  // Tile range accessor
  const range_type& range() const;

  // Number of elements in the tile
  size_type size() const;

  // Initialization check. False if the tile is fully initialized.
  bool empty() const;

  // MADNESS-compliant serialization
  template <typename Archive>
  void serialize(Archive& ar);

  // Permutation operation

  // result[perm ^ i] = (*this)[i]
  MyTensor permute(const TiledArray::Permutation& perm) const;

  // Scaling operations

  // result[i] = (*this)[i] * factor
  MyTensor scale(const numeric_type factor) const;
  // result[perm ^ i] = (*this)[i] * factor
  MyTensor scale(const numeric_type factor, const TiledArray::Permutation& perm) const;
  // (*this)[i] *= factor
  MyTensor& scale_to(const numeric_type factor) const;

  // Addition operations

  // result[i] = (*this)[i] + right[i]
  MyTensor add(const MyTensor& right) const;
  // result[i] = ((*this)[i] + right[i]) * factor
  MyTensor add(const MyTensor& right, const numeric_type factor) const;
  // result[i] = (*this)[i] + value
  MyTensor add(const value_type& value) const;

  // result[perm ^ i] = (*this)[i] + right[i]
  MyTensor add(const MyTensor& right, const TiledArray::Permutation& perm) const;
  // result[perm ^ i] = ((*this)[i] + right[i]) * factor
  MyTensor add(const MyTensor& right, const numeric_type factor, const TiledArray::Permutation& perm) const;
  // result[perm ^ i] = (*this)[i] + value
  MyTensor add(const value_type& value, const TiledArray::Permutation& perm) const;

  // (*this)[i] += right[i]
  MyTensor& add_to(const MyTensor& right) const;
  // ((*this)[i] += right[i]) *= factor
  MyTensor& add_to(const MyTensor& right, const numeric_type factor) const;
  // (*this)[i] += value
  MyTensor& add_to(const value_type& value) const;

  // Subtraction operations

  // result[i] = (*this)[i] - right[i]
  MyTensor subt(const MyTensor& right) const;
  // result[i] = ((*this)[i] - right[i]) * factor
  MyTensor subt(const MyTensor& right, const numeric_type factor) const;
  // result[i] = (*this)[i] - value
  MyTensor subt(const value_type& value) const;

  // result[perm ^ i] = (*this)[i] - right[i]
  MyTensor subt(const MyTensor& right, const TiledArray::Permutation& perm) const;
  // result[perm ^ i] = ((*this)[i] - right[i]) * factor
  MyTensor subt(const MyTensor& right, const numeric_type factor, const TiledArray::Permutation& perm) const;
  // result[perm ^ i] = (*this)[i] - value
  MyTensor subt(const value_type value, const TiledArray::Permutation& perm) const;

  // (*this)[i] -= right[i]
  MyTensor& subt_to(const MyTensor& right);
  // ((*this)[i] -= right[i]) *= factor
  MyTensor& subt_to(const MyTensor& right, const numeric_type factor);
  // (*this)[i] -= value
  MyTensor& subt_to(const value_type& value);

  // (Entrywise) multiplication operations (Hadamard product)

  // result[i] = (*this)[i] * right[i]
  MyTensor mult(const MyTensor& right) const;
  // result[i] = ((*this)[i] * right[i]) * factor
  MyTensor mult(const MyTensor& right, const numeric_type factor) const;

  // result[perm ^ i] = (*this)[i] * right[i]
  MyTensor mult(const MyTensor& right, const TiledArray::Permutation& perm) const;
  // result[perm^ i] = ((*this)[i] * right[i]) * factor
  MyTensor mult(const MyTensor& right, const numeric_type factor, const TiledArray::Permutation& perm) const;

  // *this[i] *= right[i]
  MyTensor& mult_to(const MyTensor& right);
  // (*this[i] *= right[i]) *= factor
  MyTensor& mult_to(const MyTensor& right, const numeric_type factor);

  // Negation operations

  // result[i] = -((*this)[i])
  MyTensor neg() const;
  // result[perm ^ i] = -((*this)[i])
  MyTensor neg(const TiledArray::Permutation& perm) const;
  // arg[i] = -((*this)[i])
  MyTensor& neg_to();

  // Contraction operations

  // GEMM operation with fused indices as defined by gemm_config; multiply this by other, return the result
  MyTensor gemm(const MyTensor& other, const numeric_type factor,
                const TiledArray::math::GemmHelper& gemm_config) const;

  // GEMM operation with fused indices as defined by gemm_config; multiply left by right, store to this
  MyTensor& gemm(const MyTensor& left, const MyTensor& right, const numeric_type factor,
                 const TiledArray::math::GemmHelper& gemm_config);

  // Reduction operations

  // Sum of hyper diagonal elements
  numeric_type trace() const;
  // foreach(i) result += arg[i]
  numeric_type sum() const;
  // foreach(i) result *= arg[i]
  numeric_type product() const;
  // foreach(i) result += arg[i] * arg[i]
  numeric_type squared_norm() const;
  // sqrt(squared_norm(arg))
  numeric_type norm() const;
  // foreach(i) result = max(result, arg[i])
  numeric_type max() const;
  // foreach(i) result = min(result, arg[i])
  numeric_type min() const;
  // foreach(i) result = max(result, abs(arg[i]))
  numeric_type abs_max() const;
  // foreach(i) result = max(result, abs(arg[i]))
  numeric_type abs_min() const;
  
} // class MyTensor
```

It is also possible to implement most of the concept requirements non-intrusively, by providing free functions. This can be helpful if you want to use an existing tensor class as a tile. Here’s an example of how to implement MyTensor without member functions:

```
class MyTensor {
public:
  // Typedefs
  typedef TiledArray::Range range_type; // Tensor range type
  typedef ... value_type;               // Element type
  typedef ... numeric_type;             // The scalar type that is compatible with value_type
  typedef ... size_type;                // Size type

public:

  // Default constructors (may be uninitialized)
  MyTensor();

  // Shallow copy constructor; see MyTensor::clone() for deep copy
  MyTensor(const MyTensor& other);

  // Shallow assignment operator; see MyTensor::clone() for deep copy
  MyTensor& operator=(const MyTensor& other);

  // Deep copy
  MyTensor clone() const;

  // Tile range accessor
  const range_type& range() const;

  // Number of elements in the tile
  size_type size() const;

  // Initialization check. False if the tile is fully initialized.
  bool empty() const;

  // MADNESS-compliant serialization
  template <typename Archive>
  void serialize(Archive& ar);

  // Scaling operations

  // result[i] = (*this)[i] * factor
  MyTensor scale(const numeric_type factor) const;
  // result[perm ^ i] = (*this)[i] * factor
  MyTensor scale(const numeric_type factor, const TiledArray::Permutation& perm) const;
  // (*this)[i] *= factor
  MyTensor& scale_to(const numeric_type factor) const;

} // class MyTensor

namespace TiledArray {

  // MyTensor is used directly evaluate expressions (see also Lazy Tiles section below)
  struct eval_trait<MyTensor> {
      typedef MyTensor type;
  };

  namespace math {

  // Permutation operation

  // returns a tile for which result[perm ^ i] = tile[i]
  MyTensor permute(const MyTensor& tile,
                   const TiledArray::Permutation& perm);

  // Addition operations

  // result[i] = arg1[i] + arg2[i]
  MyTensor add(const MyTensor& arg1,
               const MyTensor& arg2);
  // result[i] = (arg1[i] + arg2[i]) * factor
  MyTensor add(const MyTensor& arg1,
               const MyTensor& arg2,
               const MyTensor::value_type factor);
  // result[i] = arg[i] + value
  MyTensor add(const MyTensor& arg,
               const MyTensor::value_type& value);

  // result[perm ^ i] = arg1[i] + arg2[i]
  MyTensor add(const MyTensor& arg1,
               const MyTensor& arg2,
               const TiledArray::Permutation& perm);
  // result[perm ^ i] = (arg1[i] + arg2[i]) * factor
  MyTensor add(const MyTensor& arg1,
               const MyTensor& arg2,
               const MyTensor::numeric_type factor,
               const TiledArray::Permutation& perm);
  // result[perm ^ i] = arg[i] + value
  MyTensor add(const MyTensor& arg,
               const MyTensor::value_type& value,
               const TiledArray::Permutation& perm);

  // result[i] += arg[i]
  void add_to(MyTensor& result,
              const MyTensor& arg);
  // (result[i] += arg[i]) *= factor
  void add_to(MyTensor& result,
              const MyTensor& arg,
              const MyTensor::numeric_type factor);
  // result[i] += value
  void add_to(MyTensor& result,
              const MyTensor::value_type& value);

  // Subtraction operations

  // result[i] = arg1[i] - arg2[i]
  MyTensor subt(const MyTensor& arg1,
                const MyTensor& arg2);
  // result[i] = (arg1[i] - arg2[i]) * factor
  MyTensor subt(const MyTensor& arg1,
                const MyTensor& arg2,
                const MyTensor::numeric_type factor);
  // result[i] = arg[i] - value
  MyTensor subt(const MyTensor& arg,
                const MyTensor::value_type& value);

  // result[perm ^ i] = arg1[i] - arg2[i]
  MyTensor subt(const MyTensor& arg1,
                const MyTensor& arg2,
                const TiledArray::Permutation& perm);
  // result[perm ^ i] = (arg1[i] - arg2[i]) * factor
  MyTensor subt(const MyTensor& arg1,
                const MyTensor& arg2,
                const MyTensor::numeric_type factor,
                const TiledArray::Permutation& perm);
  // result[perm ^ i] = arg[i] - value
  MyTensor subt(const MyTensor& arg,
                const MyTensor::value_type value,
                const TiledArray::Permutation& perm);

  // result[i] -= arg[i]
  void subt_to(MyTensor& result,
               const MyTensor& arg);
  // (result[i] -= arg[i]) *= factor
  void subt_to(MyTensor& result,
               const MyTensor& arg,
               const MyTensor::numeric_type factor);
  // result[i] -= value
  void subt_to(MyTensor& result,
               const MyTensor::value_type& value);

  // (Entrywise) multiplication operations (Hadamard product)

  // result[i] = arg1[i] * arg2[i]
  MyTensor mult(const MyTensor& arg1,
                const MyTensor& arg2);
  // result[i] = (arg1[i] * arg2[i]) * factor
  MyTensor mult(const MyTensor& arg1,
                const MyTensor& arg2,
                const MyTensor::numeric_type factor);

  // result[perm ^ i] = arg1[i] * arg2[i]
  MyTensor mult(const MyTensor& arg1,
                const MyTensor& arg2,
                const TiledArray::Permutation& perm);
  // result[perm^ i] = (arg1[i] * arg2[i]) * factor
  MyTensor mult(const MyTensor& arg1,
                const MyTensor& arg2,
                const MyTensor::numeric_type factor,
                const TiledArray::Permutation& perm);

  // result[i] *= arg[i]
  void mult_to(MyTensor& result,
               const MyTensor& arg);
  // (result[i] *= arg[i]) *= factor
  void mult_to(MyTensor& result,
               const MyTensor& arg,
               const MyTensor::numeric_type factor);

  // Negation operations

  // result[i] = -(arg[i])
  MyTensor neg(const MyTensor& arg);
  // result[perm ^ i] = -(arg[i])
  MyTensor neg(const MyTensor& arg,
               const TiledArray::Permutation& perm);
  // result[i] = -(result[i])
  void neg_to(MyTensor& result);

  // Contraction operations

  // GEMM operation with fused indices as defined by gemm_config; multiply arg1 by arg2, return the result
  MyTensor gemm(const MyTensor& arg1,
                const MyTensor& arg2,
                const MyTensor::numeric_type factor,
                const TiledArray::math::GemmHelper& gemm_config);

  // GEMM operation with fused indices as defined by gemm_config; multiply left by right, store to result
  void gemm(MyTensor& result,
            const MyTensor& arg1,
            const MyTensor& arg2,
            const MyTensor::numeric_type factor,
            const TiledArray::math::GemmHelper& gemm_config);


  // Reduction operations

  // Sum of hyper diagonal elements
  MyTensor::numeric_type trace(const MyTensor& arg);
  // foreach(i) result += arg[i]
  MyTensor::numeric_type sum(const MyTensor& arg);
  // foreach(i) result *= arg[i]
  MyTensor::numeric_type product(const MyTensor& arg);
  // foreach(i) result += arg[i] * arg[i]
  MyTensor::numeric_type squared_norm(const MyTensor& arg);
  // sqrt(squared_norm(arg))
  MyTensor::numeric_type norm(const MyTensor& arg);
  // foreach(i) result = max(result, arg[i])
  MyTensor::numeric_type max(const MyTensor& arg);
  // foreach(i) result = min(result, arg[i])
  MyTensor::numeric_type min(const MyTensor& arg);
  // foreach(i) result = max(result, abs(arg[i]))
  MyTensor::numeric_type abs_max(const MyTensor& arg);
  // foreach(i) result = min(result, abs(arg[i]))
  MyTensor::numeric_type abs_min(const MyTensor& arg);
  
```

## User-Defined Lazy Tiles

Lazy tiles are generated only when they are needed and discarded immediately after use. Common uses for lazy tiles include computing arrays on-the-fly, reading them from disk, etc. Lazy tiles are used internally by `TiledArray::DistArray` to generate data tiles that are then fed into arithmetic operations.

The main requirements of lazy tiles are:

1. `typedef ... eval_type`, which is the data tile type (e.g. TiledArray::Tensor).
2. `eval_type` cannot be the same object type as the lazy tile itself.
3. `explicit operator eval_type() const`, which is the function used to generate the data tile.

Lazy tiles should have the following interface.

```
class MyLazyTile {
public:
  typedef ... eval_type; // The data tile to which this tile will be converted to; typically TiledArray::Tensor
                         // Can instead define TiledArray::eval_trait<MyLazyTile>::type

  // Default constructor
  MyLazyTile();

  // Copy constructor
  MyLazyTile(const MyLazyTile& other);

  // Assignment operator
  MyLazyTile& operator=(const MyLazyTile& other);

  // Convert lazy tile to data tile
  explicit operator eval_type() const;

  // MADNESS compliant serialization
  template <typename Archive>
  void serialize(const Archive&);

}; // class MyLazyTile
```

# User Defined Shapes

You can define a shape object for your `Array` object, which defines the sparsity of an array. A shape object is a replicated object, so you should design your shape object accordingly. You may implement an initialization algorithm for your shape that communicates with other processes. However, communication is not allowed after the object has been initialized, shape arithmetic operations must be completely local (non-communicating) operations. 

```
class MyShape {
public:

  // Return true if range matches the range of this shape
  bool validate(const Range& shape);

  // Returns true if the 
  template <typename Index>
  bool is_zero(const Index&);

  /// Returns true if this shape is dense.
  bool is_dense();

  // Permute shape
  MyShape perm(const TiledArray::Permutation& perm);

  // Scale shape
  template <typename Scalar>
  MyShape scale(const Scalar factor);

  // Scale and permute shape
  template <typename Scalar>
  MyShape scale(const Scalar factor, const TiledArray::Permutation& perm);

  // Add shapes
  MyShape add(const MyShape& right);

  // Add shapes and permute the result
  MyShape add(const MyShape& right, const TiledArray::Permutation& perm);

  // Add and scale shapes
  template <typename Scalar>
  MyShape add(const MyShape& right, const Scalar factor);

  // Add and scale shapes, and permute the result
  template <typename Scalar>
  MyShape add(const MyShape& right, const Scalar factor, const TiledArray::Permutation& perm);

  // Add a constant to a shape
  template <typename Scalar>
  MyShape add(const Scalar value)

  // Add a constant to and scale a shape, and permute the result
  template <typename Scalar>
  MyShape add(const Scalar value, const TiledArray::Permutation& perm);

  // Subtract shapes
  MyShape subt(const MyShape& right);

  // Subtract shapes, and permute the result
  MyShape subt(const MyShape& right, const TiledArray::Permutation& perm);

  // Subtract and scale shapes 
  template <typename Scalar>
  MyShape subt(const MyShape& right, const Scalar factor);

  // Subtract and scale shapes, and permute the result
  template <typename Scalar>
  MyShape subt(const MyShape& right, const Scalar factor, const TiledArray::Permutation& perm);

  // Subtract a constant value
  template <typename Scalar>
  MyShape subt(const Scalar value);

  // Subtract a constant value, and permute the result
  template <typename Scalar>
  MyShape subt(const Scalar value, const TiledArray::Permutation& perm);

  // (Entrywise) multiplication of shapes
  MyShape mult(const MyShape& right);

  // (Entrywise) multiplication of shapes, followed by permutation
  MyShape mult(const MyShape& right, const TiledArray::Permutation& perm);

  // (Entrywise) multiplication of shapes, followed by scaling
  template <typename Scalar>
  MyShape mult(const MyShape& right, const Scalar factor);

  // (Entrywise) multiplication of shapes, followed by scaling, followed by permutation
  template <typename Scalar>
  MyShape mult(const MyShape& right, const Scalar factor, const TiledArray::Permutation& perm);

  // Contract and scale shapes
  template <typename Scalar>
  MyShape gemm(const MyShape& right, const Scalar factor,
      const TiledArray::math::GemmHelper& gemm_helper);

  // Contract and scale shapes, and permute the result
  template <typename Scalar>
  MyShape gemm(const MyShape& right, const Scalar factor, 
      const TiledArray::math::GemmHelper& gemm_helper, const TiledArray::Permutation& perm);
}; // class MyShape
```

# User Defined Process Map

You can also create process maps for your `Array` object, which is used by TiledArray to determine the process that owns a tile for a given `Array` object. For a process map to be valid, all tiles are owned by exactly one process and all processes must agree on this tile ownership. The exception to these rules is a replicated process map. In addition, a process map must maintain a list of local tiles.

```
class MyPmap : public TiledArray::Pmap {
protected:

  // Import Pmap protected variables
  using Pmap::rank_;  // The rank of this process
  using Pmap::procs_; // The number of processes
  using Pmap::size_;  // The number of tiles mapped among all processes
  using Pmap::local_; // A list of local tiles (you must initialize this in the constructor)

public:
  typedef Pmap::size_type size_type; // Key type

  // Constructor
  MyPmap(madness::World& world, size_type size);

  // Virtual destructor
  virtual ~MyPmap();

  // Returns the process that owns tile
  virtual size_type owner(const size_type tile) const;

  // Returns true if tile is owned by this process
  virtual bool is_local(const size_type tile) const;
}; // class MyPmap
```
