# Introduction

This directory contains classes and functions that are used to mutate tiles in
tensor arithmatic expressions. The these classes are divided in to two groups: the tile
interface and tile operations. The tile interface is the set of functions and
functors that define basic tile unit operations such as addition, subtraction,
contraction, *etc*. These interface functions/functors are user level objects,
meaning a user may provide overloaded functions and class (partial)
specializations for user defined tile types. The tile operations are the
functors used by TiledArray in the distributed evaluators. These tile operations
use the unit operations defined by tile interface functions and classes to
implement several different code paths for each high level operations. For
example, an addition operation may use the `add_to` function, instead of the
`add`, to avoid allocation of a temporary tile in an addition operation between
two arrays. Or it may use other functions if one of the tiles in the addition
operation is a zero tile.

In the following sections, we document the requirements for a user defined tile
type and the function/functor interface required to define tile arthmatic
operations. We also document the tile operation classes used internally by
TiledArray. While the tile operations are not a part of the tile interface and
is not intended for users to override the functionality, it is important to have
a basic understanding of these classes as they diffine the group of tile
interface functions/functors that must be implemented by the user for each high
level arithmatic operation.

# User Defined Tiles

There are two catagories of user defined tile types. The first, and most
obvious, is the arithmatic tile type. This type of tile may be used in all
types of arithmatic operations and contains the data that represents a 
tensor-like object. The second type is a lazy tile, which is used to generate
an arithmatic tile. Lazy tiles should store the minimum information necessare to
generate a tile, such as the tile range information or external data. 

## Arithmetic Tiles

The default arithmatic tile type used in `DistArray` is `Tensor`. However, 
TiledArray also supports user-defined tile types, by replacing the `Tile`
template parameter of `DistArray` with a custom tile type. 
```c++
namespace TiledArray {

  template <typename Tile = TiledArray::Tensor<double>, typename Policy = TiledArray::DensePolicy>
  class DistArray {
     // ...
  };
  
}
```
There are few scenarios where one would like to provide a non-standard type as a
tile; for example, users may want to provide a more efficient implementation of
certain tile operations. An arithmatic tile represent an n-th order tensor
object, and may use any arbitrary data representation. 

* Tile objects must define a shallow copy constructor and assignment operator.
If you would like to use a tile object type that is not or cannot be converted
to a shallow copy object, you may wrap your tile in a `TiledArray::Tile` object.



## Lazy Tiles

A lazy tile is a place-holder for an arithmetic tile object. It is used to
lazily generate tile data as needed by TiledArray. A tile is treated as a lazy
tile if
* he lazy tile defines an `eval_type` member type, or
* the `eval_trait` class is specalized for the lazy tile type.
The output tile is generate by the lazy tile via a type conversion operator
defined by the lazy tile class.

In addition to lazy evaluation of tiles, the generated arithmatic tile may be
optionally flagged as a consumable tile. Meaning the tile may be treated as a
temporary object and its member data may be consumed in algebraic tensor
operations. The idea is similar to C++ move constructor and assignment operator.
To mark a the output of a lazy tile as consumable, set the `is_consumable`
member variable of the `TiledArray::eval_trait` class to `true`. See the example
below.

##### Interface:
```c++
// User-defined lazy tile class
class LazyTile {
public:
  // ...
  
  typedef ... eval_type;
  
  operator eval_type() const;
  
  // ...
};

namespace TiledArray {

  template <>
  struct eval_trait<LazyTile> {
    typedef ... type;
    static constexpr bool is_consumable = true;
  };
  
} // namespace TiledArray 

```

# Tile Interface

When implementing a user-defined tile type, tile arithmatic operations must also be implement so that the tiles may be used in TiledArray tensor expressions. Below is a list of all tile interface functions, functors, and classes that will need to be implemented.

## Empty

The empty function is used to check that a tile has been initialized and may be used in tile algebraic operations. The empty function returns `true` if the tile is not usable in tile algebraic operations, otherwise `false`. Tile empty functions can be implemented with any one of the following methods:

### Member Fuction

A tile class may define a member function named `empty`.

##### Interface: 
```c++
class Tile {
public:

  // ...
  
  bool empty() const;
  
  // ...
}; // class Tile 
```

### Free Function

The empty function may be a non-member function `empty`. The function must be in
the same namespace as the tile class.

##### Interface: 
```c++
class Tile; 
  
bool empty(const Tile&);
```

## Cast

Tile cast operations convert a tile from one type to another. By default, TiledArray will use the cast functor to modify the return type of other arithmatic functions when the return type does not match the default return type. Tile cast can be implemented with any one of the following methods:

### Type Conversion Constructor

The output tile type may define a type conversion constructor that accept the
input tile as an argument. The constructor may be marked as `explicit`, if
reqired.

##### Interface:
```c++
class ToTile {
public:
  ToTile(const FromTile&);
};
```

### Type Conversion Operator

The input tile type may define a type conversion operator for the output tile
type.

##### Interface:
```c++
class FromTile {
public:
  operator ToTile() const;
};
```

### Define `TiledArray::Cast` Functor

In some cases it is not possible, or undesirable, to define a type conversion
constructor and/or operator in the output and input classes, repectively. In
this case, it is neccessary to provdied a (partial) specialization of the the
`TiledArray::Cast` functor. The cast functor is constructed by TiledArray and,
therefore, may not define any constructors, other than the default constructor.

##### Interface:
```c++
// User-defined tile classes
class ToTile;
class FromTile;

namespace TiledArray {

  template <typename Result, typename Arg>
  class Cast;

  // User-defined tile cast operation
  //   ToTile <- FromTile
  template <>
  class Cast<ToTile, FromTile> {
  public:
    typedef ToTile result_type;
    typedef FromTile argument_type;

    result_type operator()(const argument_type&);

  }; // class Cast<ToTile, FromTile>

} // namespace TiledArray
```

## Clone

Tile clone operations create a deep copy of a tile. The input tile object may 
not share any mutable data with the output tile object. A tile clone operation
can be implemented with `clone` member function, `clone` free function, or by specializing the `TiledArray::Clone` class.

### `clone` Member Fuction

A tile class may define a member function named `clone`.

##### Interface:
```c++
class Tile {
public:
  // ...
  
  Tile clone();
  
  // ...
}; // class Tile 
```

### `clone` Free Function

The clone function may be a non-member function `clone`. The function must be in
the same namespace as the tile class.<br>

##### Interface:
```c++
class Tile; 
 
Tile clone(const Tile&);
```

### Specialize `TiledArray::Clone` Functor

The Clone functor may be used to define the clone operation, in the same way the member and non-member `clone` functions given above. In addition, the `TiledArray::Clone` may also be used to combine cast and clone operations into a single step. By default, TiledArray will [cast](#Cast) the input tile instead of cloning it. This behavior may be overridden by providing a (partial) specialization for `TiledArray::Clone`.

##### Interface:

Clone tile operation

```c++
// User defined tile types
class Tile;

namespace TiledArray {

  template <typename Result, typename Arg>
  class Clone;
  
  // User-defined tile cast operation
  template <>
  class Clone<Tile, Tile> {
  public:
    typedef Tile result_type;
    typedef Tile argument_type;

    result_type operator()(const argument_type&);

  }; // class Clone<Tile, Tile>

} // namespace TiledArray
```

Combine clone and cast interface:

```c++
// User defined tile types
class FromTile;
class ToTile;

namespace TiledArray {

  template <typename Result, typename Arg>
  class Clone;
  
  // User-defined tile clone operation
  //   ToTile <- FromTile
  template <>
  class Clone<ToTile, FromTile> {
  public:
    typedef ToTile result_type;
    typedef FromTile argument_type;

    result_type operator()(const argument_type&);

  }; // class Clone<ToTile, FromTile>

} // namespace TiledArray
```

# Internal Tile Operations  

Tile operations are classes that use a combination of the [tile interface](#Tile Interface)
functions. The scenarios that are handled by these classes include zero tiles,
permutations, and consumable (temporary) tiles. While these classes are not a
part of the user interface, it is to know which tile interface functions are in
a given tensor operation. For example, a tensor addition operation uses the
`Add`/`add`, `AddTo`/`add_to`, `Permute`/`permute`, `Clone`/`clone`, and `Cast`
functions. Therefore user that needs support for addition of a user defined 
tile, must implement each of these interface functions.


