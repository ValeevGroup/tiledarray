#ifndef TILEDARRAY_ARRAY_BASE_H__INCLUDED
#define TILEDARRAY_ARRAY_BASE_H__INCLUDED

// This needs to be defined before world/worldreduce.h
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/coordinate_system.h>
#include <TiledArray/tensor_base.h>
#include <TiledArray/bitset.h>
#include <world/worldtypes.h>
#include <world/shared_ptr.h>

#define TILEDARRAY_ANNOTATED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED )  \
    TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF( BASE , DERIVED )

#define TILEDARRAY_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_ANNOTATED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::pmap_interface pmap_interface; \
    typedef typename base::trange_type trange_type;

#define TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::value_type value_type; \
    typedef typename base::const_reference const_reference; \
    typedef typename base::const_iterator const_iterator;

#define TILEDARRAY_WRITABLE_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED )

#define TILEDARRAY_ANNOTATED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_TENSOR_BASE_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::vars;

#define TILEDARRAY_TILED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_ANNOTATED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::owner; \
    using base::is_local; \
    using base::is_zero; \
    using base::get_world; \
    using base::get_pmap; \
    using base::is_dense; \
    using base::get_shape; \
    using base::trange;

#define TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_TILED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::operator[]; \
    using base::begin; \
    using base::end;

#define TILEDARRAY_WRITABLE_TILED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::set;

namespace madness {
  // Forward declaration
  class World;
  template <typename> class WorldDCPmapInterface;
  template <typename> class Future;
} // namespace madness

namespace TiledArray {
  namespace expressions {

    class VariableList;

    template <typename Derived>
    class AnnotatedTensor : public TensorBase<Derived> {
    public:

      TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF(TensorBase<Derived>, Derived)
      TILEDARRAY_TENSOR_BASE_INHERIT_MEMBER(TensorBase<Derived>, Derived)

      inline const VariableList& vars() const { return derived().vars(); }

    }; // class AnnotatedTensor

    template <typename Derived>
    class TiledTensor : public AnnotatedTensor<Derived> {
    public:

      TILEDARRAY_ANNOTATED_TENSOR_INHERIT_TYPEDEF(AnnotatedTensor<Derived>, Derived)
      typedef madness::WorldDCPmapInterface<size_type> pmap_interface;
      typedef typename TensorTraits<Derived>::trange_type trange_type;

      TILEDARRAY_ANNOTATED_TENSOR_INHERIT_MEMBER(AnnotatedTensor<Derived>, Derived)

      eval_type eval() const { return derived(); }

      // Tile locality info
      inline ProcessID owner(size_type i) const { return derived().owner(i); }
      inline bool is_local(size_type i) const { return derived().is_local(i); }
      inline bool is_zero(size_type i) const { return derived().is_zero(i); }
      inline madness::World& get_world() const { return derived().get_world(); }
      inline const std::shared_ptr<pmap_interface>& get_pmap() const { return derived().get_pmap(); }
      inline bool is_dense() const { return derived().is_dense(); }
      inline const TiledArray::detail::Bitset<>& get_shape() const { return derived().get_shape(); }
      inline trange_type trange() const { return derived().trange(); }

    }; // class TiledTensor

    template <typename Derived>
    class ReadableTiledTensor : public TiledTensor<Derived> {
    public:

      TILEDARRAY_TILED_TENSOR_INHERIT_TYPEDEF(TiledTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;
      typedef typename TensorTraits<Derived>::const_iterator const_iterator;

      TILEDARRAY_TILED_TENSOR_INHERIT_MEMBER(TiledTensor<Derived>, Derived)

      // element access
      const_reference operator[](size_type i) const { return derived()[i]; }

      // iterator factory
      const_iterator begin() const { return derived().begin(); }
      const_iterator end() const { return derived().end(); }

    }; // class ReadableTiledTensor

    template <typename Derived>
    class WritableTiledTensor : public ReadableTiledTensor<Derived> {
    public:

      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<Derived>, Derived)

      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_MEMBER(ReadableTiledTensor<Derived>, Derived)

      // element access
      void set(size_type i, const value_type& v) { return derived().set(i, v); }
      void set(size_type i, const madness::Future<value_type>& v) { return derived().set(i, v); }

    }; // class WritableTiledTensor

  } // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_ARRAY_BASE_H__INCLUDED
