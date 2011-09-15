#ifndef TILEDARRAY_ARRAY_BASE_H__INCLUDED
#define TILEDARRAY_ARRAY_BASE_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/tensor_base.h>
#include <TiledArray/bitset.h>
#include <world/worldtypes.h>
#include <world/sharedptr.h>

#define TILEDARRAY_ANNOTATED_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED )  \
    TILEDARRAY_TENSOR_BASE_INHEIRATE_TYPEDEF( BASE , DERIVED )

#define TILEDARRAY_TILED_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_ANNOTATED_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::pmap_interface pmap_interface; \
    typedef typename base::trange_type trange_type;

#define TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_TILED_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::value_type value_type; \
    typedef typename base::const_reference const_reference; \
    typedef typename base::const_iterator const_iterator;

#define TILEDARRAY_WRITABLE_TILED_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::reference reference; \
    typedef typename base::iterator iterator;

#define TILEDARRAY_ANNOTATED_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_TENSOR_BASE_INHEIRATE_MEMBER( BASE , DERIVED ) \
    inline const ::TiledArray::expressions::VariableList& vars() const { return base::vars(); }

#define TILEDARRAY_TILED_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_ANNOTATED_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    inline ProcessID owner(size_type i) const { return base::owner(i); } \
    inline bool is_local(size_type i) const { return base::is_local(i); } \
    inline bool is_zero(size_type i) const { return base::is_zero(i); } \
    inline madness::World& get_world() const { return base::get_world(); } \
    inline std::shared_ptr<pmap_interface> get_pmap() const { return base::get_pmap(); } \
    inline bool is_dense() const { return base::is_dense(); } \
    inline const TiledArray::detail::Bitset<>& get_shape() const { return base::get_shape(); } \
    inline size_array size(size_type i) const { return base::size(i); } \
    inline size_type volume(size_type i) const { return base::volume(i); } \
    inline trange_type trange() const { return base::trange(); }

#define TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_TILED_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    inline const_reference operator[](size_type i) const { return base::operator[](i); } \
    inline const_iterator begin() const { return base::begin(); } \
    inline const_iterator end() const { return base::end(); }

#define TILEDARRAY_WRITABLE_TILED_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    inline void set(size_type i, const madness::Future<value_type>& f) { return base::insert(i, f); }

namespace madness {
  // Forward declaration
  class World;
  template <typename> class WorldDCPmapInterface;
  template <typename> class Future;
} // namespace madness

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class AnnotatedTensor : public TensorBase<Derived> {
    public:

      TILEDARRAY_TENSOR_BASE_INHEIRATE_TYPEDEF(TensorBase<Derived>, Derived)
      TILEDARRAY_TENSOR_BASE_INHEIRATE_MEMBER(TensorBase<Derived>, Derived)

      inline const VariableList& vars() const { return derived().vars(); }

    }; // class AnnotatedTensor

    template <typename Derived>
    class TiledTensor : public AnnotatedTensor<Derived> {
    public:

      TILEDARRAY_ANNOTATED_TENSOR_INHEIRATE_TYPEDEF(AnnotatedTensor<Derived>, Derived)
      typedef madness::WorldDCPmapInterface<size_type> pmap_interface;
      typedef typename TensorTraits<Derived>::trange_type trange_type;

      TILEDARRAY_ANNOTATED_TENSOR_INHEIRATE_MEMBER(AnnotatedTensor<Derived>, Derived)

      // Tile locality info
      inline ProcessID owner(size_type i) const { return derived().owner(i); }
      inline bool is_local(size_type i) const { return derived().is_local(i); }
      inline bool is_zero(size_type i) const { return derived().is_zero(i); }
      inline madness::World& get_world() const { return derived().get_world(); }
      inline std::shared_ptr<pmap_interface> get_pmap() const { return derived().get_pmap(); }
      inline bool is_dense() const { return derived().is_dense(); }
      inline const TiledArray::detail::Bitset<>& get_shape() const { return derived().get_shape(); }

      // Tile dimension info
      inline size_array size(size_type i) const { return derived().size(i); }
      inline size_type volume(size_type i) const { return derived().volume(i); }
      inline trange_type trange() const { return derived().trange(); }

    }; // class TiledTensor

    template <typename Derived>
    class ReadableTiledTensor : public TiledTensor<Derived> {
    public:

      TILEDARRAY_TILED_TENSOR_INHEIRATE_TYPEDEF(TiledTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;
      typedef typename TensorTraits<Derived>::const_iterator const_iterator;

      TILEDARRAY_TILED_TENSOR_INHEIRATE_MEMBER(TiledTensor<Derived>, Derived)

      // element access
      const_reference operator[](size_type i) const { return derived()[i]; }

      // iterator factory
      const_iterator begin() const { return derived().begin(); }
      const_iterator end() const { return derived().end(); }

    }; // class ReadableTiledTensor

    template <typename Derived>
    class WritableTiledTensor : public ReadableTiledTensor<Derived> {
    public:

      TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_TYPEDEF(ReadableTiledTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::reference reference;
      typedef typename TensorTraits<Derived>::iterator iterator;

      TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_MEMBER(ReadableTiledTensor<Derived>, Derived)

      // element access
      void set(size_type i, const madness::Future<value_type>& f) { return derived().insert(i, f); }

    }; // class WritableTiledTensor

  } // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_ARRAY_BASE_H__INCLUDED
