#ifndef TILEDARRAY_ARRAY_BASE_H__INCLUDED
#define TILEDARRAY_ARRAY_BASE_H__INCLUDED

// This needs to be defined before world/worldreduce.h
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/coordinate_system.h>
#include <TiledArray/tensor_base.h>
#include <TiledArray/bitset.h>
#include <world/worldtypes.h>
#include <world/shared_ptr.h>
#include <world/world.h>

namespace madness {
  // Forward declaration
  class World;
  template <typename> class WorldDCPmapInterface;
} // namespace madness

namespace TiledArray {

  template <typename, typename> class Array;

  namespace expressions {

    class VariableList;

    template <typename Derived>
    class AnnotatedTensor : public TensorBase<Derived> {
    public:

      typedef TensorBase<Derived> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;

      using base::derived;
      using base::range;
      using base::size;

      inline const VariableList& vars() const { return derived().vars(); }

    }; // class AnnotatedTensor

    template <typename Derived>
    class TiledTensor : public AnnotatedTensor<Derived> {
    public:

      typedef AnnotatedTensor<Derived> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef madness::WorldDCPmapInterface<size_type> pmap_interface;
      typedef typename TensorTraits<Derived>::trange_type trange_type;

      using base::derived;
      using base::range;
      using base::size;
      using base::vars;

      madness::Future<bool> eval(const VariableList& v) { return derived().eval(v); }

      // Tile locality info
      inline ProcessID owner(size_type i) const { return derived().owner(i); }
      inline bool is_local(size_type i) const { return derived().is_local(i); }
      inline bool is_zero(size_type i) const { return derived().is_zero(i); }
      inline madness::World& get_world() const { return derived().get_world(); }
      inline const std::shared_ptr<pmap_interface>& get_pmap() const { return derived().get_pmap(); }
      inline bool is_dense() const { return derived().is_dense(); }
      inline const TiledArray::detail::Bitset<>& get_shape() const { return derived().get_shape(); }
      inline trange_type trange() const { return derived().trange(); }

      /// Conversion operator

      /// \tparam T The array element type
      /// \tparam CS The array coordinates system
      /// Evaluate this object and convert it to array type.
      /// \return Return an \c Array<T,CS> object that matches this object.
      template <typename T, typename CS>
      operator Array<T, CS>();

    }; // class TiledTensor

    template <typename Derived>
    class ReadableTiledTensor : public TiledTensor<Derived> {
    public:

      typedef TiledTensor<Derived> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::pmap_interface pmap_interface;
      typedef typename base::trange_type trange_type;
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;
      typedef typename TensorTraits<Derived>::const_iterator const_iterator;

      using base::derived;
      using base::range;
      using base::size;
      using base::vars;
      using base::owner;
      using base::is_local;
      using base::is_zero;
      using base::get_world;
      using base::get_pmap;
      using base::is_dense;
      using base::get_shape;
      using base::trange;

      // element access
      const_reference operator[](size_type i) const { return derived()[i]; }

      // iterator factory
      const_iterator begin() const { return derived().begin(); }
      const_iterator end() const { return derived().end(); }

    }; // class ReadableTiledTensor

    template <typename Derived>
    class WritableTiledTensor : public ReadableTiledTensor<Derived> {
    public:

      typedef ReadableTiledTensor<Derived> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::pmap_interface pmap_interface;
      typedef typename base::trange_type trange_type;
      typedef typename base::value_type value_type;
      typedef typename base::const_reference const_reference;
      typedef typename base::const_iterator const_iterator;
      typedef typename TensorTraits<Derived>::iterator iterator;

      using base::derived;
      using base::range;
      using base::size;
      using base::vars;
      using base::owner;
      using base::is_local;
      using base::is_zero;
      using base::get_world;
      using base::get_pmap;
      using base::is_dense;
      using base::get_shape;
      using base::trange;
      using base::operator[];
      using base::begin;
      using base::end;

      // iterator factory
      iterator begin() { derived().begin(); }
      iterator end() { derived().end(); }

      // element access
      void set(size_type i, const value_type& v) { return derived().set(i, v); }
      void set(size_type i, const madness::Future<value_type>& v) { return derived().set(i, v); }

    }; // class WritableTiledTensor

  } // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_ARRAY_BASE_H__INCLUDED
