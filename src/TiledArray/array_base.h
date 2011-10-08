#ifndef TILEDARRAY_ARRAY_BASE_H__INCLUDED
#define TILEDARRAY_ARRAY_BASE_H__INCLUDED

// This needs to be defined before world/worldreduce.h
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/coordinate_system.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/tensor_base.h>
#include <TiledArray/bitset.h>
#include <TiledArray/future_tensor.h>
#include <TiledArray/eval_task.h>
#include <TiledArray/eval_tensor.h>
#include <world/worldtypes.h>
#include <world/shared_ptr.h>
#include <world/worldreduce.h>

#define TILEDARRAY_ANNOTATED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED )  \
    TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF( BASE , DERIVED )

#define TILEDARRAY_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_ANNOTATED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::wobj_type wobj_type; \
    typedef typename base::pmap_interface pmap_interface; \
    typedef typename base::trange_type trange_type;

#define TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_TILED_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::value_type value_type; \
    typedef typename base::remote_type remote_type; \
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
    using base::get_shape;

#define TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_TILED_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::get_local; \
    using base::get_remote; \
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

    template <typename Derived>
    class AnnotatedTensor : public TensorBase<Derived> {
    public:

      TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF(TensorBase<Derived>, Derived)
      TILEDARRAY_TENSOR_BASE_INHERIT_MEMBER(TensorBase<Derived>, Derived)

      inline const VariableList& vars() const { return derived().vars(); }

    }; // class AnnotatedTensor

    template <typename Derived>
    class TiledTensor : public AnnotatedTensor<Derived>, public madness::WorldReduce<Derived> {
    public:

      TILEDARRAY_ANNOTATED_TENSOR_INHERIT_TYPEDEF(AnnotatedTensor<Derived>, Derived)
      typedef madness::WorldReduce<Derived> wobj_type;
      typedef madness::WorldDCPmapInterface<size_type> pmap_interface;
      typedef typename TensorTraits<Derived>::trange_type trange_type;

      TILEDARRAY_ANNOTATED_TENSOR_INHERIT_MEMBER(AnnotatedTensor<Derived>, Derived)

      TiledTensor(madness::World& world) : wobj_type(world) { }

      void process_pending() { wobj_type::process_pending(); }

      // Tile locality info
      inline ProcessID owner(size_type i) const { return derived().owner(i); }
      inline bool is_local(size_type i) const { return derived().is_local(i); }
      inline bool is_zero(size_type i) const { return derived().is_zero(i); }
      inline madness::World& get_world() const { return wobj_type::get_world(); }
      inline std::shared_ptr<pmap_interface> get_pmap() const { return derived().get_pmap(); }
      inline bool is_dense() const { return derived().is_dense(); }
      inline const TiledArray::detail::Bitset<>& get_shape() const { return derived().get_shape(); }
      inline trange_type trange() const { return derived().trange(); }

    }; // class TiledTensor

    template <typename Derived>
    class ReadableTiledTensor : public TiledTensor<Derived> {
    public:

      TILEDARRAY_TILED_TENSOR_INHERIT_TYPEDEF(TiledTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef EvalTensor<typename value_type::value_type> eval_tensor;
      typedef FutureTensor<eval_tensor> remote_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;
      typedef typename TensorTraits<Derived>::const_iterator const_iterator;

      TILEDARRAY_TILED_TENSOR_INHERIT_MEMBER(TiledTensor<Derived>, Derived)

      ReadableTiledTensor(madness::World& world) : base(world) { }

      // element access
      const_reference get_local(size_type i) const { return derived().get_local(i); }

      remote_type get_remote(size_type i) const {
        TA_ASSERT(! is_local(i));
        madness::Future<eval_tensor> result;
        wobj_type::task(owner(i), & get_remote_handler, i, result.remote_ref(get_world()),
            madness::TaskAttributes::hipri());
        return remote_type(result);
      }

      // iterator factory
      const_iterator begin() const { return derived().begin(); }
      const_iterator end() const { return derived().end(); }

    private:

      /// Task function for remote tensor fetch

      /// \tparam i The tensor index
      madness::Void get_remote_handler(size_type i,
          const madness::RemoteReference<madness::FutureImpl<eval_tensor> >& ref) {
        TA_ASSERT(is_local(i));
        // Construct a task to evaluate the local tensor.
        typedef detail::EvalTask<eval_tensor, value_type> eval_task;
        eval_task* task = new eval_task(get_local(i), madness::Future<eval_tensor>(ref));
        try {
          get_world().taskq.add(task);
        } catch(...) {
          delete task;
          throw;
        }

        return madness::None;
      }

    }; // class ReadableTiledTensor

    template <typename Derived>
    class WritableTiledTensor : public ReadableTiledTensor<Derived> {
    public:

      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<Derived>, Derived)

      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_MEMBER(ReadableTiledTensor<Derived>, Derived)

      WritableTiledTensor(madness::World& world) : base(world) { }

      // element access
      void set(size_type i, const value_type& v) { return derived().insert(i, v); }

    }; // class WritableTiledTensor

  } // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_ARRAY_BASE_H__INCLUDED
