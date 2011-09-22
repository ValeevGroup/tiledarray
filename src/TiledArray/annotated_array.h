#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/array_base.h>
#include <TiledArray/future_tensor.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/eval_task.h>
#include <world/shared_ptr.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename> class AnnotatedArray;

    namespace detail {

      template <typename T>
      struct MakeFutTensor {
        typedef const madness::Future<T>& argument_type;
        typedef FutureTensor<T> result_type;

        result_type operator()(argument_type future) const {
          return result_type(future);
        }
      }; // struct MakeFutTensor

    }  // namespace detail

    template <typename T>
    struct TensorTraits<AnnotatedArray<T> > {
      typedef typename T::size_type size_type;
      typedef typename T::size_array size_array;
      typedef typename T::trange_type trange_type;
      typedef FutureTensor<typename T::value_type> value_type;
      typedef value_type const_reference;
      typedef value_type remote_type;
      typedef TiledArray::detail::UnaryTransformIterator<typename T::const_iterator,
          detail::MakeFutTensor<typename T::value_type> > const_iterator;
    }; //  struct TensorTraits<AnnotatedArray<T> >

    template <typename T>
    struct Eval<AnnotatedArray<T> > {
      typedef const AnnotatedArray<T>& type;
    }; // struct Eval<AnnotatedArray<T> >

    /// Wrapper object that adds annotation to tiled tensor objects.

    /// \tparam T The array object type.
    template <typename T>
    class AnnotatedArray : public WritableTiledTensor<AnnotatedArray<T> > {
    public:
      typedef AnnotatedArray<T>               AnnotatedArray_; ///< This object type
      TILEDARRAY_WRITABLE_TILED_TENSOR_INHEIRATE_TYPEDEF(WritableTiledTensor<AnnotatedArray<T> >, AnnotatedArray<T>)
      typedef typename T::coordinate_system   coordinate_system; ///< The array coordinate system type
      typedef T                               array_type; ///< The array type

    private:
      // not allowed
      AnnotatedArray_& operator =(const AnnotatedArray_& other);

      typedef detail::MakeFutTensor<typename T::value_type> transform_op; ///< The tile transform operation type

    public:
      /// Constructor

      /// \param a A const reference to an array_type object
      /// \param v A const reference to a variable list object
      /// \throw std::runtime_error When the dimensions of the array and
      /// variable list are not equal.
      AnnotatedArray(const array_type& a, const VariableList& v) :
          array_(a), vars_(v), op_()
      {
        TA_ASSERT(array_type::coordinate_system::dim == v.dim());
      }

      /// Copy constructor

      /// \param other The AnnotatedArray to be copied
      AnnotatedArray(const AnnotatedArray_& other) :
          array_(other.array_), vars_(other.vars_), op_(other.op_)
      { }

      /// Evaluate tensor

      /// \return The evaluated tensor
      const AnnotatedArray_& eval() const { return *this; }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(dim() == dest.dim());
        TA_ASSERT(std::equal(size().begin(), size().end(), dest.size().begin()));

        // Add result tiles to dest and wait for all tiles to be added.
        madness::Future<bool> done =
            get_world().taskq.for_each(madness::Range<const_iterator>(begin(),
            end(), 8), detail::EvalTo<Dest, const_iterator>(dest));
        done.get();
      }

      /// Tensor dimension accessor

      /// \return The number of dimension in the tensor
      unsigned int dim() const { return coordinate_system::dim; }


      /// Tensor data and tile ordering accessor

      /// \return The tensor data and tile ordering
      TiledArray::detail::DimensionOrderType order() const { return coordinate_system::order; }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const size_array& size() const { return array_.range().size(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type volume() const { return array_.range().volume(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const { return array_.owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return array_.is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const { return array_.is_zero(i); }

      /// World object accessor

      /// \return A reference to the world where tensor lives
      madness::World& get_world() const { return array_.get_world(); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return array_.get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return array_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return array_.get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return array_.trange(); }

      /// Local tile accessor

      /// Access a tensor that is available locally.
      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When \c is_local(i) returns \c false .
      const_reference get_local(size_type i) const {
        TA_ASSERT(is_local(i));
        return op_(array_.find(i));
      }

      /// Remote tile accessor

      /// Access a tensor that is available remotely.
      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When \c is_local(i) returns \c true .
      remote_type get_remote(size_type i) const {
        TA_ASSERT(! is_local(i));
        return op_(array_.find(i));
      }

      /// Array object accessor

      /// \return A reference to the array object
      array_type& array() { return array_; }

      /// Array object const accessor

      /// \return A const reference to the array object
      const array_type& array() const { return array_; }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return const_iterator(array_.begin(), op_); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return const_iterator(array_.end(), op_); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return vars_; }

      void set(size_type i, const value_type& tile) { array_.set(i, tile); }

    private:
      array_type array_;  ///< pointer to the array object
      VariableList vars_; ///< variable list
      transform_op op_;   ///< Tile conversion operator
    }; // class AnnotatedArray

    /// AnnotatedArray for array objects.

    /// This object binds array dimension annotations to an array object.
    /// \tparam T The array object type.
    template <typename T>
    class AnnotatedArray<const T> : public ReadableTiledTensor<AnnotatedArray<const T> > {
    public:
      typedef AnnotatedArray<const T>                         AnnotatedArray_;

      TILEDARRAY_WRITABLE_TILED_TENSOR_INHEIRATE_TYPEDEF(ReadableTiledTensor<AnnotatedArray_>, AnnotatedArray_)

      typedef typename T::coordinate_system             coordinate_system;
      typedef T                                         array_type;

    private:
      // not allowed
      AnnotatedArray_& operator =(const AnnotatedArray_& other);

      typedef detail::MakeFutTensor<typename T::value_type> transform_op; ///< The tile transform operation type

    public:
      /// Constructor

      /// \param a A const reference to an array_type object
      /// \param v A const reference to a variable list object
      /// \throw std::runtime_error When the dimensions of the array and
      /// variable list are not equal.
      AnnotatedArray(const array_type& a, const VariableList& v) :
          array_(a), vars_(v), op_()
      {
        TA_ASSERT(array_type::coordinate_system::dim == v.dim());
      }

      /// Copy constructor

      /// \param other The AnnotatedArray to be copied
      AnnotatedArray(const AnnotatedArray_& other) :
          array_(other.array_), vars_(other.vars_), op_(other.op_)
      { }

      /// Evaluate tensor

      /// \return The evaluated tensor
      const AnnotatedArray_& eval() const { return *this; }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(dim() == dest.dim());
        TA_ASSERT(std::equal(size().begin(), size().end(), dest.size().begin()));

        // Add result tiles to dest and wait for all tiles to be added.
        madness::Future<bool> done =
            get_world().taskq.for_each(madness::Range<const_iterator>(begin(),
            end(), 8), detail::EvalTo<Dest, const_iterator>(dest));
        done.get();
      }

      /// Tensor dimension accessor

      /// \return The number of dimension in the tensor
      unsigned int dim() const { return coordinate_system::dim; }


      /// Tensor data and tile ordering accessor

      /// \return The tensor data and tile ordering
      TiledArray::detail::DimensionOrderType order() const { return coordinate_system::order; }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const size_array& size() const { return array_.range().size(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type volume() const { return array_.range().volume(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const { return array_.owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return array_.is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const { return array_.is_zero(i); }

      /// World object accessor

      /// \return A reference to the world where tensor lives
      madness::World& get_world() const { return array_.get_world(); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return array_.get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return array_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return array_.get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return array_.trange(); }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const { return op_(array_.find(i)); }

      /// Array object accessor

      /// \return A reference to the array object
      array_type& array() { return array_; }

      /// Array object const accessor

      /// \return A const reference to the array object
      const array_type& array() const { return array_; }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return const_iterator(array_.begin(), op_); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return const_iterator(array_.end(), op_); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return vars_; }

    private:
      array_type array_;  ///< pointer to the array object
      VariableList vars_; ///< variable list
      transform_op op_;   ///< Tile conversion operator
    }; // class AnnotatedArray<const T>

  } // namespace expressions
} //namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
