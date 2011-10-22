#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/array_base.h>
#include <TiledArray/variable_list.h>
#include <world/shared_ptr.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename> class AnnotatedArray;

    template <typename T>
    struct TensorTraits<AnnotatedArray<T> > {
      typedef typename T::range_type range_type;
      typedef typename T::trange_type trange_type;
      typedef typename T::value_type value_type;
      typedef typename T::future const_reference;
      typedef typename T::const_iterator const_iterator;
    }; //  struct TensorTraits<AnnotatedArray<T> >

    /// Wrapper object that adds annotation to tiled tensor objects.

    /// \tparam T The array object type.
    template <typename T>
    class AnnotatedArray : public ReadableTiledTensor<AnnotatedArray<T> > {
    public:
      typedef AnnotatedArray<T>               AnnotatedArray_; ///< This object type
      TILEDARRAY_WRITABLE_TILED_TENSOR_INHERIT_TYPEDEF(WritableTiledTensor<AnnotatedArray<T> >, AnnotatedArray<T>)
      typedef typename T::coordinate_system   coordinate_system; ///< The array coordinate system type
      typedef T                               array_type; ///< The array type

    private:
      // not allowed
      AnnotatedArray_& operator =(const AnnotatedArray_& other);

    public:

      /// Constructor

      /// \param a A const reference to an array_type object
      /// \param v A const reference to a variable list object
      /// \throw std::runtime_error When the dimensions of the array and
      /// variable list are not equal.
      AnnotatedArray(const array_type& a, const VariableList& v) :
          array_(a), vars_(v)
      {
        TA_ASSERT(a.range().dim() == v.dim());
      }


      AnnotatedArray(const AnnotatedArray_& other) :
          array_(other.array_), vars_(other.vars_)
      { }

      const AnnotatedArray_& eval() const { return *this; }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(range() == dest.range());

        // Add result tiles to dest
        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
      }

      /// Tensor tile range object accessor

      /// \return A const reference to the tensor range object
      const range_type& range() const { return array_.range(); }

      /// Tensor tile size accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return array_.range().volume(); }

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
      const_reference operator[](size_type i) const { return array_.find(i); }

      /// Array object accessor

      /// \return A reference to the array object
      array_type& array() { return array_; }

      /// Array object const accessor

      /// \return A const reference to the array object
      const array_type& array() const { return array_; }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return array_.begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return array_.end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return vars_; }

      madness::World& get_world() const { return array_.get_world(); }

    private:
      const array_type& array_;  ///< pointer to the array object
      VariableList vars_; ///< variable list
    }; // class AnnotatedArray


  } // namespace expressions
} //namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
