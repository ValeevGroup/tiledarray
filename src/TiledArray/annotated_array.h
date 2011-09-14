#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/array_base.h>
#include <TiledArray/future_tensor.h>
#include <world/sharedptr.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename> class AnnotatedArray;
    template <typename T>
    void swap(AnnotatedArray<T>&, AnnotatedArray<T>&);

    template <typename T>
    struct TensorTraits<AnnotatedArray<T> > {
      typedef typename T::size_type size_type;
      typedef typename T::size_array size_array;
      typedef typename T::trange_type trange_type;
      typedef FutureTensor<typename T::value_type> value_type;
      typedef value_type const_reference;
      typedef value_type reference;
      typedef typename T::const_iterator const_iterator;
      typedef typename T::iterator iterator;
    }; //  struct TensorTraits<AnnotatedArray<T> >

    template <typename T>
    struct TensorTraits<AnnotatedArray<const T> > {
      typedef typename T::size_type size_type;
      typedef typename T::size_array size_array;
      typedef typename T::trange_type trange_type;
      typedef typename T::value_type value_type;
      typedef typename T::const_reference const_reference;
      typedef typename T::const_iterator const_iterator;
    }; // struct TensorTraits<AnnotatedArray<const T> >

    template <typename T>
    struct Eval<AnnotatedArray<T> > {
      typedef const AnnotatedArray<T>& type;
    }; // struct Eval<AnnotatedArray<T> >

    /// AnnotatedArray for array objects.

    /// This object binds array dimension annotations to an array object.
    /// \tparam T The array object type.
    template <typename T>
    class AnnotatedArray : public WritableTiledTensor<AnnotatedArray<T> > {
    public:
      typedef AnnotatedArray<T>                         AnnotatedArray_;

      TILEDARRAY_WRITABLE_TILED_TENSOR_INHEIRATE_TYPEDEF(WritableTiledTensor<AnnotatedArray<T> >, AnnotatedArray<T>)

      typedef typename T::coordinate_system             coordinate_system;
      typedef T                                         array_type;

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
        TA_ASSERT(array_type::coordinate_system::dim == v.dim());
      }

      /// Copy constructor

      /// \param other The AnnotatedArray to be copied
      AnnotatedArray(const AnnotatedArray_& other) :
          array_(other.array_), vars_(other.vars_)
      { }

      /// Destructor
      ~AnnotatedArray() { }

      // dimension information
      unsigned int dim() const { return coordinate_system::dim; }
      TiledArray::detail::DimensionOrderType order() const { return coordinate_system::order; }
      const size_array& size() const { return array_.range().size(); }
      size_type volume() const { return array_.range().volume(); }


      // Tile locality info
      ProcessID owner(size_type i) const { return array_.owner(i); }
      bool is_local(size_type i) const { return array_.is_local(i); }
      bool is_zero(size_type i) const { return array_.is_zero(i); }

      madness::World& get_world() const { return array_.get_world(); }
      std::shared_ptr<pmap_interface> get_pmap() const { return array_.get_pmap(); }

      bool is_dense() const { return array_.is_dense(); }
      const TiledArray::detail::Bitset<>& get_shape() const { return array_.get_shape(); }

      // Tile dimension info
      size_array size(size_type i) const { return array_.make_range(i).size(); }
      size_type volume(size_type i) const { return array_.make_range(i).volume(); }
      trange_type trange() const { return array_.trange(); }

      reference operator[](size_type i) { return value_type(array_.find(i)); }
      const_reference operator[](size_type i) const { return value_type(array_.find(i)); }

      const AnnotatedArray_& eval() const { return *this; }

      /// Array object accessor

      /// \return A reference to the array object
      array_type& array() { return array_; }

      /// Array object const accessor

      /// \return A const reference to the array object
      const array_type& array() const { return array_; }


      /// Array begin iterator

      /// \return An iterator to the first element of the array.
      iterator begin() { return array_.begin(); }


      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return array_.begin(); }

      /// Array end iterator

      /// \return An iterator to one past the last element of the array.
      iterator end() { return array_.end(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return array_.end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return vars_; }

    private:

      array_type array_;  ///< pointer to the array object
      VariableList vars_; ///< variable list
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


      // dimension information
      unsigned int dim() const { return coordinate_system::dim; }
      TiledArray::detail::DimensionOrderType order() const { return coordinate_system::order; }
      const size_array& size() const { return array_.range().size(); }
      size_type volume() const { return array_.range().volume(); }


      // Tile locality info
      ProcessID owner(size_type i) const { return array_.owner(i); }
      bool is_local(size_type i) const { return array_.is_local(i); }
      bool is_zero(size_type i) const { return array_.is_zero(i); }

      // Tile dimension info
      size_array size(size_type i) const { return array_.make_range(i).size(); }
      size_type volume(size_type i) const { return array_.make_range(i).volume(); }

      const_reference operator[](size_type i) const { return array_.find(i); }

      /// Constructor

      /// \param a A const reference to an array_type object
      /// \param v A const reference to a variable list object
      /// \throw std::runtime_error When the dimensions of the array and
      /// variable list are not equal.
      AnnotatedArray(const array_type& a, const VariableList& v) :
          array_(a), vars_(v)
      {
        TA_ASSERT(array_type::coordinate_system::dim == v.dim());
      }

      /// Copy constructor

      /// \param other The AnnotatedArray to be copied
      AnnotatedArray(const AnnotatedArray_& other) :
          array_(other.array_), vars_(other.vars_)
      { }

      /// Destructor
      ~AnnotatedArray() { }

      /// AnnotatedArray assignment operator.

      /// \param other The AnnotatedArray to be copied
      AnnotatedArray_& operator =(const AnnotatedArray_& other) {
        array_ = other.array_;
        vars_ = other.vars_;
        return *this;
      }

      const AnnotatedArray_& eval() const { return *this; }

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

    private:

      array_type array_;  ///< pointer to the array object
      VariableList vars_; ///< variable list
    }; // class AnnotatedArray<const T>

  } // namespace expressions
} //namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
