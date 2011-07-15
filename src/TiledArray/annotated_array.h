#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/variable_list.h>
#include <world/sharedptr.h>

namespace TiledArray {
  namespace expressions {

    template <typename>
    class AnnotatedArray;
    template <typename>
    class Expression;
    template <typename T>
    void swap(AnnotatedArray<T>&, AnnotatedArray<T>&);

    /// AnnotatedArray for array objects.

    /// This object binds array dimension annotations to an array object.
    /// \tparam T The array object type.
    template <typename T>
    class AnnotatedArray {
    public:
      typedef AnnotatedArray<T>                         AnnotatedArray_;

      typedef typename T::coordinate_system             coordinate_system;

      typedef typename coordinate_system::volume_type   volume_type;
      typedef typename coordinate_system::index         index;
      typedef typename coordinate_system::ordinal_index ordinal_index;
      typedef typename coordinate_system::size_array    size_array;

      typedef T array_type;
      typedef typename array_type::value_type           value_type;
      typedef typename array_type::iterator             iterator;
      typedef typename array_type::const_iterator       const_iterator;
      typedef typename array_type::range_type           range_type;

    private:
      AnnotatedArray();

    public:
      /// Primary constructor

      /// \param a A const reference to an array_type object
      /// \param v A const reference to a variable list object
      /// \throw std::runtime_error When the dimensions of the array and
      /// variable list are not equal.
      AnnotatedArray(array_type& a, const VariableList& v) :
          array_(a), vars_(v)
      {
        TA_ASSERT(array_type::coordinate_system::dim == v.dim(), std::runtime_error,
            "The dimensions of the array do not match the dimensions of the variable list.");
      }

      /// Default constructor

      /// \param a A const reference to an array_type object
      /// \param v A const reference to a variable list object
      /// \throw std::runtime_error When the dimensions of the array and
      /// variable list are not equal.
      AnnotatedArray(const array_type& a, const VariableList& v) :
          array_(a), vars_(v)
      {
        TA_ASSERT(array_type::coordinate_system::dim == v.dim(), std::runtime_error,
            "The dimensions of the array do not match the dimensions of the variable list.");
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

      /// Assign the result of the given expression to this object.

      /// The expression is evaluated and the results are assigned the array
      /// here.
      /// \tparam ExpType The type of the expression
      /// \param exp The expression that will be evaluated and assigned to the array
      /// \return This annotated array object
      template <typename ExpType>
      AnnotatedArray_& operator =(const Expression<ExpType>& exp) {
        return exp.eval(*this);
      }

      /// Array object accessor

      /// \return A reference to the array object
      array_type& array() { return array_; }

      /// Array object const accessor

      /// \return A const reference to the array object
      const array_type& array() const { return array_; }

      /// Array range accessor

      /// \return A const reference to the range object.
      const range_type& range() const { return array_.range(); }

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

      void swap(AnnotatedArray& other) {
        std::swap(array_, other.array_);
        std::swap(vars_, other.vars_);
      }

    private:

      array_type array_;  ///< pointer to the array object
      VariableList vars_; ///< variable list
    }; // class AnnotatedArray

    /// Exchange the values of a0 and a1.
    template<typename T>
    void swap(AnnotatedArray<T>& a0, AnnotatedArray<T>& a1) {
      a0.swap(a1);
    }

  } // namespace expressions
} //namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
