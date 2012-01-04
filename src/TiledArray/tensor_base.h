#ifndef TILEDARRAY_TENSOR_BASE_H__INCLUDED
#define TILEDARRAY_TENSOR_BASE_H__INCLUDED

#include <cstdlib>
#include <TiledArray/error.h>

namespace TiledArray {
  namespace expressions {

    template <typename> struct TensorTraits;
    template <typename T> struct Eval { typedef const T& type; };

    /// Tensor base

    /// This class is the base class for all tensor expressions. It uses CRTP
    /// to avoid the overhead that is normally associated with virtual classes.
    /// \tparam Derived The derived class type
    template <typename Derived>
    class TensorBase {
    public:
      typedef std::size_t size_type;
      typedef typename TensorTraits<Derived>::range_type range_type;
      typedef typename Eval<Derived>::type eval_type;

      // Access this object type
      inline Derived& derived() { return *static_cast<Derived*>(this); }
      inline const Derived& derived() const { return *static_cast<const Derived*>(this); }

      // dimension information
      inline const range_type& range() const { return derived().range(); }
      inline size_type size() const { return derived().size(); }

      inline eval_type eval() const { derived().eval(); }

    }; // class TensorBase

    template <typename Derived>
    class ReadableTensor : public TensorBase<Derived> {
    public:
      typedef TensorBase<Derived> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;

      using base::derived;
      using base::range;
      using base::size;

      /// Evaluate this tensor to another

      /// This tensor is evaluated and written to \c dest . \c dest[i] must be
      /// return a non-const reference to the i-th element and \c value_type
      /// must be implicitly convertible to the value type of \c dest .
      /// \tparam Dest The destination object type.
      /// \param dest The destination object that will hold the evaluated tensor.
      /// \note This is the default implementation, derived class can reimplement
      /// it in a more optimized way.
      template <typename Dest>
      inline void eval_to(Dest& dest) const {
        const size_type s = size();
        TA_ASSERT(s == dest.size());
        for(size_type i = 0; i < s; ++i)
          dest[i] = derived()[i];
      }

      /// Add this tensor to another

      /// This tensor is evaluated and added to \c dest . \c dest[i] must be
      /// return a non-const reference to the i-th element and \c value_type
      /// must be implicitly convertible to the value type of \c dest .
      /// \tparam Dest The destination object type.
      /// \param dest The destination object that will hold the evaluated tensor.
      /// \note This is the default implementation, derived class can reimplement
      /// it in a more optimized way.
      template<typename Dest>
      inline void add_to(Dest& dest) const {
        // This is the default implementation,
        // derived class can reimplement it in a more optimized way.
        const size_type s = size();
        TA_ASSERT(s == dest.size());
        for(size_type i = 0; i < s; ++i)
          dest[i] += derived()[i];
      }

      /// Evaluate this tensor to another

      /// This tensor is evaluated and written to \c dest . \c dest[i] must be
      /// return a non-const reference to the i-th element and \c value_type
      /// must be implicitly convertible to the value type of \c dest .
      /// \tparam Dest The destination object type.
      /// \param dest The destination object that will hold the evaluated tensor.
      /// \note This is the default implementation, derived class can reimplement
      /// it in a more optimized way.
      template<typename Dest>
      inline void sub_to(Dest& dest) const {
        // This is the default implementation,
        // derived class can reimplement it in a more optimized way.
        const size_type s = size();
        TA_ASSERT(s == dest.size());
        for(size_type i = 0; i < s; ++i)
          dest[i] -= derived()[i];
      }

      /// Element accessor

      /// \param i The element to be accessed
      /// \return const reference to i-th element of this tensor
      inline const_reference operator[](size_type i) const { return derived()[i]; }

    }; // class ReadableTensor

    template <typename Derived>
    class DirectReadableTensor : public ReadableTensor<Derived> {
    public:
      typedef ReadableTensor<Derived> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::value_type value_type;
      typedef typename base::const_reference const_reference;
      typedef typename TensorTraits<Derived>::difference_type difference_type;
      typedef typename TensorTraits<Derived>::const_iterator const_iterator;
      typedef typename TensorTraits<Derived>::const_pointer const_pointer;

      using base::derived;
      using base::range;
      using base::size;
      using base::eval_to;
      using base::add_to;
      using base::sub_to;
      using base::operator[];

      // iterator factory
      inline const_iterator begin() const { return derived().begin(); }
      inline const_iterator end() const { return derived().end(); }

      // data accessor
      inline const_pointer data() const { return derived().data(); }
    }; // class DirectReadableTensor

    template <typename Derived>
    class DirectWritableTensor : public DirectReadableTensor<Derived> {
    public:
      typedef DirectReadableTensor<Derived> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::value_type value_type;
      typedef typename base::const_reference const_reference;
      typedef typename base::difference_type difference_type;
      typedef typename base::const_iterator const_iterator;
      typedef typename base::const_pointer const_pointer;
      typedef typename TensorTraits<Derived>::reference reference;
      typedef typename TensorTraits<Derived>::iterator iterator;
      typedef typename TensorTraits<Derived>::pointer pointer;

      using base::derived;
      using base::range;
      using base::size;
      using base::eval_to;
      using base::add_to;
      using base::sub_to;
      using base::operator[];
      using base::data;
      using base::begin;
      using base::end;

      template <typename D>
      Derived& operator +=(const ReadableTensor<D>& other) {
        other.derived().add_to(derived());
        return derived();
      }

      template <typename D>
      DirectWritableTensor<Derived>& operator -=(const ReadableTensor<D>& other) {
        other.derived().sub_to(derived());
        return *this;
      }

      // iterator factory
      inline iterator begin() { return derived().begin(); }
      inline iterator end() { return derived().end(); }

      // data accessor
      inline reference operator[](size_type i) { return derived()[i]; }
      inline pointer data() { return derived().data(); }
    }; // class DirectWritableTensor

  } // namespace expressions
}  // namespace TiledArray
#endif // TILEDARRAY_TENSOR_BASE_H__INCLUDED
