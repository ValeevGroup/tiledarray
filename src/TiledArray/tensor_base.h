#ifndef TILEDARRAY_TENSOR_BASE_H__INCLUDED
#define TILEDARRAY_TENSOR_BASE_H__INCLUDED

#include <cstdlib>
#include <TiledArray/error.h>

// Inherit
#define TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef BASE base; \
    typedef typename base::size_type size_type; \
    typedef typename base::range_type range_type;

#define TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::value_type value_type; \
    typedef typename base::const_reference const_reference;

#define TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::difference_type difference_type; \
    typedef typename base::const_iterator const_iterator; \
    typedef typename base::const_pointer const_pointer;

#define TILEDARRAY_DIRECT_WRITABLE_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF( BASE , DERIVED ) \
    typedef typename base::reference reference; \
    typedef typename base::iterator iterator; \
    typedef typename base::pointer pointer;

#define TILEDARRAY_TENSOR_BASE_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::derived; \
    using base::range; \
    using base::size;

#define TILEDARRAY_READABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_TENSOR_BASE_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::eval_to; \
    using base::add_to; \
    using base::sub_to; \
    using base::operator[];

#define TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::data; \
    using base::begin; \
    using base::end;

#define TILEDARRAY_DIRECT_WRITABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \

namespace TiledArray {
  namespace expressions {

    template <typename> struct TensorTraits;
    template <typename> struct Eval;

    /// Tensor base

    /// This class is the base class for all tensor expressions. It uses CRTP
    /// to avoid the overhead that is normally associated with virtual classes.
    /// \tparam Derived The derived class type
    template <typename Derived>
    class TensorBase {
    public:
      typedef std::size_t size_type;
      typedef typename TensorTraits<Derived>::range_type range_type;

      // Access this object type
      inline Derived& derived() { return *static_cast<Derived*>(this); }
      inline const Derived& derived() const { return *static_cast<const Derived*>(this); }

      // dimension information
      inline const range_type& range() const { return derived().range(); }
      inline size_type size() const { return derived().size(); }

      inline typename Eval<Derived>::type eval() const { derived().eval(); }

    }; // class TensorBase

    template <typename Derived>
    class ReadableTensor : public TensorBase<Derived> {
    public:
      TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF(TensorBase<Derived>, Derived)
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;

      TILEDARRAY_TENSOR_BASE_INHERIT_MEMBER(TensorBase<Derived>, Derived)

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
      TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF(ReadableTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::difference_type difference_type;
      typedef typename TensorTraits<Derived>::const_iterator const_iterator;
      typedef typename TensorTraits<Derived>::const_pointer const_pointer;

      TILEDARRAY_READABLE_TENSOR_INHERIT_MEMBER(ReadableTensor<Derived>, Derived)

      // iterator factory
      inline const_iterator begin() const { return derived().begin(); }
      inline const_iterator end() const { return derived().end(); }

      // data accessor
      inline const_pointer data() const { return derived().data(); }
    }; // class DirectReadableTensor

    template <typename Derived>
    class DirectWritableTensor : public DirectReadableTensor<Derived> {
    public:
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF(DirectReadableTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::reference reference;
      typedef typename TensorTraits<Derived>::iterator iterator;
      typedef typename TensorTraits<Derived>::pointer pointer;

      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_MEMBER(DirectReadableTensor<Derived>, Derived)

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
