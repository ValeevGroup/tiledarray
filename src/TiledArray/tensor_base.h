#ifndef TILEDARRAY_TENSOR_BASE_H__INCLUDED
#define TILEDARRAY_TENSOR_BASE_H__INCLUDED

#include <TiledArray/coordinate_system.h>

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
    using base::operator[];

#define TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    using base::data; \
    using base::begin; \
    using base::end;

#define TILEDARRAY_DIRECT_WRITABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_MEMBER( BASE , DERIVED )

namespace madness {
  class TaskInterface;
}  // namespace madness

namespace TiledArray {
  namespace expressions {

    template <typename> struct TensorTraits;
    template <typename> struct Eval;

    template <typename T>
    struct TensorArg {
      typedef const T& type;
    };

    template <typename T>
    struct TensorMem {
      typedef const T type;
    };


    template <typename Derived>
    class TensorBase {
    public:
      typedef std::size_t size_type;
      typedef typename TensorTraits<Derived>::range_type range_type;

      // Access this object type
      Derived& derived() { return *static_cast<Derived*>(this); }
      const Derived& derived() const { return *static_cast<const Derived*>(this); }

      // dimension information
      const range_type& range() const { return derived().range(); }
      size_type size() const { return derived().size(); }

      typename Eval<Derived>::type eval() const { derived().eval(); }

      template<typename Dest>
      void eval_to(Dest& dest) const { derived().eval_to(dest); }

      template<typename Dest>
      void add_to(Dest& dest) const { derived().add_to(dest); }

      template<typename Dest>
      void sub_to(Dest& dest) const { derived().sub_to(dest); }

    }; // class TensorBase

    template <typename Derived>
    class ReadableTensor : public TensorBase<Derived> {
    public:
      TILEDARRAY_TENSOR_BASE_INHERIT_TYPEDEF(TensorBase<Derived>, Derived)
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;

      TILEDARRAY_TENSOR_BASE_INHERIT_MEMBER(TensorBase<Derived>, Derived)

      template <typename Dest>
      void eval_to(Dest& dest) const {
        const size_type s = size();
        TA_ASSERT(s == dest.size());
        for(size_type i = 0; i < s; ++i)
          dest[i] = derived()[i];
      }

      template<typename Dest>
      void add_to(Dest& dest) const {
        // This is the default implementation,
        // derived class can reimplement it in a more optimized way.
        const size_type s = size();
        TA_ASSERT(s == dest.size());
        for(size_type i = 0; i < s; ++i)
          dest[i] += derived()[i];
      }

      template<typename Dest>
      void sub_to(Dest& dest) const {
        // This is the default implementation,
        // derived class can reimplement it in a more optimized way.
        const size_type s = size();
        TA_ASSERT(s == dest.size());
        for(size_type i = 0; i < s; ++i)
          dest[i] -= derived()[i];
      }

      // element access
      const_reference operator[](size_type i) const { return derived()[i]; }

      void check_dependency(madness::TaskInterface* task) const { derived().check_dependency(task); }

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
      const_iterator begin() const { return derived().begin(); }
      const_iterator end() const { return derived().end(); }

      // data accessor
      const_pointer data() const { return derived().data(); }
    }; // class DirectReadableTensor

    template <typename Derived>
    class DirectWritableTensor : public DirectReadableTensor<Derived> {
    public:
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF(DirectReadableTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::reference reference;
      typedef typename TensorTraits<Derived>::iterator iterator;
      typedef typename TensorTraits<Derived>::pointer pointer;

      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_MEMBER(DirectReadableTensor<Derived>, Derived)

      // iterator factory
      iterator begin() { return derived().begin(); }
      iterator end() { return derived().end(); }

      // data accessor
      reference operator[](size_type i) { return derived()[i]; }
      pointer data() { return derived().data(); }
    }; // class DirectWritableTensor

  } // namespace expressions
}  // namespace TiledArray
#endif // TILEDARRAY_TENSOR_BASE_H__INCLUDED
