#ifndef TILEDARRAY_TENSOR_BASE_H__INCLUDED
#define TILEDARRAY_TENSOR_BASE_H__INCLUDED

#include <TiledArray/coordinate_system.h>

#define TILEDARRAY_TENSOR_BASE_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      typedef BASE base; \
      typedef typename base::size_type size_type; \
      typedef typename base::range_type range_type;

#define TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      TILEDARRAY_TENSOR_BASE_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      typedef typename base::value_type value_type; \
      typedef typename base::const_reference const_reference; \
      typedef typename base::const_iterator const_iterator;

#define TILEDARRAY_WRITABLE_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      typedef typename base::reference reference; \
      typedef typename base::iterator iterator;

#define TILEDARRAY_DIRECT_READABLE_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      typedef typename base::difference_type difference_type; \
      typedef typename base::const_pointer const_pointer;

#define TILEDARRAY_DIRECT_WRITABLE_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      TILEDARRAY_WRITABLE_TENSOR_INHEIRATE_TYPEDEF( BASE , DERIVED ) \
      typedef typename base::difference_type difference_type; \
      typedef typename base::const_pointer const_pointer; \
      typedef typename base::pointer pointer;

#define TILEDARRAY_TENSOR_BASE_INHEIRATE_MEMBER( BASE , DERIVED ) \
    using base::derived; \
    using base::range; \
    using base::size;

#define TILEDARRAY_READABLE_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_TENSOR_BASE_INHEIRATE_MEMBER( BASE , DERIVED ) \
    using base::operator[]; \
    using base::begin; \
    using base::end;

#define TILEDARRAY_WRITABLE_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED )

#define TILEDARRAY_DIRECT_READABLE_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_READABLE_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    using base::data;

#define TILEDARRAY_DIRECT_WRITABLE_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED ) \
    TILEDARRAY_WRITABLE_TENSOR_INHEIRATE_MEMBER( BASE , DERIVED )

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
      typedef typename TensorTraits<Derived>::size_type size_type;
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
      void add_to(Dest& dest) const {
        // This is the default implementation,
        // derived class can reimplement it in a more optimized way.
        TA_ASSERT(size() == dest.size());
        typename Dest::storage_type temp(size());
        eval_to(temp);
        dest += temp;
      }

      template<typename Dest>
      void sub_to(Dest& dest) const {
        // This is the default implementation,
        // derived class can reimplement it in a more optimized way.
        typename Dest::storage_type temp(size());
        eval_to(temp);
        dest -= temp;
      }

    }; // class TensorBase

    template <typename Derived>
    class ReadableTensor : public TensorBase<Derived> {
    public:
      TILEDARRAY_TENSOR_BASE_INHEIRATE_TYPEDEF(TensorBase<Derived>, Derived)
      typedef typename TensorTraits<Derived>::value_type value_type;
      typedef typename TensorTraits<Derived>::const_reference const_reference;
      typedef typename TensorTraits<Derived>::const_iterator const_iterator;

      TILEDARRAY_TENSOR_BASE_INHEIRATE_MEMBER(TensorBase<Derived>, Derived)

      // element access
      const_reference operator[](size_type i) const { return derived()[i]; }

      // iterator factory
      const_iterator begin() const { return derived().begin(); }
      const_iterator end() const { return derived().end(); }

      void check_dependency(madness::TaskInterface* task) const { derived().check_dependency(task); }

    }; // class ReadableTensor

    template <typename Derived>
    class WritableTensor : public ReadableTensor<Derived> {
    public:
      TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF(ReadableTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::reference reference;
      typedef typename TensorTraits<Derived>::iterator iterator;

      TILEDARRAY_READABLE_TENSOR_INHEIRATE_MEMBER(ReadableTensor<Derived>, Derived)

      reference operator[](size_type i) { return derived()[i]; }

      // iterator factory
      iterator begin() { return derived().begin(); }
      iterator end() { return derived().end(); }
    }; // class WritableTensor


    template <typename Derived>
    class DirectReadableTensor : public ReadableTensor<Derived> {
    public:
      TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF(ReadableTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::difference_type difference_type;
      typedef typename TensorTraits<Derived>::const_pointer const_pointer;

      TILEDARRAY_READABLE_TENSOR_INHEIRATE_MEMBER(ReadableTensor<Derived>, Derived)

      // data accessor
      const_pointer data() const { return derived().data(); }
    }; // class DirectReadableTensor

    template <typename Derived>
    class DirectWritableTensor : public WritableTensor<Derived> {
    public:
      TILEDARRAY_WRITABLE_TENSOR_INHEIRATE_TYPEDEF(WritableTensor<Derived>, Derived)
      typedef typename TensorTraits<Derived>::difference_type difference_type;
      typedef typename TensorTraits<Derived>::const_pointer const_pointer;
      typedef typename TensorTraits<Derived>::pointer pointer;

      TILEDARRAY_WRITABLE_TENSOR_INHEIRATE_MEMBER(WritableTensor<Derived>, Derived)

      // data accessor
      const_pointer data() const { return base::data(); }
      pointer data() { return derived().data(); }
    }; // class DirectWritableTensor

  } // namespace expressions
}  // namespace TiledArray
#endif // TILEDARRAY_TENSOR_BASE_H__INCLUDED
