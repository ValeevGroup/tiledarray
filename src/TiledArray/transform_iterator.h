#ifndef TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED
#define TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED

#include <TiledArray/error.h>
#include <typeinfo>
#include <iterator>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/iterator/iterator_facade.hpp>

namespace TiledArray {
  namespace detail {



    /// Polymorphic transform iterator

    /// This iterator will transform an arbitrary iterator with an arbitrary
    /// transformation function. It is typed by the return type of the
    template<typename Value>
    class PolyTransformIterator : public boost::iterator_facade<PolyTransformIterator<Value>,
        Value, std::input_iterator_tag, Value>
    {
      // Convenience typedefs.
      typedef boost::iterator_facade<PolyTransformIterator<Value>, Value, std::input_iterator_tag, Value> iterator_facade_;
      typedef PolyTransformIterator<Value> PolyTransformIterator_;

      // Forward declarations of internal classes
      class HolderBase;
      template<typename Iterator, typename Functor>
      class Holder;

      /// Default constructor not allowed.
      PolyTransformIterator();

    public:
      // Iterator typedefs
      typedef typename iterator_facade_::difference_type difference_type;
      typedef typename iterator_facade_::value_type value_type;
      typedef typename iterator_facade_::pointer pointer;
      typedef typename iterator_facade_::reference reference;
      typedef std::input_iterator_tag iterator_category;

      /// Construct from an iterator
      template<typename Iterator, typename Functor>
      PolyTransformIterator(const Iterator& it, const Functor& f) : holder_(new Holder<Iterator, Functor>(it, f)) {
//        BOOST_STATIC_ASSERT((boost::is_same<typename std::iterator_traits<Iterator>::value_type, value_type>::value));
      }

      /// Copy constructor
      PolyTransformIterator(const PolyTransformIterator_& other) : holder_(other.holder_->clone()) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor
      PolyTransformIterator(PolyTransformIterator_&& other) : holder_(other.holder_) { other.holder_ = NULL; }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Destructor
      ~PolyTransformIterator() { delete holder_; }

      /// Assignment operator
      PolyTransformIterator_& operator =(const PolyTransformIterator_& other) {
        if(this != &other) {
          delete holder_;
          holder_ = other.holder_->clone();
        }

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move assignment operator
      PolyTransformIterator_& operator =(PolyTransformIterator_&& other) {
        if(this != &other) { // make sure we are not assigning this pointer to itself.
          delete holder_;
          holder_ = other.holder_->clone();
        }

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    private:
      // Give
      friend class boost::iterator_core_access;

      reference dereference() const {
        return holder_->dereference();
      }
      bool equal(const PolyTransformIterator_& other) const {
        if(holder_->type() == other.holder_->type())
          return holder_->equal(other.holder_);

        return holder_->void_ptr() == other.holder_->void_ptr();
      }
      void increment() {
        holder_->increment();
      }

      HolderBase* holder_;

      class HolderBase
      {
      public:
        virtual ~HolderBase() { }

        virtual const std::type_info& type() const = 0;
        virtual HolderBase* clone() const = 0;
        virtual reference dereference() const = 0;
        virtual bool equal(const HolderBase* other) const = 0;
        virtual const void* void_ptr() const = 0;
        virtual void increment() = 0;
      }; // class holder_base

      template<typename Iterator, typename Functor>
      class Holder : public HolderBase
      {
        typedef Iterator iterator_type;
        typedef Functor functor_type;
        typedef Holder<Iterator, Functor> Holder_;
      public:

        Holder(const iterator_type& iter, const functor_type& f) : it_(iter), f_(f) { }
        virtual ~Holder() { }

        virtual const std::type_info& type() const { return typeid(it_); }
        virtual HolderBase* clone() const { return new Holder_(it_, f_); }
        virtual reference dereference() const { return f_(*it_); }
        virtual void increment() { ++it_; }
        virtual bool equal(const HolderBase* other) const {
          const Holder_* h = dynamic_cast<const Holder_*>(other);
          return it_ == h->it_;
        }
        virtual const void* void_ptr() const { return static_cast<const void*>(&(*it_)); }

        iterator_type it_;
        functor_type f_;
      }; // class holder

    }; // class PolyTransformIterator

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED
