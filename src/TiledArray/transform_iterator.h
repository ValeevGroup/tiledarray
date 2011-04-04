#ifndef TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED
#define TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <typeinfo>
#include <iterator>
#include <boost/static_assert.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/scoped_ptr.hpp>

namespace TiledArray {
  namespace detail {

    /// Polymorphic transform iterator

    /// This iterator will transform an arbitrary iterator with an arbitrary
    /// transformation function. The Value template parameter must be compatible
    /// with the return type of the transformation function.
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
        BOOST_STATIC_ASSERT((std::is_same<typename std::iterator_traits<Iterator>::value_type, value_type>::value));
      }

      /// Copy constructor
      PolyTransformIterator(const PolyTransformIterator_& other) : holder_(other.holder_->clone()) { }

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

    private:
      // Give boost::iterator_facade access to private member functions.
      friend class boost::iterator_core_access;

      /// Return a transformed object.
      reference dereference() const {
        return holder_->dereference();
      }

      /// Compare this iterator with another iterator

      /// If the base iterators used to construct the transformation iterator
      /// have the same type, then the iterators are compared using the ==
      /// operator. Otherwise, they are compared with void pointers to the base
      /// iterator's to the data pointed to by those iterators.
      template<typename OtherValue>
      bool equal(const PolyTransformIterator<OtherValue>& other) const {
        if(holder_->type() == other.holder_->type())
          return holder_->equal(other.holder_);

        return holder_->void_ptr() == other.holder_->void_ptr();
      }

      /// Increment the base pointer.
      void increment() {
        holder_->increment();
      }

      HolderBase* holder_;  ///< Base pointer to the iterator/transform function holder.

      /// Provides the interface to the iterator/transformation function holder.
      class HolderBase
      {
      public:
        virtual ~HolderBase() { }

        /// Returns the type_info object of the base iterator.
        virtual const std::type_info& type() const = 0;
        /// Returns a base pointer to a copy of the actual object.
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
