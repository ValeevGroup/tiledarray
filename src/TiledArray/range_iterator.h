#ifndef TILEDARRAY_RANGE_ITERATOR_H__INCLUDED
#define TILEDARRAY_RANGE_ITERATOR_H__INCLUDED

#include <boost/iterator/iterator_facade.hpp>

namespace TiledArray {
  namespace detail {

    /// Coordinate index iterate

    /// This is an input iterator that is used to iterate over the coordinate
    /// indexes of a \c Range.
    /// \tparam Value The value type of the iterator
    /// \tparam Container The container that the iterator references
    /// \note The container object must define the function
    /// \c Container::increment(Value&) \c const, and be accessible to
    /// \c RangeIterator.
    template <typename Value, typename Container>
    class RangeIterator : public boost::iterator_facade<
        RangeIterator<Value,Container>, Value, std::input_iterator_tag, const Value& >
    {
    private:
      typedef boost::iterator_facade<RangeIterator<Value,Container>, Value,
          std::input_iterator_tag, const Value& > iterator_facade_; ///< Base class

    public:
      typedef RangeIterator<Value,Container> RangeIterator_; ///< This class type

      // Standard iterator typedefs
      typedef typename iterator_facade_::value_type value_type; ///< Iterator value type
      typedef typename iterator_facade_::reference reference; ///< Iterator reference type
      typedef typename iterator_facade_::pointer pointer; ///< Iterator pointer type
      typedef typename iterator_facade_::iterator_category iterator_category; /// Iterator category tag
      typedef typename iterator_facade_::difference_type difference_type; ///< Iterator difference type

      /// Copy constructor

      /// \param other The other iterator to be copied
      RangeIterator(const RangeIterator_& other) :
        container_(other.container_), current_(other.current_) {
      }

      /// Construct an index iterator

      /// \param v The initial value of the iterator index
      /// \param c The container that the iterator will reference
      RangeIterator(const Value& v, const Container* c) :
          container_(c), current_(v)
      { }

      /// Copy constructor

      /// \param other The other iterator to be copied
      /// \return A reference to this object
      RangeIterator_& operator=(const RangeIterator_& other) {
        current_ = other.current_;
        container_ = other.container_;

        return *this;
      }

    private:

      /// Compare this iterator with \c other for equality

      /// \param other The other iterator to be checked for equality
      /// \return \c true when the value of the two iterators are the same and
      /// they point to the same container, otherwise \c false
      bool equal(const RangeIterator_ & other) const {
        return current_ == other.current_ && container_ == other.container_;
      }

      /// Increment the current value

      /// This calls \c Container::increment(Value&) and passes the current
      /// value as the function argument.
      void increment() {
        container_->increment(current_);
      }

      /// Dereference the iterator

      /// \return A const reference to the current value
      reference dereference() const {
        return current_;
      }

      RangeIterator();

      // boost::iterator_core_access requires access to private members for
      // boost::iterator_facade to function correctly.
      friend class boost::iterator_core_access;

      const Container* container_;  ///< The container that the iterator references
      Value current_;               ///< The current value of the iterator
    }; // class RangeIterator

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_RANGE_ITERATOR_H__INCLUDED
