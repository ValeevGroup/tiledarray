#ifndef TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED
#define TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED

#include <boost/iterator/iterator_facade.hpp>

namespace TiledArray {

  namespace detail {

    /// Wrapper class iterators to provide the index interface.

    /// This class wraps an iterator object which dereferences into a pair, where
    /// the first is the index and second is the data.
    /// \arg \c It is the iterator type.
    template<typename It>
    class IndexedIterator : public boost::iterator_facade<IndexedIterator<It>,
        typename It::value_type::second_type, typename It::iterator_category,
        typename It::reference, typename It::difference_type>
    {
    private:
      typedef boost::iterator_facade<IndexedIterator<It>,
          typename It::value_type::second_type,
          typename It::iterator_category> iterator_facade_; ///< Base class type


    public:
      typedef IndexedIterator<It> IndexedIterator_; ///< this object type
      typedef typename It::value_type::first_type index_type;
      typedef It iterator_type;
      typedef typename iterator_facade_::value_type value_type;
      typedef typename iterator_facade_::reference reference;
      typedef typename iterator_facade_::pointer pointer;
      typedef typename iterator_facade_::difference_type difference_type;
      typedef typename iterator_facade_::iterator_category iterator_category;

      /// Default constructor
      IndexedIterator() : it_() { }
      /// Construct with a base iterator
      IndexedIterator(iterator_type it) : it_(it) { }
      /// Copy constructor
      IndexedIterator(const IndexedIterator_& other) : it_(other.it_) { }
      /// Copy a convertible iterator
      template<typename OtherIt>
      IndexedIterator(const IndexedIterator<OtherIt>& other) : it_(other.it_) { }
  #ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move copy constructor
      IndexedIterator(IndexedIterator_&& other) : it_(std::move(other.it_)) { }
  #endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Assignment operator
      IndexedIterator_& operator=(const IndexedIterator_& other) {
        it_ = other.it_;
        return *this;
      }

  #ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move assignment operator
      IndexedIterator_& operator=(IndexedIterator_&& other) {
        it_ = std::move(other.it_);
        return *this;
      }
  #endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Returns the current index.
      const index_type& index() const { return it_->first; }

    private:
      /// Returns a reference to the iterator data.
      reference dereference() { return it_->second; }

      /// Returns true if the base iterators are equal.
      bool equal(const IndexedIterator_& other) const { return it_ == other.it_; }

      /// Increments the base iterator.
      void increment() { ++it_; }

      /// Decrements the base iterator.
      void decrement() { --it_; }

      /// Advances the base iterator n positions.
      void advance(difference_type n) { std::advance(it_, n); }

      /// Returns the difference between this base iterator (lh) and the other base iterator (rh).
      difference_type distance_to(const IndexedIterator_& other) const {
        return std::distance(it_, other.it_);
      }

      friend class boost::iterator_core_access;

      iterator_type it_; ///< Base iterator
    }; // class IndexedIterator


    /// Wrapper class iterators to provide the index interface.

    /// This is a specialization that uses a pair to store an index and iterator
    /// object. This is appropriate for iterators that do NOT dereference into a
    /// pair with key type and data type. The Index type should have the
    /// following operators defined (when the equivalent iterator operations are
    /// valid): ++, --, -, and +=.
    /// \arg \c Ix is the index type.
    /// \arg \c It is the iterator type.
    template<typename Ix, typename It>
    class IndexedIterator<std::pair<Ix, It> > :
        public boost::iterator_facade<IndexedIterator<std::pair<Ix, It> >,
            typename std::iterator_traits<It>::value_type,
            typename std::iterator_traits<It>::iterator_category,
            typename std::iterator_traits<It>::reference,
            typename std::iterator_traits<It>::difference_type>
    {
    private:
      typedef boost::iterator_facade<IndexedIterator<std::pair<Ix, It> >,
          typename std::iterator_traits<It>::value_type,
          typename std::iterator_traits<It>::iterator_category,
          typename std::iterator_traits<It>::reference,
          typename std::iterator_traits<It>::difference_type> iterator_facade_;

    public:
      typedef IndexedIterator<It> IndexedIterator_;
      typedef Ix index_type;
      typedef It iterator_type;
      typedef typename iterator_facade_::value_type value_type;
      typedef typename iterator_facade_::reference reference;
      typedef typename iterator_facade_::pointer pointer;
      typedef typename iterator_facade_::difference_type difference_type;
      typedef typename iterator_facade_::iterator_category iterator_category;

      /// Default constructor
      IndexedIterator() : it_() { }

      /// Construct with an index-iterator pair.
      IndexedIterator(std::pair<index_type, iterator_type> it) : it_(it) { }

      /// Copy constructor
      IndexedIterator(const IndexedIterator_& other) : it_(other.it_) { }

      /// Construct with a convertible iterator.
      template<typename OtherIt>
      IndexedIterator(const IndexedIterator<OtherIt>& other) : it_(other.it_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move Constructor
      IndexedIterator(IndexedIterator_&& other) : it_(std::move(other.it_.first), std::move(other.it_.second)) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Assignment operator
      IndexedIterator_& operator=(const IndexedIterator_& other) {
        it_ = other.it_;
        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move assignment operator
      IndexedIterator_& operator=(IndexedIterator_&& other) {
        it_.first = std::move(other.it_.first);
        it_.second = std::move(other.it_.second);

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Returns the current index
      const index_type& index() const { return it_.first; }

    private:

      /// Dereferences to base iterator
      reference dereference() const { return * (it_.second); }

      /// Compares base iterators for equality.
      bool equal(const IndexedIterator_& other) const {
        return (it_.second == other.it_.second);
      }

      /// Increments the index and base iterator
      void increment() {
        ++(it_.first);
        ++(it_.second);
      }

      /// Decrements the index and base iterator
      void decrement() {
        --(it_.first);
        --(it_.second);
      }

      /// Advances the index and base iterator n positions.
      void advance(difference_type n) {
        it_.first += n;
        std::advance(it_.second, n);
      }

      /// Returns the distance between this and the other base iterators.
      difference_type distance_to(const IndexedIterator_& other) const {
        return std::distance(it_.second, other.it_.second);
      }

      friend class boost::iterator_core_access;

      std::pair<Ix, It> it_; ///< pair with the index and base iterator.
    }; // class IndexedIterator

  } // namespace detail

}  // namespace TiledArray


#endif // TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED
