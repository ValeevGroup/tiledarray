#ifndef TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED
#define TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED

#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace detail {

    /// Iterator adapter for indexed iterators

    /// This class wraps an indexed iterator so that only the data is is given
    /// when dereferencing the iterator. It also provides a const accessor
    /// member function for the iterators index. Otherwise, the iterator behaves
    /// the same as the base iterator. \c Iterator type must conform to the
    /// standard c++ iterator requirements.
    /// \tparam Iterator The base iterator type.
    /// \note Iterator::value_type must be a \c std::pair<const \c K, \c D>
    /// type, where \c K is the key (or index) type and \c D is the data type.
    /// \note If Iterator::value_type is a \c TiledArray::detail::Key type, then
    /// index type will default to \c Key::key1_type; unless one of the two
    /// \c Key types is a \c TiledArray::ArrayCoordinate type, in which case it
    /// will be the \c TiledArray::ArrayCoordinate type.
    template <typename Iterator>
    class IndexedIterator {
    private:
      // Used to selectively enable functions with madness::enable_if
      struct Enabler { };

    public:
      typedef IndexedIterator<Iterator> IndexedIterator_; ///< this object type
      typedef typename std::iterator_traits<Iterator>::value_type::first_type index_type; ///< The index (or key) type for the iterator
      typedef Iterator base_type; ///< The base iterator type
      typedef typename std::iterator_traits<Iterator>::value_type::second_type value_type; ///< iterator value_type
      typedef typename madness::if_<std::is_const<typename std::remove_reference<typename std::iterator_traits<Iterator>::reference>::type >,
          const value_type&, value_type&>::type reference; ///< iterator reference type
      typedef typename madness::if_<std::is_const<typename std::remove_pointer<typename std::iterator_traits<Iterator>::pointer>::type >,
          const value_type*, value_type*>::type pointer; ///< iterator pointer type
      typedef typename std::iterator_traits<Iterator>::difference_type difference_type; ///< iterator difference type
      typedef typename std::iterator_traits<Iterator>::iterator_category iterator_category; ///< iterator traversal category

      /// Default constructor

      /// Constructs an indexed iterator that uses the default constructor for
      /// the base iterator.
      /// \note This constructor is only enabled for forward, bidirectional, and
      /// random access type pointers.
      IndexedIterator(typename madness::enable_if<is_forward_iterator<Iterator>, Enabler >::type = Enabler()) :
          it_()
      { }

      /// Construct with an iterator

      /// \tparam It Any iterator type that implicitly convertible to the
      /// base iterator type
      /// \param it An iterator that will initialize the internal iterator.
      /// \note \c it must be implicitly convertible to the base iterator type.
      template <typename It>
      explicit IndexedIterator(const It& it,
          typename madness::enable_if<std::is_convertible<It, Iterator>, Enabler>::type = Enabler()) :
          it_(it)
      { }

      /// Copy constructor

      /// \param other The other indexed iterator to be copied.
      IndexedIterator(const IndexedIterator_& other) : it_(other.it_) { }

      /// Copy a convertible iterator

      /// \tparam It Another iterator type that is convertible to the
      /// internal iterator type.
      /// \param other The indexed iterator that is to be copied.
      template<typename It>
      IndexedIterator(const IndexedIterator<It>& other,
          typename madness::enable_if<std::is_convertible<It, Iterator>, Enabler>::type = Enabler()) :
          it_(other.base())
      { }

      /// Assignment operator

      /// \param other The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      IndexedIterator_& operator=(const IndexedIterator_& other) {
        it_ = other.base();
        return *this;
      }

      /// Assignment operator for other indexed iterator types

      /// \tparam It Another iterator type that is convertible to the base
      /// iterator type.
      /// \param other The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      template <typename It>
      typename madness::enable_if<std::is_convertible<It, Iterator>, IndexedIterator_&>::type
      operator=(const IndexedIterator<It>& other) {
        it_ = other.base();
        return *this;
      }

      /// Assignment operator for other iterator types

      /// \tparam It Another iterator type that is convertible to the base
      /// iterator type.
      /// \param it The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      template <typename It>
      typename madness::enable_if<std::is_convertible<It, Iterator>, IndexedIterator_&>::type
      operator=(const It& it) {
        it_ = it;
        return *this;
      }

      /// Assignment operator for other iterator types

      /// \tparam It Another iterator type that is convertible to the base
      /// iterator type.
      /// \param it The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      IndexedIterator_& operator+=(const difference_type n) {
        it_ += n;
        return *this;
      }

      /// Assignment operator for other iterator types

      /// \tparam It Another iterator type that is convertible to the base
      /// iterator type.
      /// \param it The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      IndexedIterator_& operator-=(const difference_type n) {
        it_ -= n;
        return *this;
      }

      /// Base iterator accessor

      /// \return A reference to the base iterator
      Iterator& base() { return it_; }

      /// Base iterator const accessor

      /// \return A const reference to the base iterator
      const Iterator& base() const { return it_; }

      /// Index accessor

      /// \return The index associated with the iterator.
      const index_type& index() const {
        return it_->first;
      }

      /// Increment operator

      /// Increment the iterator
      /// \return The modified iterator
      IndexedIterator_& operator++() {
        ++it_;
        return *this;
      }

      /// Increment operator

      /// Increment the iterator
      /// \return An unmodified copy of the iterator
      IndexedIterator_ operator++(int) {
        IndexedIterator_ temp(it_++);
        return temp;
      }

      /// Decrement operator

      /// Decrement the iterator
      /// \return The modified iterator
      IndexedIterator_& operator--() {
        --it_;
        return *this;
      }

      /// Decrement operator

      /// Decrement the iterator
      /// \return An unmodified copy of the iterator
      IndexedIterator_ operator--(int) {
        IndexedIterator_ temp(it_--);
        return temp;
      }

      /// Dereference operator

      /// \return A \c reference to the current data
      reference operator*() const { return it_->second; }

      /// Pointer operator

      /// \return A \c pointer to the current data
      pointer operator->() const { return &(it_->second); }

      /// Offset dereference operator
      reference operator[] (difference_type n) const
      { return it_[n].second; }

    private:

      base_type it_;
    }; // class IndexedIterator

    /// IndexedIterator equality operator

    /// Compares the iterators for equality.
    /// \tparam LeftIt The left-hand base iterator type
    /// \tparam RightIt The right-hand base iterator type
    /// \param left_it The left-hand iterator to be compared
    /// \param right_it The right-hand iterator to be compared
    /// \return \c true if the the base iterators of \c left_it and \c right_it
    /// are equal, otherwise \c false .
    template <typename LeftIt, typename RightIt>
    bool operator==(const IndexedIterator<LeftIt>& left_it, const IndexedIterator<RightIt>& right_it) {
      return left_it.base() == right_it.base();
    }

    /// IndexedIterator inequality operator

    /// Compares the iterators for inequality.
    /// \tparam LeftIt The left-hand base iterator type
    /// \tparam RightIt The right-hand base iterator type
    /// \param left_it The left-hand iterator to be compared
    /// \param right_it The right-hand iterator to be compared
    /// \return \c true if the the base iterators of \c left_it and \c right_it
    /// are not equal, otherwise \c false .
    template <typename LeftIt, typename RightIt>
    bool operator!=(const IndexedIterator<LeftIt>& left_it, const IndexedIterator<RightIt>& right_it) {
      return left_it.base() != right_it.base();
    }

    /// IndexedIterator less-than operator

    /// Check that the left-hand iterator is less than the right-hand iterator.
    /// \tparam LeftIt The left-hand base iterator type
    /// \tparam RightIt The right-hand base iterator type
    /// \param left_it The left-hand iterator to be compared
    /// \param right_it The right-hand iterator to be compared
    /// \return \c true if the the base iterators of \c left_it is less than
    /// \c right_it , otherwise \c false .
    template <typename LeftIt, typename RightIt>
    bool operator<(const IndexedIterator<LeftIt>& left_it, const IndexedIterator<RightIt>& right_it) {
      return left_it.base() < right_it.base();
    }

    /// IndexedIterator greater-than operator

    /// Check that the left-hand iterator is greater than the right-hand iterator.
    /// \tparam LeftIt The left-hand base iterator type
    /// \tparam RightIt The right-hand base iterator type
    /// \param left_it The left-hand iterator to be compared
    /// \param right_it The right-hand iterator to be compared
    /// \return \c true if the the base iterators of \c left_it is greater than
    /// \c right_it , otherwise \c false .
    template <typename LeftIt, typename RightIt>
    bool operator>(const IndexedIterator<LeftIt>& left_it, const IndexedIterator<RightIt>& right_it) {
      return left_it.base() > right_it.base();
    }

    /// IndexedIterator less-than-or-equal to operator

    /// Check that the left-hand iterator is less than or equal to the
    /// right-hand iterator.
    /// \tparam LeftIt The left-hand base iterator type
    /// \tparam RightIt The right-hand base iterator type
    /// \param left_it The left-hand iterator to be compared
    /// \param right_it The right-hand iterator to be compared
    /// \return \c true if the the base iterators of \c left_it is less than or
    /// equal to \c right_it , otherwise \c false .
    template <typename LeftIt, typename RightIt>
    bool operator<=(const IndexedIterator<LeftIt>& left_it, const IndexedIterator<RightIt>& right_it) {
      return left_it.base() <= right_it.base();
    }

    /// IndexedIterator greater-than-or-equal to operator

    /// Check that the left-hand iterator is greater than or equal to the
    /// right-hand iterator.
    /// \tparam LeftIt The left-hand base iterator type
    /// \tparam RightIt The right-hand base iterator type
    /// \param left_it The left-hand iterator to be compared
    /// \param right_it The right-hand iterator to be compared
    /// \return \c true if the the base iterators of \c left_it is greater than
    /// or equal to \c right_it , otherwise \c false .
    template <typename LeftIt, typename RightIt>
    bool operator>=(const IndexedIterator<LeftIt>& left_it, const IndexedIterator<RightIt>& right_it) {
      return left_it.base() >= right_it.base();
    }

    /// IndexedIterator addition operator

    /// \tparam It The base iterator type
    /// \param n The distance to advance it
    /// \param it The iterator to be advanced
    /// \return A copy of \c it that has been advanced by \c n.
    template <typename It>
    IndexedIterator<It> operator+(const typename IndexedIterator<It>::difference_type n, const IndexedIterator<It>& it) {
      return IndexedIterator<It>(n + it.base());
    }

    /// IndexedIterator addition operator

    /// \tparam It The base iterator type
    /// \param it The iterator to be advanced
    /// \param n The distance to advance it
    /// \return A copy of \c it that has been advanced by \c n.
    template <typename It>
    IndexedIterator<It> operator+(const IndexedIterator<It>& it, const typename IndexedIterator<It>::difference_type& n) {
      return IndexedIterator<It>(it.base() + n);
    }

    /// IndexedIterator difference operator

    /// \tparam LeftIt The left base iterator type
    /// \tparam RightIt The right base iterator type
    /// \param left_it The left-hand iterator
    /// \param right_it The right-hand iterator
    /// \return The distance between \c left_it and \c right_it .
    template <typename LeftIt, typename RightIt>
    typename IndexedIterator<LeftIt>::difference_type operator-(const IndexedIterator<LeftIt>& left_it, const IndexedIterator<RightIt>& right_it) {
      return left_it.base() - right_it.base();
    }

    /// IndexedIterator inverse advance operator

    /// \tparam It The base iterator type
    /// \param it The iterator to be advanced
    /// \param n The distance to advance it
    /// \return A copy of \c it that has been advanced by \c -n .
    template <typename It>
    IndexedIterator<It> operator-(const IndexedIterator<It>& it, const typename IndexedIterator<It>::difference_type& n) {
      return IndexedIterator<It>(it.base() - n);
    }

  } // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED
