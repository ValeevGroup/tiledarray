#ifndef TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED
#define TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED

#include <TiledArray/type_traits.h>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/if.hpp>
#include <iterator>

namespace TiledArray {

  template <typename, unsigned int, typename>
  class ArrayCoordinate;

  namespace detail {

    template <typename Iterator>
    struct BaseIteratorTraits {
      typedef typename std::iterator_traits<Iterator>::value_type base_value_type;
      typedef typename std::iterator_traits<Iterator>::reference base_reference;
      typedef typename std::iterator_traits<Iterator>::value_type::second_type value_type;
      typedef typename boost::mpl::if_<std::is_const<typename std::remove_reference<base_reference>::type >,
          const value_type&, value_type&>::type reference;
    };

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
    class IndexedIterator : public boost::iterator_adaptor<
        IndexedIterator<Iterator>,                          // Derived class type
        Iterator,                                           // Base iterator
        typename BaseIteratorTraits<Iterator>::value_type,  // Value type
        boost::use_default,                                 // Iterator category
        typename BaseIteratorTraits<Iterator>::reference>   // reference type

    {
    private:
      /// The base class type
      typedef boost::iterator_adaptor<IndexedIterator<Iterator>, Iterator,
          typename BaseIteratorTraits<Iterator>::value_type, boost::use_default,
          typename BaseIteratorTraits<Iterator>::reference> iterator_adaptor_;

      // Used to selectively enable functions with boost::enable_if
      struct Enabler { };

    public:
      typedef IndexedIterator<Iterator> IndexedIterator_; ///< this object type
      typedef typename std::iterator_traits<Iterator>::value_type::first_type index_type; ///< The index (or key) type for the iterator
      typedef typename iterator_adaptor_::base_type base_type; ///< The base iterator type
      typedef typename iterator_adaptor_::value_type value_type; ///< iterator value_type
      typedef typename iterator_adaptor_::reference reference; ///< iterator reference type
      typedef typename iterator_adaptor_::pointer pointer; ///< iterator pointer type
      typedef typename iterator_adaptor_::difference_type difference_type; ///< iterator difference type
      typedef typename iterator_adaptor_::iterator_category iterator_category; ///< iterator traversal category

      /// Default constructor

      /// Constructs an indexed iterator that uses the default constructor for
      /// the base iterator.
      /// \note This constructor is only enabled for forward, bidirectional, and
      /// random access type pointers.
      IndexedIterator(typename boost::enable_if<is_forward_iterator<Iterator>, Enabler >::type = Enabler()) :
          iterator_adaptor_()
      { }

      /// Construct with an iterator

      /// \tparam It Any iterator type that implicitly convertible to the
      /// base iterator type
      /// \param it An iterator that will initialize the internal iterator.
      /// \note \c it must be implicitly convertible to the base iterator type.
      template <typename It>
      explicit IndexedIterator(const It& it,
          typename boost::enable_if<std::is_convertible<It, Iterator>, Enabler>::type = Enabler()) :
          iterator_adaptor_(it)
      { }

      /// Copy constructor

      /// \param other The other indexed iterator to be copied.
      IndexedIterator(const IndexedIterator_& other) : iterator_adaptor_(other) { }

      /// Copy a convertible iterator

      /// \tparam It Another iterator type that is convertible to the
      /// internal iterator type.
      /// \param other The indexed iterator that is to be copied.
      template<typename It>
      IndexedIterator(const IndexedIterator<It>& other,
          typename boost::enable_if<std::is_convertible<It, Iterator>, Enabler>::type = Enabler()) :
          iterator_adaptor_(other.base())
      { }

      /// Assignment operator

      /// \param other The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      IndexedIterator_& operator=(const IndexedIterator_& other) {
        iterator_adaptor_::base_reference() = other.base_reference();
        return *this;
      }

      /// Assignment operator for other indexed iterator types

      /// \tparam It Another iterator type that is convertible to the base
      /// iterator type.
      /// \param other The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      template <typename It>
      typename boost::enable_if<std::is_convertible<It, Iterator>, IndexedIterator_&>::type
      operator=(const IndexedIterator<It>& other) {
        iterator_adaptor_::base_reference() = other.base();
        return *this;
      }

      /// Assignment operator for other iterator types

      /// \tparam It Another iterator type that is convertible to the base
      /// iterator type.
      /// \param it The indexed iterator that is to be copied.
      /// \return A reference to this iterator
      template <typename It>
      typename boost::enable_if<std::is_convertible<It, Iterator>, IndexedIterator_&>::type
      operator=(const It& it) {
        iterator_adaptor_::base_reference() = it;
        return *this;
      }

      /// Index accessor

      /// \return The index associated with the iterator.
      index_type& index() const {
        return iterator_adaptor_::base_reference()->first;
      }

      reference operator[] (difference_type n) const
      { return static_cast<reference>(iterator_adaptor_::operator[](n)); }

    private:
      /// Returns a reference to the iterator data.
      reference dereference() const { return iterator_adaptor_::base_reference()->second; }

      friend class boost::iterator_core_access;
    }; // class IndexedIterator

  } // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_INDEXED_ITERATOR_H__INCLUDED
