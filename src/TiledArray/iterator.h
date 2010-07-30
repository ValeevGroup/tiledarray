#ifndef TILEDARRAY_ITERATOR_H__INCLUDED
#define TILEDARRAY_ITERATOR_H__INCLUDED

//#include <TiledArray/coordinate_system.h>
#include <boost/iterator/iterator_facade.hpp>

namespace TiledArray {

  namespace detail {

    /// Input iterator used to iterate over multidimensional indices
    template <typename Value, typename Container>
    class IndexIterator : public boost::iterator_facade<
        IndexIterator<Value,Container>, Value, std::input_iterator_tag >
    {
    public:
      typedef IndexIterator<Value,Container> IndexIterator_;
      typedef boost::iterator_facade<IndexIterator_, Value, std::input_iterator_tag > iterator_facade_;

      IndexIterator(const IndexIterator_& other) :
        container_(other.container_), current_(other.current_) {
      }

      ~IndexIterator() {}

      IndexIterator(const Value& cur, const Container* container) :
        container_(container), current_(cur) {
      }

    protected:
      friend class boost::iterator_core_access;

      bool equal(const IndexIterator_ & other) const {
        return current_ == other.current_ && container_ == other.container_;
      }

      // user must provide void Container::increment(index& current) const;
      void increment() {
        container_->increment(current_);
      }

      Value& dereference() const {
        return const_cast<Value&>(current_);
      }

    private:
      IndexIterator();

      const Container* container_;
      Value current_;
    }; // class IndexIterator

    /// Element Iterator used to iterate over elements of dense arrays.
    template <typename Value, typename IndexIt, typename Container, typename Reference = Value& >
    class ElementIterator : public boost::iterator_facade<
        ElementIterator<Value, IndexIt, Container>, Value, std::output_iterator_tag, Reference>
    {
    public:
      typedef ElementIterator<Value, IndexIt, Container> ElementIterator_;
      typedef typename IndexIt::value_type index_type;
      typedef typename boost::remove_const<Container>::type container_type;

      /// Primary constructor
      ElementIterator(const IndexIt& it, container_type * const container) :
        container_(container), current_(it)
      {}

      /// Copy constructor
      ElementIterator(const ElementIterator_& other) :
        container_(other.container_), current_(other.current_)
      {}

      /// Copy constructor for iterators of other types (i.e. const_iterator to iterator).
      template<typename OtherValue>
      ElementIterator(const ElementIterator<OtherValue, IndexIt, Container>& other) :
        container_(other.container_), current_(other.current_)
      {}

      ~ElementIterator() {}

      const index_type & index() const {
        return *current_;
      }

    private:
      friend class boost::iterator_core_access;

      ElementIterator();

      bool equal(const ElementIterator_ & other) const {
        return current_ == other.current_ && container_ == other.container_;
      }

      void increment() {
        ++current_;
      }

      Reference dereference() const {
        return container_->operator[](*current_);
      }

      template<typename OtherValue>
      bool equal(const ElementIterator<OtherValue, IndexIt, Container>& other) {
        return current_ == other.current_ && container_ == other.container_;
      }

      container_type * container_;
      IndexIt current_;

    }; // class ElementIterator

  } // namespace detail

} // namespace TiledArray

#endif // TILEDARRAY_ITERATOR_H__INCLUDED
