#ifndef TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
#define TILEDARRAY_ANNOTATED_TILE_H__INCLUDED

#include <type_traits.h>
#include <boost/type_traits.hpp>
#include <vector>
#include <numeric>
#include <variable_list.h>
#include <cstddef>

namespace TiledArray {
  // forward declaration
  template<typename T, unsigned int DIM, typename CS>
  class Tile;

  namespace detail {

    /// Annotated tile.
    template<typename T>
    class AnnotatedTile {
    private:
      typedef Eigen::aligned_allocator<T> alloc_type;
    public:
      typedef AnnotatedTile<T> AnnotatedTile_;
      typedef typename boost::remove_const<T>::type value_type;
      typedef typename mirror_const<T, value_type>::reference reference_type;
      typedef const value_type & const_reference_type;
      typedef typename mirror_const<T, value_type>::pointer ptr_type;
      typedef const value_type * const_ptr_type;
      typedef std::vector<std::size_t> size_array;
      typedef ptr_type iterator;
      typedef const_ptr_type const_iterator;

      template<unsigned int DIM, typename CS>
      AnnotatedTile(const Tile<value_type,DIM,CS>& t, const detail::VariableList& var) :
          data_(const_cast<value_type*>(t.begin())), size_(t.size().begin(), t.size().end()),
          n_(t.volume()), var_(var), dim_(Tile<value_type,DIM,CS>::dim()),
          order_(CS::ordering()), owner_(false), alloc_()
      {
        TA_ASSERT( dim_ == var_.count() ,
            std::runtime_error("AnnotatedTile<...>::AnnotatedTile(...): The number of variables in the variable list does not match the tile dimensions."));
      }

      AnnotatedTile(const size_array& size, const VariableList& var, value_type val = value_type(),
          DimensionOrderType order = decreasing_dimension_order) :
          data_(NULL), size_(size),
          n_(std::accumulate(size.begin(), size.end(), std::size_t(1), std::multiplies<std::size_t>())),
          var_(var), dim_(size.size()), order_(order), owner_(false), alloc_()
      {
        create_(val);
      }

      template<typename InIter>
      AnnotatedTile(const size_array& size, const VariableList& var, InIter first, InIter last,
          DimensionOrderType order = decreasing_dimension_order) :
          data_(NULL), size_(size),
          n_(std::accumulate(size.begin(), size.end(), std::size_t(1), std::multiplies<std::size_t>())),
          var_(var), dim_(size.size()), order_(order), owner_(false), alloc_()
      {
        create_(first, last);
      }

      AnnotatedTile(const AnnotatedTile_& other) :
          data_(NULL), size_(other.size_), n_(other.n_), var_(other.var_),
          dim_(other.dim_), order_(other.order_), owner_(false), alloc_(other.alloc_)
      {
        if(other.owner_)
          create_(other.begin(), other.end());
        else
          data_ = other.data_;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      AnnotatedTile(AnnotatedTile_&& other) :
          data_(other.data_), size_(std::move(other.size_)), n_(other.n_),
          var_(std::move(other.var_)), dim_(other.dim_), order_(other.order_),
          owner_(true), alloc_(std::move(other.alloc_))
      {
        if(other.owner_) {
          other.data_ = NULL;
          other.owner_ = false;
        }
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      ~AnnotatedTile() { destroy_(); }

      AnnotatedTile_& operator =(const AnnotatedTile_& other) {
        destroy_();
        size_ = other.size_;
        n_ = other.n_;
        var_ = other.var_;
        dim_ = other.dim_;
        order_ = other.order_;
        alloc_ = other.alloc_;

        if(other.owner_)
          create_(other.begin(), other.end());
        else
          data_ = other.data_;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      AnnotatedTile& operator =(AnnotatedTile&& other) {
        if(this != &other) {
          destroy_();
          data_ = other.data_;
          if(other.owner_) {
            other.data_ = NULL;
            other.owner_ = false;
          }
          size_ = std::move(other.size_);
          n_ = other.n_;
          var_ = std::move(other.var_);
          dim_ = other.dim_;
          order_ = other.order_;
          alloc_ = std::move(other.alloc_);
        }

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      iterator begin() { return data_; }
      const_iterator begin() const { return data_ + n_; }
      iterator end() { return data_; }
      const_iterator end() const { return data_ + n_; }
      ptr_type data() { return data_; }
      const_ptr_type data() const { return data_; }
      const size_array& size() const { return size_; }
      std::size_t volume() const { return n_; }
      const VariableList& vars() const { return var_; }
      unsigned int dim() const { return dim_; }
      DimensionOrderType order() const { return order_; }
      alloc_type get_allocator() const { return alloc_; }

    private:
      /// Allocate and initialize the array w/ a constant value.

      /// All elements will contain the given value.
      void create_(const value_type val) {
        owner_ = true;
        dim_ = alloc_.allocate(n_);
        for(std::size_t i = 0; i < n_; ++i)
          alloc_.construct(dim_ + i, val);
      }

      /// Allocate and initialize the array.

      /// All elements will be initialized to the values given by the iterators.
      /// If the iterator range does not contain enough elements to fill the array,
      /// the remaining elements will be initialized with the default constructor.
      template <typename InIter>
      void create_(InIter first, InIter last) {
        owner_ = true;
        data_ = alloc_.allocate(n_);
        for(std::size_t i = 0; i < n_; ++i) {
          if(first != last) {
            alloc_.construct(data_ + i, *first);
            ++first;
          } else {
            alloc_.construct(data_ + i, value_type());
          }
        }
      }

      /// Destroy the array
      void destroy_() {
        if(!owner_)
          return;
        value_type* p = data_;
        const value_type* const end = data_ + n_;
        for(; p != end; ++p)
          alloc_.destroy(p);

        alloc_.deallocate(data_, n_);
        dim_ = NULL;
        owner_ = false;
      }

      ptr_type data_;           ///< tile data
      size_array size_;         ///< tile size
      std::size_t n_;           ///< tile volume
      VariableList var_;        ///< variable list
      unsigned int dim_;        ///< tile dimensions
      DimensionOrderType order_;///< dimension ordering
      bool owner_;              ///< true when tile data is owned by this object.
      alloc_type alloc_;        ///< allocator

      template<typename U, unsigned int DIM, typename CS>
      friend class ::TiledArray::Tile;
    }; // class AnnotatedTile

  } // namespace detail
} // namespace TiledArray
#endif // TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
