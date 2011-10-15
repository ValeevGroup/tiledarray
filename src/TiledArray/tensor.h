#ifndef TILEDARRAY_TENSOR_H__INCLUDED
#define TILEDARRAY_TENSOR_H__INCLUDED

#include <TiledArray/tensor_base.h>
#include <TiledArray/dense_storage.h>
#include <TiledArray/range.h>

namespace TiledArray {
  namespace expressions {

    template <typename, typename, typename>
    class Tensor;

    template <typename T, typename R, typename A>
    struct TensorTraits<Tensor<T, R, A> > {
      typedef DenseStorage<T,A> storage_type;
      typedef R range_type;
      typedef typename storage_type::value_type value_type;
      typedef typename storage_type::const_reference const_reference;
      typedef typename storage_type::reference reference;
      typedef typename storage_type::const_iterator const_iterator;
      typedef typename storage_type::iterator iterator;
      typedef typename storage_type::difference_type difference_type;
      typedef typename storage_type::const_pointer const_pointer;
      typedef typename storage_type::pointer pointer;
    };  // struct TensorTraits<Tensor<T, A> >

    template <typename T, typename R, typename A>
    struct Eval<Tensor<T, R, A> > {
      typedef const Tensor<T, R, A>& type;
    }; // struct Eval<Tensor<T, R, A> >

    /// Evaluation tensor

    /// This tensor is used as an evaluated intermediate for other tensors.
    /// \tparma T the value type of this tensor
    /// \tparam R The range type of this tensor (default = DynamicRange)
    /// \tparam A The allocator type for the data
    template <typename T, typename R = DynamicRange, typename A = Eigen::aligned_allocator<T> >
    class Tensor : public DirectWritableTensor<Tensor<T, R, A> > {
    public:
      typedef Tensor<T, R, A> Tensor_;
      TILEDARRAY_DIRECT_WRITABLE_TENSOR_INHERIT_TYPEDEF(DirectWritableTensor<Tensor_> , Tensor_ );
      typedef DenseStorage<T,A> storage_type;

      /// Default constructor

      /// Construct an empty tensor that has no data or dimensions
      Tensor() : range_(), data_() { }

      /// Construct an evaluated tensor

      /// This will take ownership of the memory held by \c data
      /// \tparam D The range derived type
      /// \param r An array with the size of of each dimension
      /// \param d The data for the tensor
      template <typename D>
      Tensor(const Range<D>& r) :
        range_(r), data_(r.volume())
      { }

      /// Construct an evaluated tensor
      template <typename D, typename InIter>
      Tensor(const Range<D>& r, InIter it) :
        range_(r), data_(r.volume(), it)
      { }

      /// Copy constructor

      /// \param other The tile to be copied.
      template <typename Derived>
      Tensor(const ReadableTensor<Derived>& other) :
          range_(other.range()), data_(other.size())
      {
        other.eval_to(data_);
      }

      /// Copy constructor

      /// \param other The tile to be copied.
      Tensor(const Tensor_& other) :
        range_(other.range_), data_(other.data_)
      { }

      Tensor_& operator=(const Tensor_& other) {
        range_ = other.range();
        storage_type temp(other.range().volume());
        other.eval_to(temp);
        temp.swap(data_);

        return *this;
      }

      template <typename D>
      Tensor_& operator=(const ReadableTensor<D>& other) {
        range_ = other.range();
        storage_type temp(other.range().volume());
        other.eval_to(temp);
        temp.swap(data_);

        return *this;
      }

      template <typename D>
      Tensor_& operator+=(const ReadableTensor<D>& other) {
        if(data_.empty()) {
          range_ = other.range();
          storage_type temp(other.range().volume());
          temp.swap(data_);
        }

        TA_ASSERT(range_ == other.range());
        other.add_to(data_);

        return *this;
      }

      template <typename D>
      Tensor_& operator-=(const ReadableTensor<D>& other) {
        if(data_.empty()) {
          range_ = other.range();
          storage_type temp(other.range().volume());
          temp.swap(data_);
        }

        TA_ASSERT(range_ == other.range());
        other.sub_to(data_);

        return *this;
      }

      /// Evaluate this tensor

      /// \return A const reference to this object.
      const eval_type& eval() const { return *this; }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return range_; }

      /// Tensor dimension size accessor

      /// \return The number of elements in the tensor
      size_type size() const { return range_.volume(); }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const { return data_[i]; }

      /// Element accessor

      /// \return The element at the \c i position.
      reference operator[](size_type i) { return data_[i]; }


      /// Element accessor

      /// \return The element at the \c i position.
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, const_reference>::type
      operator[](const Index& i) const { return data_[range_.ord(i)]; }

      /// Element accessor

      /// \return The element at the \c i position.
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, reference>::type
      operator[](const Index& i) { return data_[range_.ord(i)]; }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const { return data_.begin(); }

      /// Iterator factory

      /// \return An iterator to the first data element
      iterator begin() { return data_.begin(); }

      /// Iterator factory

      /// \return An iterator to the last data element
      const_iterator end() const { return data_.end(); }

      /// Iterator factory

      /// \return An iterator to the last data element
      iterator end() { return data_.end(); }

      /// Data direct access

      /// \return A const pointer to the tensor data
      const_pointer data() const { return data_.data(); }

      /// Data direct access

      /// \return A const pointer to the tensor data
      pointer data() { return data_.data(); }

      template <typename Archive>
      void serialize(Archive& ar) {
        ar & range_ & data_;
      }

      void swap(Tensor_& other) {
        range_.swap(other.range_);
        data_.swap(other.data_);
      }

    private:

      range_type range_; ///< Tensor size info
      storage_type data_; ///< Tensor data
    };

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_H__INCLUDED
