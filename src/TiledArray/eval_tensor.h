#ifndef TILEDARRAY_EVAL_TENSOR_H__INCLUDED
#define TILEDARRAY_EVAL_TENSOR_H__INCLUDED

#include <TiledArray/tensor_base.h>
#include <TiledArray/dense_storage.h>
#include <vector>
#include <algorithm>

namespace TiledArray {
  namespace expressions {

    /// Basic tensor size information
    class TensorSize {
    public:
      typedef std::size_t size_type;  ///< The size type
      typedef std::vector<size_type> size_array; ///< Tensor sizes array

      TensorSize() :
        size_(), order_(TiledArray::detail::decreasing_dimension_order)
      { }

      /// Construct
      template <typename SizeArray>
      TensorSize(const SizeArray& s, TiledArray::detail::DimensionOrderType o) :
        size_(s.begin(), s.end()), order_(o)
      { }

      TensorSize(const TensorSize& other) :
        size_(other.size_), order_(other.order_)
      { }

      template <typename T>
      TensorSize(const T& other) :
        size_(other.size().begin(), other.size().end()), order_(other.order())
      { }

      TensorSize& operator=(const TensorSize& other) {
        size_ = other.size_;
        order_ = other.order_;

        return *this;
      }

      template <typename T>
      TensorSize& operator=(const T& other) {
        size_.resize(other.dim());
        std::copy(other.size().begin(), other.size().end(), size_.begin());
        order_ = other.order();

        return *this;
      }

      /// Tensor dimension accessor

      /// \return The number of dimensions
      unsigned int dim() const { return size_.size(); }

      /// Data ordering

      /// \return The data ordering type
      TiledArray::detail::DimensionOrderType order() const { return order_; }

      /// Tensor dimension size accessor

      /// \return An array that contains the sizes of each tensor dimension
      const size_array& size() const { return size_; }

      /// Tensor volume

      /// \return The total number of elements in the tensor
      size_type volume() const {
        return std::accumulate(size_.begin(), size_.end(), size_type(1), std::multiplies<size_type>());
      }

      template <typename Archive>
      void serialize(const Archive& ar) {
        ar & size_ & order_;
      }

    protected:
      size_array size_; ///< The sizes of each dimension
      TiledArray::detail::DimensionOrderType order_; ///< Data ordering
    }; // class TensorSize


    template <typename T, typename A = Eigen::aligned_allocator<T> >
    class EvalTensor : public DirectReadableTensor<EvalTensor<T, A> > {
    public:
      typedef DirectReadableTensor<EvalTensor<T, A> > base;
      typedef DenseStorage<T,A> storage_type;

      typedef typename TensorSize::size_type size_type;
      typedef typename TensorSize::size_array size_array;
      typedef typename storage_type::value_type value_type;
      typedef typename storage_type::const_reference const_reference;
      typedef typename storage_type::const_iterator const_iterator;
      typedef typename storage_type::const_pointer const_pointer;

      /// Construct an evaluated tensor

      /// This will take ownership of the memory held by \c data
      /// \tparam SizeArray The input size array type.
      /// \param s An array with the size of of each dimension
      /// \param o The dimension ordering
      /// \param d The data for the tensor
      template <typename SizeArray>
      EvalTensor(const SizeArray& s, TiledArray::detail::DimensionOrderType o, const storage_type& d) :
        size_(s, o), data_(d)
      { }

      /// Construct an evaluated tensor
      template <typename SizeArray>
      EvalTensor(const SizeArray& s, TiledArray::detail::DimensionOrderType o, storage_type& d) :
        size_(s, o), data_()
      {
        data_.swap(d);
      }

    private:
      // not allowed
      EvalTensor<T,A>& operator=(const EvalTensor<T,A>&);
      EvalTensor(const EvalTensor<T,A>&);

    public:

      /// Evaluate this tensor

      /// \return A const reference to this object.
      const EvalTensor<T, A>& eval() const { return *this; }

      template <typename Dest>
      void eval_to(const Dest& dest) const {
        TA_ASSERT(volume() == dest.volume());
        std::copy(begin(), end(), dest.begin());
      }

      /// Tensor dimension accessor

      /// \return The number of dimensions
      unsigned int dim() const { return size_.dim(); }

      /// Data ordering

      /// \return The data ordering type
      TiledArray::detail::DimensionOrderType order() const { return size_.order(); }

      /// Tensor dimension size accessor

      /// \return An array that contains the sizes of each tensor dimension
      const size_array& size() const { return size_.size(); }

      /// Tensor volume

      /// \return The total number of elements in the tensor
      size_type volume() const { return size_.volume(); }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const { return data_[i]; }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const { return data_.begin(); }

      /// Iterator factory

      /// \return An iterator to the last data element
      const_iterator end() const { return data_.end(); }

      /// Data direct access

      /// \return A const pointer to the tensor data
      const_pointer data() const { return data_.data(); }

    private:
      TensorSize size_; ///< Tensor size info
      storage_type data_; ///< Tensor data
    };

    template <typename T, typename A>
    struct TensorTraits<EvalTensor<T, A> > {
      typedef typename EvalTensor<T, A>::size_type size_type;
      typedef typename EvalTensor<T, A>::size_array size_array;
      typedef typename EvalTensor<T, A>::value_type value_type;
      typedef typename EvalTensor<T, A>::const_reference const_reference;
      typedef typename EvalTensor<T, A>::const_iterator const_iterator;
      typedef typename EvalTensor<T, A>::const_pointer const_pointer;
    }; // struct TensorTraits<EvalTensor<T, A> >

    template <typename T, typename A>
    struct Eval<EvalTensor<T, A> > {
      typedef const EvalTensor<T, A>& type;
    }; // struct Eval<EvalTensor<T, A> >

  } // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_EVAL_TENSOR_H__INCLUDED
