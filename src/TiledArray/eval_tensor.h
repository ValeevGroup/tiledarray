#ifndef TILEDARRAY_EVAL_TENSOR_H__INCLUDED
#define TILEDARRAY_EVAL_TENSOR_H__INCLUDED

#include <TiledArray/tensor_base.h>
#include <TiledArray/dense_storage.h>
#include <TiledArray/range.h>
#include <vector>
#include <algorithm>

namespace TiledArray {
  namespace expressions {

    template <typename, typename>
    class EvalTensor;

    template <typename T, typename A>
    struct TensorTraits<EvalTensor<T, A> > {
      typedef DenseStorage<T,A> storage_type;
      typedef DynamicRange range_type;
      typedef typename storage_type::value_type value_type;
      typedef typename storage_type::reference reference;
      typedef typename storage_type::const_reference const_reference;
      typedef typename storage_type::iterator iterator;
      typedef typename storage_type::const_iterator const_iterator;
      typedef typename storage_type::difference_type difference_type;
      typedef typename storage_type::pointer pointer;
      typedef typename storage_type::const_pointer const_pointer;
    };  // struct TensorTraits<EvalTensor<T, A> >

    template <typename T, typename A>
    struct Eval<EvalTensor<T, A> > {
      typedef const EvalTensor<T, A>& type;
    }; // struct Eval<EvalTensor<T, A> >


    template <typename T, typename A>
    struct TensorArg<EvalTensor<T, A> > {
      typedef const EvalTensor<T, A>& type;
    }; // struct TensorArg<EvalTensor<T, A> >

    template <typename T, typename A>
    struct TensorMem<EvalTensor<T, A> > {
      typedef const EvalTensor<T, A>& type;
    }; // struct TensorMem<EvalTensor<T, A> >

    template <typename T, typename A = Eigen::aligned_allocator<T> >
    class EvalTensor : public DirectReadableTensor<EvalTensor<T, A> > {
    public:
      typedef EvalTensor<T, A> EvalTensor_;
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHEIRATE_TYPEDEF(DirectReadableTensor<EvalTensor_> , EvalTensor_ );
      typedef DenseStorage<T,A> storage_type;

      /// Construct an evaluated tensor

      /// This will take ownership of the memory held by \c data
      /// \tparam SizeArray The input size array type.
      /// \param s An array with the size of of each dimension
      /// \param o The dimension ordering
      /// \param d The data for the tensor
      template <typename D>
      EvalTensor(const Range<D>& r, const storage_type& d) :
        range_(r), data_(d)
      { }

      /// Construct an evaluated tensor
      template <typename D, typename InIter>
      EvalTensor(const Range<D>& r, InIter it) :
        range_(r), data_(r.volume(), it)
      { }

      /// Copy constructor

      /// \param other The tile to be copied.
      template <typename Derived>
      EvalTensor(const ReadableTensor<Derived>& other) :
          range_(other.range()), data_(other.size(), other.begin())
      { }

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
        TA_ASSERT(size() == dest.size());
        std::copy(begin(), end(), dest.begin());
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return range_; }

      /// Tensor dimension size accessor

      /// \return The number of elements in the tensor
      size_type size() const { return range_.volume(); }

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

      /// Check the tensor dependancies

      /// Evaluate this tensor's dependancies and add them to the task object
      void check_dependancies(madness::TaskInterface*) const { }

      template <typename Archive>
      void serialize(Archive& ar) {
        ar & range_ & data_;
      }

    private:
      range_type range_; ///< Tensor size info
      storage_type data_; ///< Tensor data
    };

  } // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_EVAL_TENSOR_H__INCLUDED
