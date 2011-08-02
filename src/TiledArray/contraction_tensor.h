#ifndef TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED

#include <TiledArray/eval_tensor.h>
#include <TiledArray/contraction.h>
#include <world/tr1/memory.h>
#include <Eigen/Core>
#include <functional>

namespace TiledArray {
  namespace expressions {

    template <typename, typename>
    class ContractionTensor;

    namespace {
      template <typename T, typename U, typename Enable = void>
      struct ContractionValue {
        typedef T type;
      };

      template <typename T>
      struct ContractionValue<T, std::complex<T> > {
        typedef std::complex<T> type;
      };

      template <typename T>
      struct ContractionValue<std::complex<T>, T> {
        typedef std::complex<T> type;
      };
    } // namespace

    template <typename LeftArg, typename RightArg>
    struct TensorTraits<ContractionTensor<LeftArg, RightArg> > {
      typedef typename TensorSize::size_type size_type;
      typedef typename TensorSize::size_array size_array;
      typedef typename ContractionValue<typename LeftArg::value_type,
          typename RightArg::value_type>::type value_type;
      typedef typename DenseStorage<value_type>::const_reference const_reference;
      typedef typename DenseStorage<value_type>::const_iterator const_iterator;
      typedef typename DenseStorage<value_type>::difference_type difference_type;
      typedef typename DenseStorage<value_type>::const_pointer const_pointer;
    }; // struct TensorTraits<ContractionTensor<LeftArg, RightArg> >

    template <typename LeftArg, typename RightArg>
    struct Eval<ContractionTensor<LeftArg, RightArg> > {
      typedef const ContractionTensor<LeftArg, RightArg>& type;
    }; // struct Eval<ContractionTensor<LeftArg, RightArg> >

    /// Tensor that is composed from two contracted argument tensors

    /// The tensor elements are constructed using a binary transformation
    /// operation.
    /// \tparam LeftArg The left-hand argument type
    /// \tparam RightArg The right-hand argument type
    /// \tparam Op The binary transform operator type.
    template <typename LeftArg, typename RightArg>
    class ContractionTensor : public DirectReadableTensor<ContractionTensor<LeftArg, RightArg> > {
    public:
      typedef ContractionTensor<LeftArg, RightArg> ContractionTensor_;
      typedef LeftArg left_tensor_type;
      typedef RightArg right_tensor_type;
      TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF(DirectReadableTensor<ContractionTensor_>, ContractionTensor_);
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object
      typedef math::Contraction<size_type> contract_type; ///< Contraction type

      /// Default constructor
      ContractionTensor() :
        left_(NULL),
        right_(NULL),
        size_(),
        data_(),
        contraction_()
      { }

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param c Shared pointer to contraction object
      ContractionTensor(const left_tensor_type& left, const right_tensor_type& right, const std::shared_ptr<contract_type>& c) :
        left_(&left),
        right_(&right),
        size_(constract_size(left.size(), right.size(), c), left.order()),
        data_(),
        contraction_(c)
      {
        TA_ASSERT(left.order() == right.order());
      }

      /// Copy constructor
      ContractionTensor(const ContractionTensor_& other) :
        left_(other.left_),
        right_(other.right_),
        size_(other.size_),
        data_(other.data_),
        contraction_(other.contraction_)
      { }

      ContractionTensor_& operator=(const ContractionTensor_& other) {
        left_ = other.left_;
        right_ = other.right_;
        size_ = other.size_;
        data_ = other.data_;
        contraction_ = other.contraction_;

        return *this;
      }


      /// Evaluate this tensor

      /// \return An evaluated tensor object
      const ContractionTensor_& eval() const {
        lazy_eval();
        return *this;
      }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
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

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const {
        lazy_eval();
        return data_.begin();
      }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const {
        lazy_eval();
        return data_.end();
      }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        lazy_eval();
        return data_[i];
      }

    private:


      template <typename LeftSize, typename RightSize>
      static size_array constract_size(const LeftSize& left, const RightSize& right, const std::shared_ptr<contract_type>& c) {
        size_array result(c->dim());
        c->contract_array(result, left, right);
        return result;
      }


      /// Contract a and b, and place the results into c.
      /// c[m,o,n,p] = a[m,i,n] * b[o,i,p]
      template <typename MatrixType>
      static void contract(const size_type m, const size_type n, const size_type o,
          const size_type p, const size_type i, const value_type* a,
          const value_type* b, value_type* c)
      {
        // calculate iterator step sizes.
        const std::size_t a_step = i * n;
        const std::size_t b_step = i * p;
        const std::size_t c_step = n * p;

        // calculate iterator boundaries
        const value_type* a_begin = NULL;
        const value_type* b_begin = NULL;
        value_type* c_begin = c;
        const value_type* const a_end = a + (m * i * n);
        const value_type* const b_end = b + (o * i * p);
//        const T* const c_end = c + (m * n * o * p);

        // iterate over the highest order dimensions of a and b, and store the
        // results of the matrix-matrix multiplication.
        for(a_begin = a; a_begin != a_end; a_begin += a_step) {
          Eigen::Map<const MatrixType> ma(a_begin, i, n);
          for(b_begin = b; b_begin != b_end; b_begin += b_step, c_begin += c_step) {
            Eigen::Map<const MatrixType> mb(b_begin, i, p);
            Eigen::Map<MatrixType> mc(c_begin, n, p);

            mc.noalias() = ma.transpose() * mb;
          }
        }
      }



      /// Evaluate the tensor only when the data is needed
      void lazy_eval() const {
        TA_ASSERT(left_ != NULL);
        TA_ASSERT(right_ != NULL);
        if(data_.volume() != volume()) {
          const size_type v = volume();

          typename contract_type::packed_size_array packed_size =
              contraction_->pack_arrays(left_->size(), right_->size());

          // We need to allocate storage and evaluate
          storage_type temp(v);
          typename Eval<left_tensor_type>::type left_eval = left_->eval();
          typename Eval<left_tensor_type>::type right_eval = right_->eval();

          if(order() == TiledArray::detail::decreasing_dimension_order) {
            typedef Eigen::Matrix< value_type , Eigen::Dynamic , Eigen::Dynamic,
                Eigen::RowMajor | Eigen::AutoAlign > matrix_type;
            contract<matrix_type>(packed_size[0], packed_size[1], packed_size[2], packed_size[3],
                packed_size[4], left_eval.data(), right_eval.data(), temp.data());
          } else {
            typedef Eigen::Matrix< value_type , Eigen::Dynamic , Eigen::Dynamic,
                Eigen::ColMajor | Eigen::AutoAlign > matrix_type;
            contract<matrix_type>(packed_size[1], packed_size[0], packed_size[3], packed_size[2],
                packed_size[4], left_eval.data(), right_eval.data(), temp.data());
          }

          temp.swap(data_);
        }
      }

      template <typename Derived>
      static const DirectReadableTensor<Derived>&
      eval_arg(const DirectReadableTensor<Derived>& arg) { return arg; }

      template <typename Derived>
      static const DirectWritableTensor<Derived>&
      eval_arg(const DirectWritableTensor<Derived>& arg) { return arg; }

      const left_tensor_type* left_; ///< Left argument
      const right_tensor_type* right_; ///< Right argument
      TensorSize size_; ///< Tensor size info
      mutable storage_type data_;
      std::shared_ptr<contract_type> contraction_;
    }; // class ContractionTensor

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
