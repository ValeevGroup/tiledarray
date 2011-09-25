#ifndef TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED

#include <TiledArray/eval_tensor.h>
#include <TiledArray/contraction.h>
#include <world/shared_ptr.h>
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
      typedef std::size_t size_type;
      typedef DynamicRange range_type;
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
      typedef math::Contraction contract_type; ///< Contraction type

    private:
      // not allowed
      ContractionTensor_& operator=(const ContractionTensor_&);

    public:

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param c Shared pointer to contraction object
      ContractionTensor(typename TensorArg<left_tensor_type>::type left, typename TensorArg<right_tensor_type>::type right, const std::shared_ptr<contract_type>& c) :
        left_(left),
        right_(right),
        range_(c->contract_range(left.range(), right.range())),
        data_(),
        contraction_(c)
      { }

      /// Copy constructor
      ContractionTensor(const ContractionTensor_& other) :
        left_(other.left_),
        right_(other.right_),
        range_(other.range_),
        data_(other.data_),
        contraction_(other.contraction_)
      { }

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
        TA_ASSERT(size() == dest.size());
        std::copy(begin(), end(), dest.begin());
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return range_; }

      /// Tensor size

      /// \return The number of elements in the tensor
      size_type size() const { return range_.volume(); }

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

      void check_dependancies(madness::TaskInterface* task) const {
        left_.check_dependancies(task);
        right_.check_dependancies(task);
      }

    private:

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
        if(data_.size() != size()) {
          const size_type s = size();

          typename contract_type::packed_size_array packed_size =
              contraction_->pack_arrays(left_.range().size(), right_.range().size());

          // We need to allocate storage and evaluate
          storage_type temp(s);
          typename Eval<left_tensor_type>::type left_eval = left_.eval();
          typename Eval<left_tensor_type>::type right_eval = right_.eval();

          if(range_.order() == TiledArray::detail::decreasing_dimension_order) {
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

      typename TensorMem<left_tensor_type>::type left_; ///< Left argument
      typename TensorMem<right_tensor_type>::type right_; ///< Right argument
      range_type range_; ///< Tensor size info
      mutable storage_type data_;
      std::shared_ptr<contract_type> contraction_;
    }; // class ContractionTensor

  } // namespace expressions



  namespace math {

    // This function needs to be here to break cyclic dependencies.
    template <typename LeftTensor, typename RightTensor>
    inline detail::Bitset<> Contraction::contract_shape(const LeftTensor& left, const RightTensor& right) {
      typedef expressions::EvalTensor<TiledArray::detail::Bitset<>::value_type > eval_tensor;
      eval_tensor left_map(left.range(), left.get_shape().begin());
      eval_tensor right_map(right.range(), right.get_shape().begin());

      expressions::ContractionTensor<eval_tensor, eval_tensor>
      contract_tensor(left_map, right_map,
          std::shared_ptr<Contraction>(this, & madness::detail::no_delete<Contraction>));

      return detail::Bitset<>(contract_tensor.begin(), contract_tensor.end());
    }

  } // namespace math


} // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
