#ifndef TILEDARRAY_CONTRACTION_TENSOR_IMPL_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TENSOR_IMPL_H__INCLUDED

#include <TiledArray/tensor_expression_impl.h>
#include <TiledArray/tensor.h>
#include <TiledArray/cyclic_pmap.h>
#include <world/shared_ptr.h>

namespace TiledArray {
  namespace expressions {

    template <typename Left, typename Right>
    class ContractionTensorImpl : public TensorExpressionImpl<DynamicTiledRange,
        Tensor<typename math::ContractionValue<typename Left::value_type::value_type,
        typename Right::value_type::value_type>::type, typename DynamicTiledRange::range_type> >
    {
    public:
      // Base class typedefs
      typedef TensorExpressionImpl<DynamicTiledRange,
          Tensor<typename math::ContractionValue<typename Left::value_type::value_type,
          typename Right::value_type::value_type>::type, typename DynamicTiledRange::range_type> >
          TensorExpressionImpl_;
      typedef typename TensorExpressionImpl_::TensorImplBase_ TensorImplBase_;

      typedef Left left_tensor_type; ///< The left tensor type
      typedef typename left_tensor_type::value_type left_value_type; /// The left tensor value type
      typedef Right right_tensor_type; ///< The right tensor type
      typedef typename right_tensor_type::value_type right_value_type; ///< The right tensor value type

      typedef typename TensorImplBase_::size_type size_type; ///< size type
      typedef typename TensorImplBase_::pmap_interface pmap_interface; ///< The process map interface type
      typedef typename TensorImplBase_::trange_type trange_type;
      typedef typename TensorImplBase_::range_type range_type;
      typedef typename TensorImplBase_::value_type value_type; ///< The result value type
      typedef typename TensorImplBase_::storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename TensorImplBase_::storage_type::future const_reference;

    private:
      left_tensor_type left_;
      right_tensor_type right_;
      std::shared_ptr<math::Contraction> cont_;

    protected:
      const ProcessID rank_; ///< This process's rank
      const ProcessID size_; ///< Then number of processes
      const size_type m_; ///< Number of element rows in the result and left matrix
      const size_type n_; ///< Number of element columns in the result matrix and rows in the right argument matrix
      const size_type k_; ///< Number of element columns in the left and right argument matrices
      const size_type mk_; ///< Number of elements in left matrix
      const size_type nk_; ///< Number of elements in right matrix
      const size_type proc_cols_; ///< Number of columns in the result process map
      const size_type proc_rows_; ///< Number of rows in the result process map
      const size_type proc_size_; ///< Number of process in the process map. This may be
                         ///< less than the number of processes in world.
      const ProcessID rank_row_; ///< This node's row in the process map
      const ProcessID rank_col_; ///< This node's column in the process map
      const size_type local_rows_; ///< The number of local element rows
      const size_type local_cols_; ///< The number of local element columns
      const size_type local_size_; ///< Number of local elements

    public:

      ContractionTensorImpl(const left_tensor_type& left, const right_tensor_type& right,
        const std::shared_ptr<math::Contraction>& cont) :
          TensorExpressionImpl_(left.get_world(),
              cont->contract_vars(left.vars(), right.vars()),
              cont->contract_trange(left.trange(), right.trange())),
          left_(left), right_(right),
          cont_(cont),
          rank_(TensorImplBase_::get_world().rank()),
          size_(TensorImplBase_::get_world().size()),
          n_(cont->right_outer_init(right_.range())),
          m_(cont_->left_outer_init(left_.range())),
          k_(cont_->left_inner_init(left_.range())),
          mk_(m_ * k_),
          nk_(n_ * k_),
          proc_cols_(std::min(size_ / std::max(std::min<std::size_t>(std::sqrt(size_ * m_ / n_), size_), 1ul), n_)),
          proc_rows_(std::min(size_ / proc_cols_, m_)),
          proc_size_(proc_cols_ * proc_rows_),
          rank_row_((rank_ < proc_size_ ? rank_ / proc_cols_ : -1)),
          rank_col_((rank_ < proc_size_ ? rank_ % proc_cols_ : -1)),
          local_rows_((rank_ < proc_size_ ? (m_ / proc_rows_) + (m_ % proc_rows_ ? 1 : 0) : 0)),
          local_cols_((rank_ < proc_size_ ? (n_ / proc_cols_) + (n_ % proc_cols_ ? 1 : 0) : 0)),
          local_size_(local_rows_ * local_cols_)
      {
        // Initialize the shape if the tensor is not dense
        if(! (left.is_dense() || right.is_dense())) {
          typedef TiledArray::detail::Bitset<>::value_type bool_type;

          // Construct left and right shape maps
          Tensor<bool_type, typename left_tensor_type::range_type>
              left_map(left_.range(), left_.get_shape().begin());
          Tensor<bool_type, typename right_tensor_type::range_type>
              right_map(right_.range(), right_.get_shape().begin());

          Tensor<bool_type, range_type> res =
                cont_->contract_tensor(left_map, right_map);

          const size_type n = TensorImplBase_::size();
          TensorImplBase_::shape(TiledArray::detail::Bitset<>(n));
          for(size_type i = 0; i < n; ++i)
            TensorImplBase_::shape(i, res[i]);
        }
      }

      virtual ~ContractionTensorImpl() { }

      const left_tensor_type& left() const { return left_; }

      left_tensor_type& left() { return left_; }

      const right_tensor_type& right() const { return right_; }

      right_tensor_type& right() { return right_; }

      /// Contraction operation

      /// Contract \c left and \c right to \c result .
      /// \param[out] result The tensor that will store the result
      /// \param[in] left The left hand tensor argument
      /// \param[in] right The right hand tensor argument
      void contract(value_type& result, const left_value_type& left, const right_value_type& right) {
        cont_->contract_tensor(result, left, right);
      }

      /// Contraction object accessor

      /// \return a shared pointer to the contraction object
      const std::shared_ptr<math::Contraction>& contract() const { return cont_; }

      /// Factory function for the left argument process map

      /// \return A shared pointer that contains the left process map
      std::shared_ptr<pmap_interface> make_left_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::CyclicPmap(
            TensorImplBase_::get_world(), m_, k_, proc_rows_, proc_cols_));
      }

      /// Factory function for the right argument process map

      /// \return A shared pointer that contains the right process map
      std::shared_ptr<pmap_interface> make_righ_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::CyclicPmap(
            TensorImplBase_::get_world(), n_, k_, proc_cols_, proc_rows_));
      }
    }; // class ContractionAlgorithmBase

  } // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_TENSOR_IMPL_H__INCLUDED
