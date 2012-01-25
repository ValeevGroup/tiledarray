#ifndef TILEDARRAY_CONTRACTION_H__INCLUDED
#define TILEDARRAY_CONTRACTION_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/range.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/bitset.h>
#include <TiledArray/permute_tensor.h>
#include <world/array.h>
#include <iterator>
#include <numeric>
#include <functional>
#include <list>

namespace TiledArray {
  namespace math {

    namespace {
      /// Contraction type selection for complex numbers

      /// \tparam T The left contraction argument type
      /// \tparam U The right contraction argument type
      template <typename T, typename U>
      struct ContractionValue {
        typedef T type; ///< The result type
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

    /// A utility class for contraction operations.

    /// \param I The index type
    /// \note All algorithms assume the object dimension layout is in the same
    /// order as the left- and right-hand variable list arguments used to
    /// construct this object.
    class Contraction {
    public:
      typedef std::vector<std::size_t> map_type;

      Contraction(const expressions::VariableList& left,
          const expressions::VariableList& right, const detail::DimensionOrderType& order)
      {

        // Reserve storage for maps.
        left_inner_.reserve(left.dim());
        left_outer_.reserve(left.dim());
        right_inner_.reserve(right.dim());
        right_outer_.reserve(right.dim());

        // construct the inner and outer maps.
        for(expressions::VariableList::const_iterator it = left.begin(); it != left.end(); ++it) {
          expressions::VariableList::const_iterator right_it =
              std::find(right.begin(), right.end(), *it);
          if(right_it == right.end())
            left_outer_.push_back(std::distance(left.begin(), it));
          else {
            left_inner_.push_back(std::distance(left.begin(), it));
            right_inner_.push_back(std::distance(right.begin(), right_it));
          }
        }

        for(expressions::VariableList::const_iterator it = right.begin(); it != right.end(); ++it) {
          expressions::VariableList::const_iterator left_it =
              std::find(left.begin(), left.end(), *it);
          if(left_it == left.end())
            right_outer_.push_back(std::distance(right.begin(), it));
        }

        init_permutation(left, left_inner_, left_outer_, order, perm_left_, do_perm_left_);
        init_permutation(right, right_inner_, right_outer_, order, perm_right_, do_perm_right_);
      }



      const Permutation& perm_left() const { return perm_left_; }
      const Permutation& perm_right() const { return perm_right_; }



      /// Contract array

      template <typename ResArray, typename LeftArray, typename RightArray>
      void contract_array(ResArray& res, const LeftArray& left, const RightArray& right) const {
        TA_ASSERT(left.size() == left_dim());
        TA_ASSERT(right.size() == right_dim());
        TA_ASSERT(res.size() == dim());

        typename ResArray::iterator res_it = res.begin();

        for(map_type::const_iterator it = left_outer_.begin(); it != left_outer_.end(); ++it)
          *res_it++ = left[*it];
        for(map_type::const_iterator it = right_outer_.begin(); it != right_outer_.end(); ++it)
          *res_it++ = right[*it];
      }

      expressions::VariableList contract_vars(const expressions::VariableList& left, const expressions::VariableList& right) {
        std::vector<std::string> res(dim());
        contract_array(res, left.data(), right.data());
        return expressions::VariableList(res.begin(),res.end());
      }

      /// Generate a contracted range object

      /// \tparam LeftRange The left-hand range type.
      /// \tparam RightRange The right-hand range type.
      /// \param left The left-hand-argument range
      /// \param right The right-hand-argument range
      /// \return A contracted range object
      /// \throw TiledArray::Exception When the order of the ranges do not match
      /// \throw TiledArray::Exception When the dimension of \c left and \c right
      /// do not match the dimensions of the variable lists used to construct
      /// this object.
      template <typename LeftRange, typename RightRange>
      DynamicRange contract_range(const LeftRange& left, const RightRange& right) const {
        TA_ASSERT(left.order() == right.order());
        TA_ASSERT(left.dim() == left_dim());
        TA_ASSERT(right.dim() == right_dim());

        typename DynamicRange::index start(dim()), finish(dim());
        contract_array(start, left.start(), right.start());
        contract_array(finish, left.finish(), right.finish());
        return DynamicRange(start, finish, left.order());
      }

      template <typename LeftTRange, typename RightTRange>
      DynamicTiledRange contract_trange(const LeftTRange& left, const RightTRange& right) const {
        TA_ASSERT(left.tiles().order() == right.tiles().order());
        typename DynamicTiledRange::Ranges ranges(dim());
        contract_array(ranges, left.data(), right.data());
        return DynamicTiledRange(ranges.begin(), ranges.end(), left.tiles().order());
      }

      // Definition is in contraction_tensor.h
      template <typename LeftTensor, typename RightTensor>
      inline detail::Bitset<> contract_shape(const LeftTensor& left, const RightTensor& right);

      /// Permute tensor contraction

      /// This function permutes the left- and right-hand tensor arguments such
      /// that the contraction can be performed with a matrix-matrix
      /// multiplication. The permutation applied to \c left and \c right are
      /// defined by the variable lists used to construct this object. The data
      /// layout will match the expected input of the \c contract_tensor()
      /// function. The resulting variable list can be obtained with the
      /// \c contract_vars() function.
      /// \tparam Left The left-hand-side tensor argument type
      /// \tparam Right The right-hand-side tensor argument type
      /// \param left The left-hand-side tensor argument
      /// \param right The right-hand-side tensor argument
      /// \throw TiledArray::Exception When the orders of left and right are not
      /// equal
      template <typename Left, typename Right>
      expressions::Tensor<typename ContractionValue<typename Left::value_type,
          typename Right::value_type>::type, DynamicRange>
      permute_contract_tensor(const Left& left, const Right& right) const {
        TA_ASSERT(left.range().order() == right.range().order());
        TA_ASSERT(left.range().dim() == left_dim());
        TA_ASSERT(right.range().dim() == right_dim());

        if(do_perm_left_) {
          if(do_perm_right_) {
            return contract_tensor(
                expressions::make_permute_tensor(left, perm_left_).eval(),
                expressions::make_permute_tensor(right, perm_right_).eval());
          } else {
            return contract_tensor(
                expressions::make_permute_tensor(left, perm_left_).eval(),
                right.eval());
          }
        } else {
          if(do_perm_right_) {
            return contract_tensor(left.eval(),
                expressions::make_permute_tensor(right, perm_right_).eval());
          } else {
            return contract_tensor(left.eval(), right.eval());
          }
        }
      }

      /// Calculate the outer dimension for the left argument

      /// Assume outer dimensions are all on the left for decreasing
      /// dimension ordering and to the right for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused outer dimensions for the left argument
      template <typename D>
      std::size_t left_outer(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin(), range.size().begin() + left_outer_dim());
        else
          return accumulate(range.size().begin() + left_inner_dim(), range.size().end());
      }

      /// Calculate the inner dimension for the left argument

      /// Assume inner dimensions are all on the right for decreasing
      /// dimension ordering and to the left for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused inner dimensions for the left argument
      template <typename D>
      std::size_t left_inner(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin() + left_outer_dim(), range.size().end());
        else
          return accumulate(range.size().begin(), range.size().begin() + left_inner_dim());
      }

      /// Calculate the outer dimension for the right argument

      /// Assume outer dimensions are all on the left for decreasing
      /// dimension ordering and to the right for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused outer dimensions for the right argument
      template <typename D>
      std::size_t right_outer(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin(), range.size().begin() + right_outer_dim());
        else
          return accumulate(range.size().begin() + right_inner_dim(), range.size().end());
      }

      /// Calculate the inner dimension for the right argument

      /// Assume inner dimensions are all on the right for decreasing
      /// dimension ordering and to the left for increasing dimension
      /// ordering.
      /// \tparam D A range type: StaticRange or DynamicRange
      /// \param range A range object for the right argument
      /// \return The size of the fused inner dimensions for the right argument
      template <typename D>
      std::size_t right_inner(const Range<D>& range) const {
        if(range.order() == TiledArray::detail::decreasing_dimension_order)
          return accumulate(range.size().begin() + right_outer_dim(), range.size().end());
        else
          return accumulate(range.size().begin(), range.size().begin() + right_inner_dim());
      }

      /// Tensor contraction

      /// The will contract \c left with \c right and return the result tensor.
      /// The contraction algorithms are: \n
      /// \c order=TiledArray::detail::decreasing_dimension_order
      /// \f[
      /// C_{m_1, m_2, \dots , n_1, n_2, \dots} =
      ///     \sum_{i_1, i_2, \dots}
      ///         A_{m_1, m_2, \dots, i_1, i_2, \dots}
      ///         B_{n_1, n_2, \dots, i_1, i_2, \dots}
      /// \f]
      /// \c order=TiledArray::detail::increasing_dimension_order
      /// \f[
      /// C_{m_1, m_2, \dots , n_1, n_2, \dots} =
      ///     \sum_{i_1, i_2, \dots}
      ///         A_{i_1, i_2, \dots, m_1, m_2, \dots}
      ///         B_{i_1, i_2, \dots, n_1, n_2, \dots}
      /// \f]
      /// If the data is not in the correct layout, then use the
      /// \c permute_contract_tensor() function instead.
      /// \tparam Left The left-hand-side tensor argument type
      /// \tparam Right The right-hand-side tensor argument type
      /// \param left The left-hand-side tensor argument
      /// \param right The right-hand-side tensor argument
      /// \throw TiledArray::Exception When the orders of left and right are not
      /// equal.
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c left tensor is not equal to \c left_dim() .
      /// \throw TiledArray::Exception  When the number of dimensions of the
      /// \c right tensor is not equal to \c right_dim() .
      /// \throw TiledArray::Exception When the inner dimensions of \c A and
      /// \c B are not coformal (i.e. the number and range of the inner
      /// dimensions are not the same).
      template <typename Left, typename Right>
      expressions::Tensor<typename ContractionValue<typename Left::value_type,
          typename Right::value_type>::type, DynamicRange>
      contract_tensor(const Left& left, const Right& right) const {
        // Check that the order and dimensions of the left and right tensors are correct.
        TA_ASSERT(left.range().order() == right.range().order());
        TA_ASSERT(left.range().dim() == left_dim());
        TA_ASSERT(right.range().dim() == right_dim());
        TA_ASSERT(check_coformal(left.range(), right.range()));

        // This function fuses the inner and outer dimensions of the left- and
        // right-hand tensors such that the contraction can be performed with a
        // matrix-matrix multiplication.

        // Construct the result tensor
        typedef typename ContractionValue<typename Left::value_type,
            typename Right::value_type>::type value_type;
        expressions::Tensor<value_type, DynamicRange>
            res(DynamicRange(
                result_index(left.range().start(), right.range().start(), left.range().order()),
                result_index(left.range().finish(), right.range().finish(), left.range().order()),
                left.range().order()));

        const std::size_t m = left_outer(left.range());
        const std::size_t i = left_inner(left.range());
        const std::size_t n = right_outer(right.range());

        if(left.range().order() == detail::decreasing_dimension_order) {

          // Construct matrix maps for tensors
          Eigen::Map<const Eigen::Matrix<typename Left::value_type, Eigen::Dynamic,
              Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> >
              ma(left.data(), m, i);
          Eigen::Map<const Eigen::Matrix<typename Right::value_type, Eigen::Dynamic,
              Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> >
              mb(right.data(), n, i);
          Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic ,
              Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> >
              mc(res.data(), m, n);

          // Do contraction
          mc.noalias() = ma * mb.transpose();

        } else {

          // Construct matrix maps for tensors
          Eigen::Map<const Eigen::Matrix<typename Left::value_type, Eigen::Dynamic,
              Eigen::Dynamic, Eigen::ColMajor | Eigen::AutoAlign> >
              ma(left.data(), i, m);
          Eigen::Map<const Eigen::Matrix<typename Right::value_type, Eigen::Dynamic,
              Eigen::Dynamic, Eigen::ColMajor | Eigen::AutoAlign> >
              mb(right.data(), i, n);
          Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic ,
              Eigen::Dynamic, Eigen::ColMajor | Eigen::AutoAlign> >
              mc(res.data(), m, n);

          // Do contraction
          mc.noalias() = ma.transpose() * mb;
        }

        return res;
      }


      unsigned int dim() const { return left_outer_.size() + right_outer_.size(); }
      std::size_t left_inner_dim() const { return left_inner_.size(); }
      std::size_t left_outer_dim() const { return left_outer_.size(); }
      std::size_t right_inner_dim() const { return right_inner_.size(); }
      std::size_t right_outer_dim() const { return right_outer_.size(); }
      std::size_t left_dim() const { return left_inner_dim() + left_outer_dim(); }
      std::size_t right_dim() const { return right_inner_dim() + right_outer_dim(); }
      std::size_t res_dim() const { return left_outer_dim() + right_outer_dim(); }

      /// Get the left argument variable list

      /// \return The permuted left variable list
      expressions::VariableList left_vars(const expressions::VariableList& v) const {
        if(do_perm_left_)
          return perm_left_ ^ v;

        return v;
      }

      /// Get the right argument variable list

      /// \return The permuted right variable list
      expressions::VariableList right_vars(const expressions::VariableList& v) const {
        if(do_perm_right_)
          return perm_right_ ^ v;

        return v;
      }

    private:

      /// Check that the left and right arrays are coformal

      ///
      template <typename LeftRange, typename RightRange>
      bool check_coformal(const LeftRange& left, const RightRange& right) const {
        const TiledArray::detail::DimensionOrderType order = left.order();
        return std::equal(
            (order == TiledArray::detail::decreasing_dimension_order ? left.start().begin() + left_outer_dim() : left.start().begin()),
            (order == TiledArray::detail::decreasing_dimension_order ? left.start().end() : left.start().begin() + left_inner_dim()),
            (order == TiledArray::detail::decreasing_dimension_order ? right.start().begin() + right_outer_dim() : right.start().begin()))
            && std::equal(
            (order == TiledArray::detail::decreasing_dimension_order ? left.finish().begin() + left_outer_dim() : left.finish().begin()),
            (order == TiledArray::detail::decreasing_dimension_order ? left.finish().end() : left.finish().begin() + left_inner_dim()),
            (order == TiledArray::detail::decreasing_dimension_order ? right.finish().begin() + right_outer_dim() : right.finish().begin()));
      }

      /// Product accumulation

      ///
      /// \tparam InIter The input iterator type
      /// \param first The start of the iterator range to be accumulated
      /// \param first The end of the iterator range to be accumulated
      /// \return The product of each value in the iterator range.
      template <typename InIter>
      static typename std::iterator_traits<InIter>::value_type accumulate(InIter first, InIter last) {
        typename std::iterator_traits<InIter>::value_type result = 1ul;
        for(; first != last; ++first)
          result *= *first;
        return result;
      }

      template <typename LeftIndex, typename RightIndex>
      DynamicRange::index result_index(const LeftIndex& l_index,
          const RightIndex& r_index, const detail::DimensionOrderType& order) const {
        DynamicRange::index result(dim());

        if(order == detail::decreasing_dimension_order)
          std::copy(r_index.begin(), r_index.begin() + right_outer_dim(),
              std::copy(l_index.begin(), l_index.begin() + left_outer_dim(), result.begin()));
        else
          std::copy(r_index.begin() + right_inner_dim(), r_index.end(),
              std::copy(l_index.begin() + left_inner_dim(), l_index.end(), result.begin()));

        return result;
      }


      static void init_inner_outer(const expressions::VariableList& first,
          const expressions::VariableList& second, map_type& inner, map_type& outer)
      {
        // Reserve storage for maps.
        inner.reserve(first.dim());
        outer.reserve(first.dim());

        // construct the inner and outer maps.
        for(expressions::VariableList::const_iterator it = first.begin(); it != first.end(); ++it) {
          expressions::VariableList::const_iterator second_it =
              std::find(second.begin(), second.end(), *it);
          if(second_it == second.end())
            outer.push_back(std::distance(first.begin(), it));
          else
            inner.push_back(std::distance(first.begin(), it));
        }
      }

      static void init_permutation(const expressions::VariableList& in_vars,
          const map_type& inner, const map_type& outer, const detail::DimensionOrderType& order,
          Permutation& perm, bool& do_perm)
      {
        std::vector<std::string> vars;
        vars.reserve(in_vars.size());

        if(order == TiledArray::detail::decreasing_dimension_order) {
          for(map_type::const_iterator it = outer.begin(); it != outer.end(); ++it)
            vars.push_back(in_vars[*it]);
          for(map_type::const_iterator it = inner.begin(); it != inner.end(); ++it)
            vars.push_back(in_vars[*it]);
        } else {
          for(map_type::const_iterator it = inner.begin(); it != inner.end(); ++it)
            vars.push_back(in_vars[*it]);
          for(map_type::const_iterator it = outer.begin(); it != outer.end(); ++it)
            vars.push_back(in_vars[*it]);
        }
        expressions::VariableList perm_vars(vars.begin(), vars.end());
        perm = perm_vars.permutation(in_vars);

        do_perm = perm_vars != in_vars;
      }


      map_type left_inner_;
      map_type left_outer_;
      map_type right_inner_;
      map_type right_outer_;

      Permutation perm_left_;
      Permutation perm_right_;

      bool do_perm_left_;
      bool do_perm_right_;
    }; // class Contraction

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_H__INCLUDED
