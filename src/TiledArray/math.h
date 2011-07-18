#ifndef TILEDARRAY_MATH_H__INCLUDED
#define TILEDARRAY_MATH_H__INCLUDED

#include <TiledArray/tiled_range.h>
#include <TiledArray/permutation.h>
#include <TiledArray/variable_list.h>
#include <world/typestuff.h>
#include <Eigen/Core>
#include <functional>
#include <iostream>

namespace TiledArray {
  namespace math {

    /// A utility class for contraction operations.

    /// \param I The index type
    template<typename I>
    class Contraction {
    public:
      typedef std::array<I, 5> packed_size_array;

      Contraction(const expressions::VariableList& left,
          const expressions::VariableList& right)
      {
        typedef std::pair<expressions::VariableList::const_iterator,
            expressions::VariableList::const_iterator> vars_iter_pair;

        vars_iter_pair lvars_common;
        vars_iter_pair rvars_common;
        expressions::detail::find_common(left.begin(), left.end(), right.begin(),
            right.end(), lvars_common, rvars_common);

        left_[0] = std::distance(left.begin(), lvars_common.first);
        left_[1] = std::distance(left.begin(), lvars_common.second);
        left_[2] = left.dim();

        right_[0] = std::distance(right.begin(), rvars_common.first);
        right_[1] = std::distance(right.begin(), rvars_common.second);
        right_[2] = right.dim();
      }

      template <typename LeftArray, typename RightArray>
      packed_size_array pack_arrays(const LeftArray& left, const RightArray& right) {
        TA_ASSERT(left.size() == left_[2], std::range_error,
            "The dimensions of the left array do not match the dimensions of the packing.");
        TA_ASSERT(right.size() == right_[2], std::range_error,
            "The dimensions of the left array do not match the dimensions of the packing.");

        // Get the left iterator
        typename LeftArray::const_iterator l0 = left.begin();
        typename LeftArray::const_iterator l1 = l0;
        std::advance(l1, left_[0]);
        typename LeftArray::const_iterator l2 = l0;
        std::advance(l2, left_[1]);
        typename LeftArray::const_iterator l3 = left.end();

        // Get the right array iterator boundaries.
        typename RightArray::const_iterator r0 = right.begin();
        typename RightArray::const_iterator r1 = r0;
        std::advance(r1, right_[0]);
        typename RightArray::const_iterator r2 = r0;
        std::advance(r2, right_[1]);
        typename RightArray::const_iterator r3 = right.end();

        // Make accumulator operation object
        std::multiplies<I> acc_op;

        // Calculate packed dimensions.
        packed_size_array result = {{
            std::accumulate(l0, l1, I(1), acc_op),
            std::accumulate(l2, l3, I(1), acc_op),
            std::accumulate(r0, r1, I(1), acc_op),
            std::accumulate(r2, r3, I(1), acc_op),
            std::accumulate(l1, l2, I(1), acc_op)
          }};

        return result;
      }

      /// Contract array

      /// a{{a1...ai},{ai+1...aj-1},{aj...ak}} + b{{b1...bi},{bi+1...bj-1},{bj...bk}}
      /// ==> c{{a1...ai},{b1...bi},{aj...ak},{bj...bk}}
      template <typename ResArray, typename LeftArray, typename RightArray>
      void contract_array(ResArray& res, const LeftArray& left, const RightArray& right) {
        TA_ASSERT(left.size() == left_[2], std::range_error,
            "The dimensions of the left array do not match the dimensions of the packing.");
        TA_ASSERT(right.size() == right_[2], std::range_error,
            "The dimensions of the left array do not match the dimensions of the packing.");
        TA_ASSERT(res.size() == (left_[2] - left_[1] + left_[0] + right_[2] - right_[1] + right_[0]),
            std::range_error, "The dimensions of the result array do not match the dimensions of the contraction.");

        std::size_t l, r, i = 0;

        for(l = 0; l < left_[0]; ++l) res[i++] = left[l];
        for(r = 0; r < right_[0]; ++r) res[i++] = right[r];
        for(l = left_[1]; l < left_[2]; ++l) res[i++] = left[l];
        for(r = right_[1]; r < right_[2]; ++r) res[i++] = right[r];
      }

      template <typename ResRange, typename LeftRange, typename RightRange>
      void contract_range(ResRange& res, const LeftRange& left, const RightRange& right) {
        typename ResRange::index start, finish;
        contract_array(start, left.start(), right.start());
        contract_array(finish, left.finish(), right.finish());
        res.resize(start, finish);
      }

      template <typename ResTRange, typename LeftTRange, typename RightTRange>
      void contract_trange(ResTRange& res, const LeftTRange& left, const RightTRange& right) {
        typename ResTRange::Ranges ranges;
        contract_array(ranges, left.data(), right.data());
        res.resize(ranges.begin(), ranges.end());
      }

    private:


      typedef std::array<std::size_t, 3> pack_boundary_array;

      pack_boundary_array left_;
      pack_boundary_array right_;
    }; // class Contraction



    template <typename T>
    struct TilePlus {
      typedef T tile_type;
      typedef typename tile_type::value_type value_type;

      typedef tile_type result_type;

      // Compiler generated functions are OK here.

      tile_type operator()(const tile_type& left, const tile_type& right) const {
        return left + right;
      }

      tile_type operator()(const value_type& left, const tile_type& right) const {
        return left + right;
      }

      tile_type operator()(const tile_type& left, const value_type& right) const {
        return left + right;
      }
    }; // struct TilePlus

    template <typename T>
    struct TileMinus {
      typedef T tile_type;
      typedef typename tile_type::value_type value_type;

      typedef tile_type result_type;

      // Compiler generated functions are OK here.

      tile_type operator()(const tile_type& left, const tile_type& right) const {
        return left - right;
      }

      tile_type operator()(const value_type& left, const tile_type& right) const {
        return left - right;
      }

      tile_type operator()(const tile_type& left, const value_type& right) const {
        return left - right;
      }
    }; // struct TileMinus

    template <typename T>
    struct TileScale {
      typedef T tile_type;
      typedef typename tile_type::value_type value_type;

      typedef tile_type result_type;
      typedef const tile_type& argument_type;

      TileScale(const value_type& value) : value_(value) { }

      result_type operator()(argument_type arg) const {
        return value_ * arg;
      }

    private:
      value_type value_;
    }; // struct TileScale

    template <typename T>
    struct TilePermute {
      typedef T tile_type;
      typedef Permutation<tile_type::coordinate_system::dim> perm_type;

      typedef tile_type result_type;
      typedef const tile_type& argument_type;

      TilePermute(const perm_type& p) : perm_(p) { }

      result_type operator()(argument_type arg) const {
        return perm_ ^ arg;
      }

      const perm_type& permu() const { return perm_; }

    private:
      perm_type perm_;
    }; // struct TileScale

    template <typename Res, typename Left, typename Right>
    struct TileContract {
      typedef Res result_type;
      typedef const Left& first_argument_type;
      typedef const Right& second_argument_type;
      typedef typename result_type::range_type range_type;
      typedef typename result_type::index index;
      typedef typename result_type::ordinal_index ordinal_index;
      typedef typename result_type::value_type value_type;

      TileContract(const std::shared_ptr<Contraction<ordinal_index> >& c, const range_type& r) :
          contraction_(c), range_(r)
      { }

      TileContract(const std::shared_ptr<Contraction<ordinal_index> >& c) :
          contraction_(c), range_()
      { }

      TileContract(const expressions::VariableList& left,
          const expressions::VariableList& right) :
            contraction_(left, right), range_()
      { }

      result_type operator()(first_argument_type left, second_argument_type right) const {
        typename Contraction<ordinal_index>::packed_size_array size =
            contraction_->pack_arrays(left.range().size(), right.range().size());

        if(range_.volume() == 0) {
          if((left.range().volume() != 0) && (right.range().volume() != 0))
            contraction_->contract_range(range_, left.range(), right.range());
        } else {
          if((left.range().volume() == 0) || (right.range().volume() == 0))
            range_ = range_type();
        }

        result_type result(range_);

        if(result.range().volume() != 0)
          contract(size[0], size[1], size[2], size[3], size[4],
              left.data(), right.data(), result.data());

        return result;
      }

    private:

      /// Contract a and b, and place the results into c.
      /// c[m,o,n,p] = a[m,i,n] * b[o,i,p]
      static void contract(const ordinal_index m, const ordinal_index n, const ordinal_index o,
          const ordinal_index p, const ordinal_index i, const value_type* a,
          const value_type* b, value_type* c)
      {
        typedef Eigen::Matrix< value_type , Eigen::Dynamic , Eigen::Dynamic,
            (result_type::coordinate_system::order == detail::decreasing_dimension_order ?
            Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;

        // determine the lower order dimension size
        const std::size_t ma1 = ( result_type::coordinate_system::order == detail::increasing_dimension_order ? m : n );
        const std::size_t mb1 = ( result_type::coordinate_system::order == detail::increasing_dimension_order ? o : p );

        // calculate iterator step sizes.
        const std::size_t a_step = i * ma1;
        const std::size_t b_step = i * mb1;
        const std::size_t c_step = ma1 * mb1;

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
          Eigen::Map<const matrix_type> ma(a_begin, i, ma1);
          for(b_begin = b; b_begin != b_end; b_begin += b_step, c_begin += c_step) {
            Eigen::Map<const matrix_type> mb(b_begin, i, mb1);
            Eigen::Map<matrix_type> mc(c_begin, ma1, mb1);

            mc = ma.transpose() * mb;
          }
        }
      }

      std::shared_ptr<Contraction<ordinal_index> > contraction_;
      mutable typename result_type::range_type range_;
    }; // struct Contract

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_H__INCLUDED
