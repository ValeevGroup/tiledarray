#ifndef TILEDARRAY_MATH_H__INCLUDED
#define TILEDARRAY_MATH_H__INCLUDED

#include <TiledArray/annotated_array.h>
#include <TiledArray/tiled_range.h>
#include <world/typestuff.h>
#include <Eigen/Core>
#include <functional>
#include <iostream>

namespace TiledArray {
  namespace math {

    template<typename I>
    class PackedSizePair {
    public:
      typedef std::vector<I> size_array;
      typedef std::array<I, 3> packed_size_array;

      template<typename LeftRange, typename RightRange>
      PackedSizePair(const LeftRange& lrange, const expressions::VariableList& lvars,
          const RightRange& rrange, const expressions::VariableList& rvars)
      {
        typedef std::pair<expressions::VariableList::const_iterator,
            expressions::VariableList::const_iterator> vars_iter_pair;
        typedef std::pair<typename LeftRange::index::const_iterator,
            typename LeftRange::index::const_iterator> lindex_iter_pair;
        typedef std::pair<typename RightRange::index::const_iterator,
            typename RightRange::index::const_iterator> rindex_iter_pair;

        expressions::VariableList vars = lvars * rvars;

        // find common variable lists
        vars_iter_pair lvars_common;
        vars_iter_pair rvars_common;
        expressions::detail::find_common(lvars.begin(), lvars.end(), rvars.begin(),
            rvars.end(), lvars_common, rvars_common);

        lindex_iter_pair lstart_common = get_common_range_iterators(lrange.start().begin(),
            std::distance(lvars.begin(), lvars_common.first),
            std::distance(lvars.begin(), lvars_common.second));
        rindex_iter_pair rstart_common = get_common_range_iterators(rrange.start().begin(),
            std::distance(rvars.begin(), rvars_common.first),
            std::distance(rvars.begin(), rvars_common.second));

        lindex_iter_pair lfinish_common = get_common_range_iterators(lrange.finish().begin(),
            std::distance(lvars.begin(), lvars_common.first),
            std::distance(lvars.begin(), lvars_common.second));
        rindex_iter_pair rfinish_common = get_common_range_iterators(rrange.finish().begin(),
            std::distance(rvars.begin(), rvars_common.first),
            std::distance(rvars.begin(), rvars_common.second));

        TA_ASSERT((std::distance(lstart_common.first, lstart_common.second) ==
            std::distance(rstart_common.first, rstart_common.second)), std::runtime_error,
            "The size of the common start index does not match.");
        TA_ASSERT(std::equal(lstart_common.first, lstart_common.second, rstart_common.first),
            std::runtime_error, "The common start dimensions do not match.");
        TA_ASSERT((std::distance(lfinish_common.first, lfinish_common.second) ==
            std::distance(rfinish_common.first, rfinish_common.second)), std::runtime_error,
            "The size of the common start index does not match.");
        TA_ASSERT(std::equal(lfinish_common.first, lfinish_common.second, rfinish_common.first),
            std::runtime_error, "The common finish dimensions do not match.");

        // find dimensions of the result tile
        size_.resize(vars.dim(), 0);
        start_.resize(vars.dim(), 0);
        finish_.resize(vars.dim(), 0);
        typename size_array::iterator size_it = size_.begin();
        typename size_array::iterator start_it = start_.begin();
        typename size_array::iterator finish_it = finish_.begin();
        typename size_array::iterator size_end = size_.end();
        expressions::VariableList::const_iterator v_it = vars.begin();
        expressions::VariableList::const_iterator lvar_begin = lvars.begin();
        expressions::VariableList::const_iterator lvar_end = lvars.end();
        expressions::VariableList::const_iterator rvar_begin = rvars.begin();
        expressions::VariableList::const_iterator rvar_end = rvars.end();
        expressions::VariableList::const_iterator find_it;
        std::iterator_traits<expressions::VariableList::const_iterator>::difference_type n = 0;
        for(; size_it != size_end; ++size_it, ++start_it, ++finish_it, ++v_it) {
          if((find_it = std::find(lvar_begin, lvar_end, *v_it)) != lvars.end()) {
            n  = std::distance(lvar_begin, find_it);
            *start_it = lrange.start()[n];
            *finish_it = lrange.finish()[n];
            *size_it = *finish_it - *start_it;
          } else {
            find_it = std::find(rvar_begin, rvar_end, *v_it);
            n  = std::distance(rvar_begin, find_it);
            *start_it = rrange.start()[n];
            *finish_it = rrange.finish()[n];
            *size_it = *finish_it - *start_it;
          }
        }

        // calculate packed tile dimensions
        packed_left_size_[0] = accumulate(lrange.finish().begin(), lfinish_common.first, lrange.start().begin());
        packed_left_size_[2] = accumulate(lfinish_common.second, lrange.finish().end(), lstart_common.second);
        packed_right_size_[0] = accumulate(rrange.finish().begin(), rfinish_common.first, rrange.start().begin());
        packed_right_size_[2] = accumulate(rfinish_common.second, rrange.finish().end(), rstart_common.second);
        packed_left_size_[1] = accumulate(lfinish_common.first, lfinish_common.second, lstart_common.first);
        packed_right_size_[1] = packed_left_size_[1];
      }

      const size_array& size() const { return size_; }
      const size_array& start() const { return start_; }
      const size_array& finish() const { return finish_; }
      const packed_size_array& packed_left_size() const { return packed_left_size_; }
      const packed_size_array& packed_right_size() const { return packed_right_size_; }

      I m() const { return packed_left_size_[0]; }
      I n() const { return packed_left_size_[2]; }
      I o() const { return packed_right_size_[0]; }
      I p() const { return packed_right_size_[2]; }
      I i() const { return packed_left_size_[1]; }

    private:

      template<typename InIter1, typename InIter2>
      static I accumulate(InIter1 first1, InIter1 last1, InIter2 first2) {
        I initial = 1;
        while(first1 != last1)
          initial *= *first1++ - *first2++;

        return initial;
      }

      template<typename InIter, typename Diff>
      static std::pair<InIter, InIter> get_common_range_iterators(InIter first, Diff d1, Diff d2) {
        std::pair<InIter, InIter> result(first, first);
        std::advance(result.first, d1);
        std::advance(result.second, d2);

        return result;
      }

      size_array size_;
      size_array start_;
      size_array finish_;
      packed_size_array packed_left_size_;
      packed_size_array packed_right_size_;
    }; // class ContractedData


    template <typename T>
    struct TilePlus {
      typedef T tile_type;
      typedef typename tile_type::value_type value_type;

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

      // Compiler generated functions are OK here.

      tile_type operator()(const value_type& left, const tile_type& right) const {
        return left * right;
      }

      tile_type operator()(const tile_type& left, const value_type& right) const {
        return left * right;
      }
    }; // struct TileMinus

    template <typename Res, typename Left, typename Right>
    struct TileContract {
      typedef Res result_type;
      typedef const Left& first_argument_type;
      typedef const Right& second_argument_type;
      typedef typename result_type::range_type range_type;
      typedef typename result_type::index index;
      typedef typename result_type::ordinal_index ordinal_index;
      typedef typename result_type::value_type value_type;

      TileContract(const std::shared_ptr<expressions::VariableList>& lvar,
        const std::shared_ptr<expressions::VariableList>& rvar) :
          left_var_(lvar), right_var_(rvar)
      { }

      result_type operator()(first_argument_type left, second_argument_type right) const {
        PackedSizePair<ordinal_index> packed_sizes(left.range(), *left_var_,
            right.range(), *right_var_);

        ;

        result_type result(range_type(index(packed_sizes.start().begin()),
            index(packed_sizes.finish().begin())));

        contract(packed_sizes.m(), packed_sizes.n(), packed_sizes.o(),
            packed_sizes.p(), packed_sizes.i(), left.data(), right.data(),
            result.data());
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
          Eigen::Map<matrix_type> ma(a_begin, i, ma1);
          for(b_begin = b; b_begin != b_end; b_begin += b_step, c_begin += c_step) {
            Eigen::Map<matrix_type> mb(b_begin, i, mb1);
            Eigen::Map<matrix_type> mc(c_begin, ma1, mb1);

            mc = ma.transpose() * mb;
          }
        }
      }

      std::shared_ptr<expressions::VariableList> left_var_;
      std::shared_ptr<expressions::VariableList> right_var_;

    }; // struct Contract

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_H__INCLUDED
