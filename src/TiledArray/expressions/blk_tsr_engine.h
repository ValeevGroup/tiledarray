/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  blk_tsr_engine.h
 *  May 20, 2015
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_BLK_TSR_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BLK_TSR_ENGINE_H__INCLUDED

#include <TiledArray/expressions/leaf_engine.h>
#include <TiledArray/tile_op/shift.h>
#include <TiledArray/tile_op/scal_shift.h>

namespace TiledArray {

  // Forward declaration
  template <typename, unsigned int, typename, typename> class Array;

  namespace expressions {

    // Forward declaration
    template <typename> class BlkTsrExpr;
    template <typename> class ScalBlkTsrExpr;
    template <typename> class BlkTsrEngine;
    template <typename> class ScalBlkTsrEngine;

    template <typename T, unsigned int DIM, typename Tile, typename Policy>
    struct EngineTrait<BlkTsrEngine<Array<T, DIM, Tile, Policy> > > {
      // Argument typedefs
      typedef Array<T, DIM, Tile, Policy> array_type; ///< The array type

      // Operational typedefs
      typedef TiledArray::math::Shift<typename array_type::eval_type,
          typename array_type::eval_type, false> op_type; ///< The tile operation
      typedef TiledArray::detail::LazyArrayTile<typename array_type::value_type,
          op_type> value_type;  ///< Tile type
      typedef typename eval_trait<value_type>::type eval_type;  ///< Evaluation tile type
      typedef typename TiledArray::detail::scalar_type<Array<T, DIM, Tile, Policy> >::type scalar_type;
      typedef Policy policy; ///< Policy type
      typedef TiledArray::detail::DistEval<value_type, policy> dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename policy::size_type size_type; ///< Size type
      typedef typename policy::trange_type trange_type; ///< Tiled range type
      typedef typename policy::shape_type shape_type; ///< Shape type
      typedef typename policy::pmap_interface pmap_interface; ///< Process map interface type

      static constexpr bool consumable = true;
      static constexpr unsigned int leaves = 1;
    };


    template <typename T, unsigned int DIM, typename Tile, typename Policy>
    struct EngineTrait<ScalBlkTsrEngine<Array<T, DIM, Tile, Policy> > > {
      // Argument typedefs
      typedef Array<T, DIM, Tile, Policy> array_type; ///< The array type

      // Operational typedefs
      typedef TiledArray::math::ScalShift<typename array_type::eval_type,
          typename array_type::eval_type, false> op_type; ///< The tile operation
      typedef TiledArray::detail::LazyArrayTile<typename array_type::value_type,
          op_type> value_type;  ///< Tile type
      typedef typename eval_trait<value_type>::type eval_type;  ///< Evaluation tile type
      typedef typename TiledArray::detail::scalar_type<Array<T, DIM, Tile, Policy> >::type scalar_type;
      typedef Policy policy; ///< Policy type
      typedef TiledArray::detail::DistEval<value_type, policy> dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename policy::size_type size_type; ///< Size type
      typedef typename policy::trange_type trange_type; ///< Tiled range type
      typedef typename policy::shape_type shape_type; ///< Shape type
      typedef typename policy::pmap_interface pmap_interface; ///< Process map interface type

      static constexpr bool consumable = true;
      static constexpr unsigned int leaves = 1;
    };

    /// Tensor expression engine

    /// \tparam Derived The derived class type
    template <typename Derived>
    class BlkTsrEngineBase : public LeafEngine<Derived> {
    public:
      // Class hierarchy typedefs
      typedef BlkTsrEngineBase<Derived> BlkTsrEngineBase_; ///< This class type
      typedef LeafEngine<Derived> LeafEngine_; ///< Leaf base class type
      typedef typename LeafEngine_::ExprEngine_ ExprEngine_; ///< Expression engine base class

      // Argument typedefs
      typedef typename EngineTrait<Derived>::array_type array_type; ///< The left-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<Derived>::value_type value_type; ///< Tensor value type
      typedef typename EngineTrait<Derived>::op_type op_type; ///< Tile operation type
      typedef typename EngineTrait<Derived>::policy policy; ///< The result policy type
      typedef typename EngineTrait<Derived>::dist_eval_type dist_eval_type; ///< This expression's distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<Derived>::size_type size_type; ///< Size type
      typedef typename EngineTrait<Derived>::trange_type trange_type; ///< Tiled range type type
      typedef typename EngineTrait<Derived>::shape_type shape_type; ///< Tensor shape type
      typedef typename EngineTrait<Derived>::pmap_interface pmap_interface; ///< Process map interface type

    protected:

      // Import base class variables to this scope
      using ExprEngine_::world_;
      using ExprEngine_::vars_;
      using ExprEngine_::perm_;
      using ExprEngine_::trange_;
      using ExprEngine_::shape_;
      using ExprEngine_::pmap_;
      using ExprEngine_::permute_tiles_;
      using LeafEngine_::array_;

      std::vector<std::size_t> lower_bound_; ///< Lower bound of the tile block
      std::vector<std::size_t> upper_bound_; ///< Upper bound of the tile block

    public:

      BlkTsrEngineBase(const BlkTsrExpr<array_type>& expr) :
        LeafEngine_(expr),
        lower_bound_(expr.lower_bound()), upper_bound_(expr.upper_bound())
      { }

      BlkTsrEngineBase(const BlkTsrExpr<const array_type>& expr) :
        LeafEngine_(expr),
        lower_bound_(expr.lower_bound()), upper_bound_(expr.upper_bound())
      { }

      BlkTsrEngineBase(const ScalBlkTsrExpr<array_type>& expr) :
        LeafEngine_(expr),
        lower_bound_(expr.lower_bound()), upper_bound_(expr.upper_bound())
      { }

      BlkTsrEngineBase(const ScalBlkTsrExpr<const array_type>& expr) :
        LeafEngine_(expr),
        lower_bound_(expr.lower_bound()), upper_bound_(expr.upper_bound())
      { }


      /// Non-permuting tiled range factory function

      /// \return The result tiled range
      trange_type make_trange() const {
        const unsigned int rank = array_.trange().tiles().rank();

        std::vector<TiledRange1> trange_data;
        trange_data.reserve(rank);
        std::vector<std::size_t> trange1_data;

        // Get temporary data pointers
        const auto* restrict const trange = array_.trange().data().data();
        const auto* restrict const lower = lower_bound_.data();
        const auto* restrict const upper = upper_bound_.data();

        for(unsigned int d = 0u; d < rank; ++d) {
          // Copy the tiling for the block
          const auto lower_d = lower[d];
          const auto upper_d = upper[d];

          // Copy and shift the tiling for the block
          auto i = lower_d;
          const auto base_d = trange[d].tile(i).first;
          trange1_data.emplace_back(0ul);
          for(; i < upper_d; ++i)
            trange1_data.emplace_back(trange[d].tile(i).second - base_d);

          // Add the trange1 to the tiled range data
          trange_data.emplace_back(trange1_data.begin(), trange1_data.end());
          trange1_data.resize(0ul);
        }

        return TiledRange(trange_data.begin(), trange_data.end());
      }


      /// Permuting tiled range factory function

      /// \return The result tiled range
      trange_type make_trange(const Permutation& perm) const {
        const unsigned int rank = array_.trange().tiles().rank();

        std::vector<TiledRange1> trange_data;
        trange_data.reserve(rank);
        std::vector<std::size_t> trange1_data;

        // Get temporary data pointers
        const auto* restrict const trange = array_.trange().data().data();
        const auto* restrict const lower = lower_bound_.data();
        const auto* restrict const upper = upper_bound_.data();

        // Construct the inverse permutation
        const Permutation inv_perm = -perm;
        for(unsigned int d = 0u; d < rank; ++d) {
          const auto inv_perm_d = inv_perm[d];

          // Copy the tiling for the block
          const auto lower_i = lower[inv_perm_d];
          const auto upper_i = upper[inv_perm_d];

          // Copy, shift, and permute the tiling of the block
          auto i = lower_i;
          const auto base_d = trange[inv_perm_d].tile(i).first;
          trange1_data.emplace_back(0ul);
          for(; i < upper_i; ++i)
            trange1_data.emplace_back(trange[inv_perm_d].tile(i).second - base_d);

          // Add the trange1 to the tiled range data
          trange_data.emplace_back(trange1_data.begin(), trange1_data.end());
          trange1_data.resize(0ul);
        }

        return TiledRange(trange_data.begin(), trange_data.end());
      }


      void init_distribution(World* world,
          const std::shared_ptr<pmap_interface>& pmap)
      {
        ExprEngine_::init_distribution(world,
            (pmap ? pmap : policy::default_pmap(*world, trange_.tiles().volume())));
      }

      /// Construct the distributed evaluator for array
      dist_eval_type make_dist_eval() const {
        // Define the distributed evaluator implementation type
        typedef TiledArray::detail::ArrayEvalImpl<array_type, op_type, policy> impl_type;

        /// Create the pimpl for the distributed evaluator
        std::shared_ptr<impl_type> pimpl(
            new impl_type(array_, *world_, trange_, shape_, pmap_, perm_,
            ExprEngine_::make_op(), lower_bound_, upper_bound_));

        return dist_eval_type(pimpl);
      }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      std::string make_tag() const {
        std::stringstream ss;
        ss << "[Block ";
        TiledArray::detail::print_array(ss, lower_bound_);
        ss << " - ";
        TiledArray::detail::print_array(ss, upper_bound_);
        ss << "] ";
        return ss.str();
      }

    }; // class BlkTsrEngineBase


    /// Tensor expression engine

    /// \tparam A The array type
    template <typename A>
    class BlkTsrEngine : public BlkTsrEngineBase<BlkTsrEngine<A> > {
    public:
      // Class hierarchy typedefs
      typedef BlkTsrEngine<A> BlkTsrEngine_; ///< This class type
      typedef BlkTsrEngineBase<BlkTsrEngine_> BlkTsrEngineBase_; ///< Block tensor base class type
      typedef typename BlkTsrEngineBase_::LeafEngine_ LeafEngine_; ///< Leaf base class type
      typedef typename LeafEngine_::ExprEngine_ ExprEngine_; ///< Expression engine base class

      // Argument typedefs
      typedef typename EngineTrait<BlkTsrEngine_>::array_type array_type; ///< The left-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<BlkTsrEngine_>::value_type value_type; ///< Tensor value type
      typedef typename EngineTrait<BlkTsrEngine_>::op_type op_type; ///< Tile operation type
      typedef typename EngineTrait<BlkTsrEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<BlkTsrEngine_>::dist_eval_type dist_eval_type; ///< This expression's distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<BlkTsrEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<BlkTsrEngine_>::trange_type trange_type; ///< Tiled range type type
      typedef typename EngineTrait<BlkTsrEngine_>::shape_type shape_type; ///< Tensor shape type
      typedef typename EngineTrait<BlkTsrEngine_>::pmap_interface pmap_interface; ///< Process map interface type

    protected:

      // Import base class variables to this scope
      using ExprEngine_::world_;
      using ExprEngine_::vars_;
      using ExprEngine_::perm_;
      using ExprEngine_::trange_;
      using ExprEngine_::shape_;
      using ExprEngine_::pmap_;
      using ExprEngine_::permute_tiles_;
      using LeafEngine_::array_;
      using BlkTsrEngineBase_::lower_bound_;
      using BlkTsrEngineBase_::upper_bound_;

    public:

      BlkTsrEngine(const BlkTsrExpr<array_type>& expr) :
        BlkTsrEngineBase_(expr)
      { }

      BlkTsrEngine(const BlkTsrExpr<const array_type>& expr) :
        BlkTsrEngineBase_(expr)
      { }


      /// Non-permuting shape factory function

      /// \return The result shape
      shape_type make_shape() {
        return array_.get_shape().block(lower_bound_, upper_bound_);
      }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      shape_type
      make_shape(const Permutation& perm) {
        return array_.get_shape().block(lower_bound_, upper_bound_, perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const {
        const unsigned int rank = trange_.tiles().rank();

        // Construct and allocate memory for the shift range
        std::vector<long> range_shift;
        range_shift.reserve(rank);

        // Get temporary data pointers
        const auto* restrict const trange = array_.trange().data().data();
        const auto* restrict const lower = lower_bound_.data();

        // Construct the inverse permutation
        for(unsigned int d = 0u; d < rank; ++d) {
          const auto lower_d = lower[d];
          const auto base_d = trange[d].tile(lower_d).first;
          range_shift.emplace_back(-base_d);
        }

        return op_type(range_shift);
      }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const {
        const unsigned int rank = trange_.tiles().rank();

        // Construct and allocate memory for the shift range
        std::vector<long> range_shift;
        range_shift.reserve(rank);

        // Get temporary data pointers
        const auto* restrict const trange = array_.trange().data().data();
        const auto* restrict const lower = lower_bound_.data();

        // Construct the inverse permutation
        const Permutation inv_perm = -perm;
        for(unsigned int d = 0u; d < rank; ++d) {
          const auto inv_perm_d = inv_perm[d];
          const auto lower_d = lower[inv_perm_d];
          const auto base_d = trange[inv_perm_d].tile(lower_d).first;
          range_shift.emplace_back(-base_d);
        }

        return op_type(range_shift, perm);
      }

    }; // class BlkTsrEngine



    /// Scaled tensor block expression engine

    /// \tparam A The array type
    template <typename A>
    class ScalBlkTsrEngine : public BlkTsrEngineBase<ScalBlkTsrEngine<A> > {
    public:
      // Class hierarchy typedefs
      typedef ScalBlkTsrEngine<A> ScalBlkTsrEngine_; ///< This class type
      typedef BlkTsrEngineBase<ScalBlkTsrEngine_> BlkTsrEngineBase_; ///< Block tensor base class type
      typedef typename BlkTsrEngineBase_::LeafEngine_ LeafEngine_; ///< Leaf base class type
      typedef typename LeafEngine_::ExprEngine_ ExprEngine_; ///< Expression engine base class

      // Argument typedefs
      typedef typename EngineTrait<ScalBlkTsrEngine_>::array_type array_type; ///< The left-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<ScalBlkTsrEngine_>::value_type value_type; ///< Tensor value type
      typedef typename EngineTrait<ScalBlkTsrEngine_>::scalar_type scalar_type; ///< Tile scalar type
      typedef typename EngineTrait<ScalBlkTsrEngine_>::op_type op_type; ///< Tile operation type
      typedef typename EngineTrait<ScalBlkTsrEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<ScalBlkTsrEngine_>::dist_eval_type dist_eval_type; ///< This expression's distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<ScalBlkTsrEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<ScalBlkTsrEngine_>::trange_type trange_type; ///< Tiled range type type
      typedef typename EngineTrait<ScalBlkTsrEngine_>::shape_type shape_type; ///< Tensor shape type
      typedef typename EngineTrait<ScalBlkTsrEngine_>::pmap_interface pmap_interface; ///< Process map interface type

    protected:

      // Import base class variables to this scope
      using ExprEngine_::world_;
      using ExprEngine_::vars_;
      using ExprEngine_::perm_;
      using ExprEngine_::trange_;
      using ExprEngine_::shape_;
      using ExprEngine_::pmap_;
      using ExprEngine_::permute_tiles_;
      using LeafEngine_::array_;
      using BlkTsrEngineBase_::lower_bound_;
      using BlkTsrEngineBase_::upper_bound_;

      scalar_type factor_;

    public:

      ScalBlkTsrEngine(const ScalBlkTsrExpr<array_type>& expr) :
        BlkTsrEngineBase_(expr), factor_(expr.factor())
      { }

      ScalBlkTsrEngine(const ScalBlkTsrExpr<const array_type>& expr) :
        BlkTsrEngineBase_(expr), factor_(expr.factor())
      { }


      /// Non-permuting shape factory function

      /// \return The result shape
      shape_type make_shape() {
        return array_.get_shape().block(lower_bound_, upper_bound_, factor_);
      }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      shape_type
      make_shape(const Permutation& perm) {
        return array_.get_shape().block(lower_bound_, upper_bound_, factor_, perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const {
        const unsigned int rank = trange_.tiles().rank();

        // Construct and allocate memory for the shift range
        std::vector<long> range_shift;
        range_shift.reserve(rank);

        // Get temporary data pointers
        const auto* restrict const trange = array_.trange().data().data();
        const auto* restrict const lower = lower_bound_.data();

        // Construct the inverse permutation
        for(unsigned int d = 0u; d < rank; ++d) {
          const auto lower_d = lower[d];
          const auto base_d = trange[d].tile(lower_d).first;
          range_shift.emplace_back(-base_d);
        }

        return op_type(range_shift, factor_);
      }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const {
        const unsigned int rank = trange_.tiles().rank();

        // Construct and allocate memory for the shift range
        std::vector<long> range_shift;
        range_shift.reserve(rank);

        // Get temporary data pointers
        const auto* restrict const trange = array_.trange().data().data();
        const auto* restrict const lower = lower_bound_.data();

        // Construct the inverse permutation
        const Permutation inv_perm = -perm;
        for(unsigned int d = 0u; d < rank; ++d) {
          const auto inv_perm_d = inv_perm[d];
          const auto lower_d = lower[inv_perm_d];
          const auto base_d = trange[inv_perm_d].tile(lower_d).first;
          range_shift.emplace_back(-base_d);
        }

        return op_type(range_shift, perm, factor_);
      }


      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      std::string make_tag() const {
        std::stringstream ss;
        ss << "[" << factor_ << "] ";
        return BlkTsrEngineBase_::make_tag() + ss.str();
      }

    }; // class ScalBlkTsrEngine


  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BLK_TSR_ENGINE_H__INCLUDED
