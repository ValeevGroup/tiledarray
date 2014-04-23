/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  mult_cont_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_MULT_CONT_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_MULT_CONT_ENGINE_H__INCLUDED

#include <TiledArray/expressions/cont_engine.h>

namespace TiledArray {
  namespace expressions {

    template <typename, typename> class MultContExpr;
    template <typename, typename> class ScalMultContExpr;
    template <typename, typename> class MultContEngine;
    template <typename, typename> class ScalMultContEngine;


    template <typename Left, typename Right>
    struct EngineTrait<MultContEngine<Left, Right> > :
      public BinaryEngineTrait<Left, Right, TiledArray::math::Mult>
    { };

    template <typename Left, typename Right>
    struct EngineTrait<ScalMultContEngine<Left, Right> > :
      public BinaryEngineTrait<Left, Right, TiledArray::math::ScalMult>
    { };


    /// Multiply/contract expression engine

    /// This expression engine will select the correct engine operations at
    /// runtime based on the argument variable lists.
    /// \tparam Left The left-hand expression engine type
    /// \tparam Right The right-hand expression engine type
    template <typename Left, typename Right>
    class MultContEngine :
        public ContEngine<MultContEngine<Left, Right>, MultEngine>
    {
    public:
      // Class hierarchy typedefs
      typedef MultContEngine<Left, Right> MultContEngine_; ///< This class type
      typedef ContEngine<MultContEngine_, MultEngine> ContEngine_; ///< Contraction expression engine base class
      typedef MultEngine<MultContEngine_> MultEngine_; ///< Multiply expression engine base class
      typedef BinaryEngine<MultContEngine_> BinaryEngine_; ///< Binary base class type
      typedef ExprEngine<MultContEngine_> ExprEngine_; ///< Expression engine base class type

      // Argument typedefs
      typedef typename EngineTrait<MultContEngine_>::left_type left_type; ///< The left-hand expression type
      typedef typename EngineTrait<MultContEngine_>::right_type right_type; ///< The right-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<MultContEngine_>::value_type value_type; ///< The result tile type
      typedef typename EngineTrait<MultContEngine_>::op_type op_type; ///< The tile operation type
      typedef typename EngineTrait<MultContEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<MultContEngine_>::dist_eval_type dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<MultContEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<MultContEngine_>::trange_type trange_type; ///< Tiled range type
      typedef typename EngineTrait<MultContEngine_>::shape_type shape_type; ///< Shape type
      typedef typename EngineTrait<MultContEngine_>::pmap_interface pmap_interface; ///< Process map interface type

    private:

      bool contract_; ///< Expression type flag (true == contraction, false ==
                      ///< coefficent-wise multiplication)

    public:

      /// Constructor

      /// \param L The left-hand argument expression type
      /// \param R The right-hand argument expression type
      /// \param expr The parent expression
      template <typename L, typename R>
      MultContEngine(const MultExpr<L, R>& expr) :
        ContEngine_(expr), contract_(false)
      { }

      /// Set the variable list for this expression

      /// This function will set the variable list for this expression and its
      /// children such that the number of permutations is minimized. The final
      /// variable list may not be set to target, which indicates that the
      /// result of this expression will be permuted to match \c target_vars.
      /// \param target_vars The target variable list for this expression
      void perm_vars(const VariableList& target_vars) {
        if(contract_)
          ContEngine_::perm_vars(target_vars);
        else
          MultEngine_::perm_vars(target_vars);
      }

      /// Initialize the variable list of this expression

      /// \param target_vars The target variable list for this expression
      void init_vars(const VariableList& target_vars) {
        BinaryEngine_::left_.init_vars();
        BinaryEngine_::right_.init_vars();

        if(BinaryEngine_::left_.vars().is_permutation(BinaryEngine_::right_.vars())) {
          MultEngine_::perm_vars(target_vars);
        } else {
          contract_ = true;
          ContEngine_::init_vars();
          ContEngine_::perm_vars(target_vars);
        }
      }

      /// Initialize the variable list of this expression
      void init_vars() {
        BinaryEngine_::left_.init_vars();
        BinaryEngine_::right_.init_vars();

        if(BinaryEngine_::left_.vars().is_permutation(BinaryEngine_::right_.vars())) {
          MultEngine_::vars();
        } else {
          contract_ = true;
          ContEngine_::init_vars();
        }
      }

      /// Initialize result tensor structure

      /// This function will initialize the permutation, tiled range, and shape
      /// for the result tensor.
      /// \param target_vars The target variable list for the result tensor
      void init_struct(const VariableList& target_vars) {
        if(contract_)
          ContEngine_::init_struct(target_vars);
        else
          MultEngine_::init_struct(target_vars);
      }

      /// Initialize result tensor distribution

      /// This function will initialize the world and process map for the result
      /// tensor.
      /// \param world The world were the result will be distributed
      /// \param pmap The process map for the result tensor tiles
      void init_distribution(madness::World* world, std::shared_ptr<pmap_interface> pmap) {
        if(contract_)
          ContEngine_::init_distribution(world, pmap);
        else
          MultEngine_::init_distribution(world, pmap);
      }

      /// Construct the distributed evaluator for this expression

      /// \return The distributed evaluator that will evaluate this expression
      dist_eval_type make_dist_eval() const {
        if(contract_)
          return ContEngine_::make_dist_eval();
        else
          return MultEngine_::make_dist_eval();
      }


    }; // class MultContEngine


    /// Multiply/contract expression engine

    /// This expression engine will select the correct engine operations at
    /// runtime based on the argument variable lists.
    /// \tparam Left The left-hand expression engine type
    /// \tparam Right The right-hand expression engine type
    template <typename Left, typename Right>
    class ScalMultContEngine :
        public ScalMultEngine<ScalMultContEngine<Left, Right> >,
        public ContEngine<ScalMultContEngine<Left, Right>, ScalMultEngine>
    {
    public:
      // Class hierarchy typedefs
      typedef MultContEngine<Left, Right> ScalMultContEngine_; ///< This class type
      typedef ContEngine<ScalMultContEngine_, ScalMultEngine> ContEngine_; ///< Contraction expression engine base class
      typedef ScalMultEngine<ScalMultContEngine_> ScalMultEngine_; ///< Multiply expression engine base class
      typedef BinaryEngine<ScalMultContEngine_> BinaryEngine_; ///< Binary base class type
      typedef ExprEngine<ScalMultContEngine_> ExprEngine_; ///< Expression engine base class type

      // Argument typedefs
      typedef typename EngineTrait<ScalMultContEngine_>::left_type left_type; ///< The left-hand expression type
      typedef typename EngineTrait<ScalMultContEngine_>::right_type right_type; ///< The right-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<ScalMultContEngine_>::value_type value_type; ///< The result tile type
      typedef typename EngineTrait<ScalMultContEngine_>::op_type op_type; ///< The tile operation type
      typedef typename EngineTrait<ScalMultContEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<ScalMultContEngine_>::dist_eval_type dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<ScalMultContEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<ScalMultContEngine_>::trange_type trange_type; ///< Tiled range type
      typedef typename EngineTrait<ScalMultContEngine_>::shape_type shape_type; ///< Shape type
      typedef typename EngineTrait<ScalMultContEngine_>::pmap_interface pmap_interface; ///< Process map interface type

    private:

      bool contract_; ///< Expression type flag (true == contraction, false ==
                      ///< coefficent-wise multiplication)

    public:

      /// Constructor

      /// \param L The left-hand argument expression type
      /// \param R The right-hand argument expression type
      /// \param expr The parent expression
      template <typename L, typename R>
      ScalMultContEngine(const ScalMultExpr<L, R>& expr) :
        ContEngine_(expr), contract_(false)
      { }

      void perm_vars(const VariableList& target_vars) {
        if(contract_)
          ContEngine_::perm_vars(target_vars);
        else
          ScalMultEngine_::perm_vars(target_vars);
      }

      void init_vars(const VariableList& target_vars) {
        BinaryEngine_::left().init_vars();
        BinaryEngine_::right().init_vars();

        if(BinaryEngine_::left().vars().is_permutation(BinaryEngine_::right().vars())) {
          ScalMultEngine_::perm_vars(target_vars);
        } else {
          contract_ = true;
          ContEngine_::init_vars();
          ContEngine_::perm_vars(target_vars);
        }
      }

      /// Initialize result tensor structure

      /// This function will initialize the permutation, tiled range, and shape
      /// for the result tensor.
      /// \param target_vars The target variable list for the result tensor
      void init_struct(const VariableList& target_vars) {
        if(contract_)
          ContEngine_::init_struct(target_vars);
        else
          ScalMultEngine_::init_struct(target_vars);
      }

      /// Initialize result tensor distribution

      /// This function will initialize the world and process map for the result
      /// tensor.
      /// \param world The world were the result will be distributed
      /// \param pmap The process map for the result tensor tiles
      void init_distribution(const VariableList& target_vars) {
        if(contract_)
          ContEngine_::init_distribution(target_vars);
        else
          ScalMultEngine_::init_distribution(target_vars);
      }

    }; // class ScalMultContEngine

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_MULT_CONT_ENGINE_H__INCLUDED
