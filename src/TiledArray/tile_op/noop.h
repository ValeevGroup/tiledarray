/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  noop.h
 *  June 27, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_NOOP_H__INCLUDED
#define TILEDARRAY_TILE_OP_NOOP_H__INCLUDED

#include <TiledArray/tile_op/permute.h>

namespace TiledArray {
  namespace math {

    /// Tile no operation (noop)

    /// This no operation will return the original or apply a permutation to the
    /// result tensor. If no permutation is given or the permutation is null,
    /// then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Arg The argument type
    /// \tparam Consumable Flag that is \c true when Arg is consumable
    template <typename Result, typename Arg, bool Consumable>
    class Noop {
    public:
      typedef Noop<Result, Arg, Consumable> Noop_; ///< This object type
      typedef typename madness::if_c<Consumable, Arg&, const Arg&>::type argument_type; ///< The argument type
      typedef Result result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation


      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute(const Arg& arg) const {
        return perm_ ^ arg;
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool C>
      static typename madness::disable_if_c<C && std::is_same<Result, Arg>::value,
          result_type>::type
      no_permute(const Arg& arg) { return arg.clone(); }

      template <bool C>
      static typename madness::enable_if_c<C && std::is_same<Result, Arg>::value,
          result_type>::type
      no_permute(Arg& arg) { return arg; }

    public:
      /// Default constructor

      /// Construct a no operation that does not permute the result tile
      Noop() : perm_() { }

      /// Permute constructor

      /// Construct a no operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Noop(const Permutation& perm) : perm_(perm) { }

      /// Copy constructor

      /// \param other The no operation object to be copied
      Noop(const Noop_& other) : perm_(other.perm_) { }

      /// Copy assignment

      /// \param other The no operation object to be copied
      /// \return A reference to this object
      Noop_& operator=(const Noop_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// No operation or possibly permute

      /// \param arg The argument
      /// \return The sum and permutation of \c arg
      result_type operator()(argument_type arg) const {
        if(perm_.dim() > 1)
          return permute(arg);

        return no_permute<Consumable>(arg);
      }

      /// No operation or possibly permute

      /// Permute if the operation has a valid permutation, otherwise clone
      /// \c arg when \c consume is false or (shallow) copy when consume is
      /// \c true .
      /// \param arg The argument
      /// \param consume Indicates the consumable behavior
      /// \return The sum and permutation of \c arg
      result_type operator()(Arg& arg, const bool consume) const {
        if(perm_.dim() > 1)
          return permute(arg);

        if(consume)
          return no_permute<true>(arg);

        return no_permute<false>(arg);
      }

      /// No operation or possibly permute

      /// Permute if the operation has a valid permutation, otherwise clone
      /// \c arg . Never consumes \c arg since it is const.
      /// \param arg The argument
      /// \param consume Indicates the consumable behavior
      /// \return The sum and permutation of \c arg
      /// \throw TiledArray::Exception If \c consume is \c true .
      result_type operator()(const Arg& arg, const bool consume) const {
        TA_ASSERT(! consume); // consume flag cannot be respected due to arg constness.
        if(perm_.dim() > 1)
          return permute(arg);

        return no_permute<false>(arg);
      }
    }; // class Noop

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_NOOP_H__INCLUDED
