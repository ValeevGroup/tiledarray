/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_ANNOTATED_TENSOR_H__INCLUDED
#define TILEDARRAY_ANNOTATED_TENSOR_H__INCLUDED

#include <TiledArray/tensor_expression.h>
#include <TiledArray/tensor.h>
#include <TiledArray/functional.h>

namespace TiledArray {
  namespace expressions {
    namespace detail {

      /// Wraps an \c Array object as a tensor expression

      /// This object converts an \c Array obect into a tensor expression and
      /// adds annotation.
      /// \tparam A The \c Array type
      /// \tparam Op The Unary transform operator type.
      template <typename A>
      class AnnotatedTensorImpl : public TensorExpressionImpl<Tensor<typename A::element_type> > {
      public:
        typedef TensorExpressionImpl<typename A::value_type> TensorExpressionImpl_; ///< The base class type
        typedef typename TensorExpressionImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
        typedef AnnotatedTensorImpl<A> AnnotatedTensorImpl_; ///< This object type
        typedef A array_type; ///< The array type

        typedef typename TensorImpl_::size_type size_type; ///< size type
        typedef typename TensorImpl_::pmap_interface pmap_interface; ///< The process map interface type
        typedef typename TensorImpl_::trange_type trange_type; ///< Tiled range type
        typedef typename TensorImpl_::range_type range_type; ///< Tile range type
        typedef typename TensorImpl_::shape_type shape_type; ///< Tile shape type
        typedef typename TensorImpl_::value_type value_type; ///< The result value type
        typedef typename TensorImpl_::numeric_type numeric_type; ///< The numeric data supporting value_type
        typedef typename TensorImpl_::storage_type::const_iterator const_iterator; ///< Tensor const iterator
        typedef typename TensorImpl_::storage_type::future const_reference; /// The storage type for this object

      public:

        /// Constructor

        /// \param left The left argument
        /// \param right The right argument
        /// \param op The element transform operation
        AnnotatedTensorImpl(const array_type& array, const VariableList& vars) :
            TensorExpressionImpl_(array.get_world(), vars, array.trange(), (array.is_dense() ? 0 : array.size())),
            array_(const_cast<array_type&>(array))
        { }

        /// Virtual destructor
        virtual ~AnnotatedTensorImpl() { }

        /// Array accessor

        /// \return A reference to the array object
				array_type& array() { return array_; }

        /// Const array accessor

        /// \return A const reference to the array object
        const array_type& array() const { return array_; }

        /// Assign a tensor expression to this object
        virtual void assign(std::shared_ptr<TensorExpressionImpl_>&, TensorExpression<value_type>& other) {
          other.eval(TensorExpressionImpl_::vars(), array_.get_pmap()->clone()).get();
          array_ = other.template convert_to_array<array_type>();
        }

      private:

        /// Task function that is used to convert an input tile to value_type and store it

        /// \param i The tile index
        /// \param value The tile from the array
        void convert_and_set_tile(const size_type i, const typename array_type::value_type& value) {
          TensorExpressionImpl_::set(i, value_type(value));
        }

        /// Task function that is used to scale an input tile to value_type and store it

        /// \param i The tile index
        /// \param value The tile from the array
        void scale_and_set_tile(const size_type i, const value_type& value) {
          TensorExpressionImpl_::set(i, value_type(value.range(), value.begin(),
              std::bind2nd(TiledArray::detail::multiplies<typename value_type::value_type,
                                                          numeric_type,
                                                          typename value_type::value_type
                                                         >(),
              TensorExpressionImpl_::scale())));
        }

        /// Task function that is used to convert an input tile to value_type, scale it, and store it

        /// \param i The tile index
        /// \param value The tile from the array
        void convert_scale_and_set_tile(const size_type i, const typename array_type::value_type& value) {
          TensorExpressionImpl_::set(i, value_type(value.range(), value.begin(),
              std::bind2nd(TiledArray::detail::multiplies<typename value_type::value_type,
                                                          numeric_type,
                                                          typename value_type::value_type
                                                         >(),
              TensorExpressionImpl_::scale())));
        }

        /// Set a tile

        /// This function will store a shallow copy of a given an input tile
        /// from the array.
        /// \tparam Value The tile type from the array
        /// \param i The tile index
        /// \param value The tile from the array
        template <typename Value>
        typename madness::enable_if<std::is_same<Value, value_type> >::type
        set_tile(size_type i, const madness::Future<Value>& value) {
          TensorExpressionImpl_::set(i, value);
        }

        /// Convert and set a tile

        /// This function will spawn a task, given an input tile from the array,
        /// that will convert it to a \c value_type tile. The new tile will then
        /// be stored in this tensor expression.
        /// \tparam Value The tile type from the array
        /// \param i The tile index
        /// \param value The tile from the array
        template <typename Value>
        typename madness::disable_if<std::is_same<Value, value_type> >::type
        set_tile(size_type i, const madness::Future<Value>& value) {
          TensorExpressionImpl_::get_world().taskq.add(this,
              & AnnotatedTensorImpl_::convert_and_set_tile, i, value);
        }

        /// Scale, and set a tile

        /// This function will spawn a task, given an input tile from the array,
        /// that will create a scaled copy of the array tile. The new tile will
        /// then be stored in this tensor expression.
        /// \tparam Value The tile type from the array
        /// \param i The tile index
        /// \param value The tile from the array
        template <typename Value>
        typename madness::enable_if<std::is_same<Value, value_type> >::type
        scale_set_tile(size_type i, const madness::Future<Value>& value) {
          TensorExpressionImpl_::get_world().taskq.add(this,
              & AnnotatedTensorImpl_::scale_and_set_tile, i, value);
        }

        /// Convert, scale, and set a tile

        /// This function will spawn a task, given an input tile from the array,
        /// that will convert it to a \c value_type tile then scale the new
        /// tile. The new tile will then be stored in this tensor expression.
        /// \tparam Value The tile type from the array
        /// \param i The tile index
        /// \param value The tile from the array
        template <typename Value>
        typename madness::disable_if<std::is_same<Value, value_type> >::type
        scale_set_tile(size_type i, const madness::Future<Value>& value) {
          TensorExpressionImpl_::get_world().taskq.add(this,
              & AnnotatedTensorImpl_::convert_scale_and_set_tile, i, value);
        }

        /// Check that a integral value is approximately equal to 1.

        /// \tparam T The integral type
        /// \param t The value to be checked
        /// \return \c true if t is equal to 1, otherwise false
        template <typename T>
        typename madness::enable_if<std::is_integral<T>, bool>::type
        is_one(const T t) {
          return t == std::integral_constant<T, 1>::value;
        }

        /// Check that a floating point value is approximately equal to 1.

        /// Check that \c is approximately equal to 1 +/- 10^-13.
        /// \tparam T The floating point type
        /// \param t The value to be checked
        /// \return \c true if t is equal to 1, otherwise false
        bool is_one(const double t) {
          return (t <= 1.0000000000001) && (t >= 0.9999999999999);
        }

        /// Check that a floating point value is approximately equal to 1.

        /// Check that \c is approximately equal to 1 +/- 10^-5.
        /// \tparam T The floating point type
        /// \param t The value to be checked
        /// \return \c true if t is equal to 1, otherwise false
        bool is_one(const float t) { return (t <= 1.00001) && (t >= 0.99999); }

        /// Function for evaluating this tensor's tiles

        /// This function is run inside a task, and will run after \c eval_children
        /// has completed. It should spawn additional tasks that evaluate the
        /// Individual result tiles.
        virtual void eval_tiles() {
          typedef typename TiledArray::detail::scalar_type<typename value_type::value_type>::type
              numeric_type; ///< the numeric type that supports Tile
          static const numeric_type one(1);

          // Make sure all local tiles are present.
          const typename pmap_interface::const_iterator end =
              TensorExpressionImpl_::pmap()->end();
          typename pmap_interface::const_iterator it =
              TensorExpressionImpl_::pmap()->begin();

            if(is_one(TensorExpressionImpl_::scale())) {
              if(array_.is_dense()) {
                for(; it != end; ++it)
                  set_tile(*it, array_.find(*it));
              } else {
                for(; it != end; ++it)
                  if(! array_.is_zero(*it))
                    set_tile(*it, array_.find(*it));
              }
            } else {
              if(array_.is_dense()) {
                for(; it != end; ++it)
                  scale_set_tile(*it, array_.find(*it));
              } else {
                for(; it != end; ++it)
                  if(! array_.is_zero(*it))
                    scale_set_tile(*it, array_.find(*it));
              }
            }
        }

        /// Function for evaluating child tensors

        /// This function should return true when the child

        /// This function should evaluate all child tensors.
        /// \param vars The variable list for this tensor (may be different from
        /// the variable list used to initialize this tensor).
        /// \param pmap The process map for this tensor
        virtual madness::Future<bool> eval_children(const expressions::VariableList&,
            const std::shared_ptr<pmap_interface>&)
        { return array_.eval(); }

        /// Construct the shape object

        /// \param shape The existing shape object
        virtual void make_shape(shape_type& shape) const {
          TA_ASSERT(shape.size() == array_.size());
          shape = array_.get_shape();
        }

        array_type& array_; ///< The referenced array
      }; // class PermuteTiledTensor

    } // namespace detail


    template <typename T, unsigned int DIM, typename Tile>
    inline TensorExpression<Tensor<T> >
    make_annotated_tensor(const Array<T, DIM, Tile>& array, const VariableList& vars) {
      typedef detail::AnnotatedTensorImpl<Array<T, DIM, Tile> > impl_type;
      std::shared_ptr<typename impl_type::TensorExpressionImpl_> pimpl(
          new impl_type(array, vars),
          madness::make_deferred_deleter<impl_type>(array.get_world()));
      return TensorExpression<Tensor<T> >(pimpl);
    }

    template <typename T, unsigned int DIM, typename Tile>
    inline TensorExpression<Tensor<T> >
    make_annotated_tensor(const Array<T, DIM, Tile>& array, const std::string& vars) {
      typedef detail::AnnotatedTensorImpl<Array<T, DIM, Tile> > impl_type;
      std::shared_ptr<typename impl_type::TensorExpressionImpl_> pimpl(
          new impl_type(array, VariableList(vars)),
          madness::make_deferred_deleter<impl_type>(array.get_world()));
      return TensorExpression<Tensor<T> >(pimpl);
    }

  } // namespace expressions
} //namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_TENSOR_H__INCLUDED
