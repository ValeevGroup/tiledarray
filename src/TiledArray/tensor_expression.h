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
 */

#ifndef TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED
#define TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED

#include <TiledArray/tensor_impl.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/pmap/blocked_pmap.h>

namespace TiledArray {

  // Forward declaration
  template <typename, unsigned int, typename> class Array;

  namespace expressions {

    template <typename> class TensorExpression;

    namespace detail {

      /// Tensor expression implementation object

      /// This class is used as the base class for other tensor expression
      /// implementation classes. It has several pure virtual function that are
      /// used by derived classes to evaluate the expression. This class also
      /// handles permutation of result tiles if necessary.
      /// \tparam Tile The result tile type for the expression.
      template <typename Tile>
      class TensorExpressionImpl : public ::TiledArray::detail::TensorImpl<Tile> {
      public:
        typedef TensorExpressionImpl<Tile> TensorExpressionImpl_; ///< This object type
        typedef TiledArray::detail::TensorImpl<Tile> TensorImpl_; ///< Tensor implementation base class

        typedef typename TensorImpl_::size_type size_type; ///< Size type
        typedef typename TensorImpl_::trange_type trange_type; ///< Tiled range type for this object
        typedef typename TensorImpl_::range_type range_type; ///< Range type this tensor
        typedef typename TensorImpl_::shape_type shape_type; ///< Shape type
        typedef typename TensorImpl_::pmap_interface pmap_interface; ///< process map interface type
        typedef typename TensorImpl_::value_type value_type; ///< Tile type
        typedef typename TensorImpl_::numeric_type numeric_type;  ///< the numeric type that supports Tile

      private:
        VariableList vars_; ///< The tensor expression variable list
        Permutation perm_; ///< The permutation to be applied to this tensor
        const typename TensorImpl_::trange_type trange_; ///< The original tiled range for this tensor
        volatile bool evaluated_; ///< A flag to indicate that the tensor has been evaluated.
                                  ///< This is just here as a sanity check to make sure evaluate is only run once.
                                  ///< It is NOT thread safe.
        numeric_type scale_; ///< The scale factor for this expression

        /// Task function for permuting result tensor

        /// \param value The unpermuted result tile
        /// \return The permuted result tile
        void permute_and_set_with_value(const size_type index, const value_type& value) {
          // Create tensor to hold the result
          value_type result(perm_ ^ value.range());

          // Construct the inverse permuted weight and size for this tensor
          std::vector<std::size_t> ip_weight = (-perm_) ^ result.range().weight();
          const typename value_type::range_type::index& start = value.range().start();

          // Coordinated iterator for the value range
          typename value_type::range_type::const_iterator value_range_it =
              value.range().begin();

          // permute the data
          const size_type end = result.size();
          for(size_type value_it = 0ul; value_it != end; ++value_it, ++value_range_it)
            result[TiledArray::detail::calc_ordinal(*value_range_it, ip_weight, start)] = value[value_it];

          // Store the permuted tensor
          TensorImpl_::set(TensorImpl_::range().ord(perm_ ^ trange_.tiles().idx(index)),
              result);
        }

        /// Permute and set tile \c i with \c value

        /// If the \c value has been set, then the tensor is permuted and set
        /// immediately. Otherwise a task is spawned that will permute and set it.
        /// \param i The unpermuted index of the tile
        /// \param value The future that holds the unpermuted result tile
        void permute_and_set(size_type i, const value_type& value) {
          permute_and_set_with_value(i, value);
        }

        /// Permute and set tile \c i with \c value

        /// If the \c value has been set, then the tensor is permuted and set
        /// immediately. Otherwise a task is spawn that will permute and set it.
        /// \param i The unpermuted index of the tile
        /// \param value The future that holds the unpermuted result tile
        void permute_and_set(size_type i, const madness::Future<value_type>& value) {
          if(value.probe())
            permute_and_set_with_value(i, value.get());
          else
            TensorImpl_::get_world().taskq.add(*this,
                & TensorExpressionImpl_::permute_and_set_with_value, i, value);
        }

      protected:

        /// Map an index value from the unpermuted index space to the permuted index space

        /// \param i The index in the unpermuted index space
        /// \return The corresponding index in the permuted index space
        size_type perm_index(size_type i) const {
          TA_ASSERT(evaluated_);
          if(perm_.dim())
            return TensorImpl_::range().ord(perm_ ^ trange_.tiles().idx(i));
          else
            return i;
        }

      public:
        /// Constructor

        /// \param world The world where the tensor lives
        /// \param tr The tiled range object
        /// \param shape The tensor shape bitset [ Default = 0 size bitset ]
        /// \note If the shape bitset is zero size, then the tensor is considered
        /// to be dense.
        TensorExpressionImpl(madness::World& world, const VariableList& vars,
            const trange_type& trange, const shape_type& shape = shape_type(0ul)) :
          TensorImpl_(world, trange, shape),
          vars_(vars),
          trange_(trange),
          perm_(),
          evaluated_(false),
          scale_(1)
        { }

        virtual ~TensorExpressionImpl() { }

        /// Variable list accessor

        /// \return The expression variable list
        const VariableList& vars() const { return vars_; }


        /// Variable list accessor

        /// \return The expression variable list
        void vars(const VariableList& vars) { vars_ = vars; }

        /// Set tensor value

        /// This will store \c value at ordinal index \c i . The tile will be
        /// permuted if necessary. Typically this function should be called by
        /// \c eval_tiles() or there in.
        /// \tparam The value type, either \c value_type or \c madness::Future<value_type>
        /// \param i The index where value will be stored.
        /// \param value The value or future value to be stored at index \c i
        /// \note The index \c i and \c value will be permuted by this function
        /// before storing the value.
        template <typename Value>
        void set(size_type i, const Value& value) {
          TA_ASSERT(evaluated_);

          if(perm_.dim())
            permute_and_set(i, value);
          else
            TensorImpl_::set(i, value);
        }

        typename value_type::range_type make_tile_range(size_type i) const {
          return trange_.make_tile_range(i);
        }

        /// Modify the expression scale factor

        /// scale = scale * value
        /// \param value The new scale factor
        void scale(const numeric_type& value) { scale_ *= value; }

        /// Modify the expression scale factor

        /// scale = scale * value
        /// \param value The new scale factor
        void set_scale(const numeric_type& value) { scale_ = value; }

        /// Get the expression scale factor

        /// \return The current scale factor
        numeric_type scale() const { return scale_; }


        virtual void assign(std::shared_ptr<TensorExpressionImpl_>& pimpl, TensorExpression<Tile>& other) {
          pimpl = other.pimpl_;
          other.pimpl_.reset();
        }

      private:

        /// Function for evaluating this tensor's tiles

        /// This function is run inside a task, and will run after \c eval_children
        /// has completed. It should spawn additional tasks that evaluate the
        /// individual result tiles.
        virtual void eval_tiles() = 0;

        /// Function for evaluating child tensors

        /// This function should return true when the child

        /// This function should evaluate all child tensors.
        /// \param vars The variable list for this tensor (may be different from
        /// the variable list used to initialize this tensor).
        /// \param pmap The process map for this tensor
        virtual madness::Future<bool> eval_children(const expressions::VariableList& vars,
            const std::shared_ptr<pmap_interface>& pmap) = 0;

        /// Construct the shape object

        /// This function is used by derived classes to create a shape object. It
        /// is run inside a task with the proper dependencies to ensure data
        /// consistency. This function is only called when the tensor is not dense.
        /// \param shape The existing shape object
        virtual void make_shape(shape_type& shape) const = 0;

        /// Set the range, shape, and variable list of this tensor

        /// This task function is used to evaluate this tensor expression. If the
        /// variable list used to initialize the object does not match the eval
        /// variable list, then the data is also permuted.
        /// \param vars The final variable list order (must be a permutation of
        /// the current variable list)
        /// \return \c true when the tensor structure has been permuted
        bool internal_eval(const VariableList& vars, bool) {
          // Evaluate the shape of this tensor

          // Permute structure if the current variable list does not match vars
          if(vars != vars_) {
            // Get the permutation to go from the current variable list to vars,
            // such that: vars = perm ^ vars_
            perm_ = vars.permutation(vars_);

            // Store the new variable list
            vars_ = vars;

            // Permute the tiled range
            TensorImpl_::trange(perm_ ^ trange_);

            // If not dense, permute the shape
            if(! TensorImpl_::is_dense()) {
              // Construct the inverse permuted weight and size for this tensor
              std::vector<std::size_t> ip_weight =
                  (-perm_) ^ TensorImpl_::trange().tiles().weight();
              const typename range_type::index& start = trange_.tiles().start();

              // Get range iterator
              typename range_type::const_iterator range_it =
                  trange_.tiles().begin();

              // Construct temp shape
              const size_type size = TensorImpl_::size();
              TiledArray::detail::Bitset<> s0(TensorImpl_::size());
              this->make_shape(s0);

              // Set the new shape
              for(size_type i = 0ul; i < size; ++i, ++range_it)
                if(s0[i])
                  TensorImpl_::shape(TiledArray::detail::calc_ordinal(*range_it,
                      ip_weight, start), true);
            }

          } else {
            if(! TensorImpl_::is_dense())
              this->make_shape(TensorImpl_::shape());
          }

          evaluated_ = true;
          this->eval_tiles();

          return true;
        }

      public:

        /// Evaluate this tensor expression object

        /// This function will:
        /// \li Evaluate the child tensors using the \c eval_children virtual function
        /// \li Set the process map for result tiles to \c pmap
        /// \li Permute the range, shape, and variable list of this tensor if
        /// \c vars is not equal to the current variable list.
        /// \li Evaluat result tiles
        /// \param vars The result variable list for this expression
        /// \param pmap The process map for storage of result tiles
        /// \return A future to a bool that will be set once the structure of this
        /// tensor has been set to its final value. Once the future has been set
        /// it is safe to access this tensor, but not with the iterator.
        madness::Future<bool> eval(const expressions::VariableList& vars,
            const std::shared_ptr<pmap_interface>& pmap)
        {
          TA_ASSERT(! evaluated_);

          madness::Future<bool> child_eval_done = this->eval_children(vars, pmap->clone());

          // Initialize the data container process map
          TensorImpl_::pmap(pmap);

          return TensorImpl_::get_world().taskq.add(*this,
              & TensorExpressionImpl_::internal_eval, vars, child_eval_done);
        }

      }; // class TensorExpressionImpl

    } // namespace detail

    /// Tensor expression object

    /// This object holds a tensor expression. It is used to store various type
    /// of tensor expressions that depend on the pimpl used to construct the
    /// expression.
    /// \tparam Tile The expression tile type
    template <typename Tile>
    class TensorExpression {
    private:
      friend class detail::TensorExpressionImpl<Tile>;
    public:
      typedef TensorExpression<Tile> TensorExpression_;
      typedef detail::TensorExpressionImpl<Tile> impl_type;
      typedef typename impl_type::size_type size_type;
      typedef typename impl_type::range_type range_type;
      typedef typename impl_type::shape_type shape_type;
      typedef typename impl_type::pmap_interface pmap_interface;
      typedef typename impl_type::trange_type trange_type;
      typedef typename impl_type::value_type value_type;
      typedef typename impl_type::numeric_type numeric_type;
      typedef typename impl_type::const_reference const_reference;
      typedef typename impl_type::const_iterator const_iterator;

      /// Constructor

      /// \param pimpl A pointer to the expression implementation object
      TensorExpression(const std::shared_ptr<impl_type>& pimpl) : pimpl_(pimpl) { }

      /// Copy constructor

      /// Create a shallow copy of \c other .
      /// \param other The object to be copied.
      TensorExpression(const TensorExpression_& other) :
          pimpl_(other.pimpl_)
      { }

      /// Assignment operator

      /// Create a shallow copy of \c other
      /// \param other The object to be copied
      /// \return A reference to this object
      TensorExpression_& operator=(const TensorExpression_& other) {
        pimpl_->assign(pimpl_, const_cast<TensorExpression_&>(other));
        return *this;
      }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(pimpl_);
        pimpl_->eval_to(dest);
      }

      /// Evaluate this tensor object with the given result variable list

      /// \c v is the dimension ordering that the parent expression expects.
      /// The returned future will be evaluated once the tensor has been evaluated.
      /// \param v The expected data layout of this tensor.
      /// \return A Future bool that will be assigned once this tensor has been
      /// evaluated.
      madness::Future<bool> eval(const VariableList& v, const std::shared_ptr<pmap_interface>& pmap) {
        TA_ASSERT(pimpl_);
        return pimpl_->eval(v, pmap);
      }

      /// Type conversion to an \c Array object

      /// \tparam DIM The array dimension
      template <unsigned int DIM>
      operator Array<typename value_type::value_type, DIM, value_type>() const {
        TA_ASSERT(pimpl_);
        TA_ASSERT(pimpl_->range().dim() == DIM);
        typedef Array<typename value_type::value_type, DIM, value_type> array_type;

        // Evaluate this tensor and wait
        pimpl_->eval(pimpl_->vars(),
            std::shared_ptr<TiledArray::Pmap<size_type> >(
            new TiledArray::detail::BlockedPmap(pimpl_->get_world(),
            pimpl_->size()))).get();

        return const_cast<TensorExpression*>(this)->convert_to_array<Array<typename value_type::value_type, DIM, value_type> >();
      }

      template <typename A>
      A convert_to_array() {
        if(pimpl_->is_dense()) {
          A array(pimpl_->get_world(), pimpl_->trange());

          typename pmap_interface::const_iterator it = pimpl_->pmap()->begin();
          const typename pmap_interface::const_iterator end = pimpl_->pmap()->end();
          for(; it != end; ++it)
            array.set(*it, pimpl_->move(*it));

          return array;
        } else {
          A array(pimpl_->get_world(), pimpl_->trange(), pimpl_->shape());

          typename pmap_interface::const_iterator it = pimpl_->pmap()->begin();
          const typename pmap_interface::const_iterator end = pimpl_->pmap()->end();
          for(; it != end; ++it)
            if(! pimpl_->is_zero(*it))
              array.set(*it, pimpl_->move(*it));

          return array;
        }
      }


      /// Modify the expression scale factor

      /// scale = scale * factor
      /// \param value The new scale factor
      void scale(const numeric_type& factor) {
        TA_ASSERT(pimpl_);
        pimpl_->scale(factor);
      }

      /// Set the expression scale factor

      /// \param value The new scale factor
      void set_scale(const numeric_type& new_scale) {
        TA_ASSERT(pimpl_);
        pimpl_->set_scale(new_scale);
      }

      /// Get the expression scale factor

      /// \return The current scale factor
      numeric_type scale() const {
        TA_ASSERT(pimpl_);
        return pimpl_->scale();
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const {
        TA_ASSERT(pimpl_);
        return pimpl_->range();
      }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const {
        TA_ASSERT(pimpl_);
        return pimpl_->size();
      }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->owner(i);
      }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_local(i);
      }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_zero(i);
      }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      const std::shared_ptr<pmap_interface>& get_pmap() const {
        TA_ASSERT(pimpl_);
        return pimpl_->pmap();
      }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_dense();
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const shape_type get_shape() const {
        TA_ASSERT(pimpl_);
        return pimpl_->shape();
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const {
        TA_ASSERT(pimpl_);
        return pimpl_->trange();
      }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->operator[](i);
      }

      /// Tile move

      /// Tile is removed after it is set.
      /// \param i The tile index
      /// \return Tile \c i
      const_reference move(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->move(i);
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const {
        TA_ASSERT(pimpl_);
        return pimpl_->begin();
      }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const {
        TA_ASSERT(pimpl_);
        return pimpl_->end();
      }

      /// Variable annotation for the array.
      const VariableList& vars() const {
        TA_ASSERT(pimpl_);
        return pimpl_->vars();
      }

      madness::World& get_world() const {
        TA_ASSERT(pimpl_);
        return pimpl_->get_world();
      }

      /// Release tensor data

      /// Clear all tensor data from memory. This is equivalent to
      /// \c UnaryTiledTensor().swap(*this) .
      void release() { pimpl_.reset(); }

    protected:
      std::shared_ptr<impl_type> pimpl_; ///< pointer to the implementation object
    }; // class TensorExpression

    // TensorExpression type traits

    template <typename T>
    struct is_tensor_expression : public std::false_type { };

    template <typename Tile>
    struct is_tensor_expression<TensorExpression<Tile> > : public std::true_type { };

  }  // namespace expressions
}  // namespace TiledArray


#endif // TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED
