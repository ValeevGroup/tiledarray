#ifndef TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED
#define TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED

#include <TiledArray/tensor_impl_base.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/permute_tensor.h>

namespace TiledArray {
  namespace expressions {

    template <typename TRange, typename Tile>
    class TensorExpressionImpl : public TiledArray::detail::TensorImplBase<TRange, Tile> {
    public:
      typedef TensorExpressionImpl<TRange, Tile> TensorExpressionImpl_; ///< This object type
      typedef TiledArray::detail::TensorImplBase<TRange, Tile> TensorImplBase_; ///< Tensor implementation base class

      typedef typename TensorImplBase_::size_type size_type; ///< Size type
      typedef typename TensorImplBase_::trange_type trange_type; ///< Tiled range type for this object
      typedef typename TensorImplBase_::range_type range_type; ///< Range type this tensor
      typedef typename TensorImplBase_::pmap_interface pmap_interface; ///< process map interface type
      typedef typename TensorImplBase_::value_type value_type; ///< Tile type

    private:
      VariableList vars_; ///< The tensor expression variable list
      Permutation perm_; ///< The permutation to be applied to this tensor
      const typename TensorImplBase_::trange_type trange_; ///< The original tiled range for this tensor
      volatile bool evaluated_; ///< A flag to indicate that the tensor has been evaluated.
                                ///< NOT thread safe.
                                ///< This is just here as a sanity check to make sure evaluate is only run once.

      /// Task function for permuting result tensor

      /// \param value The unpermuted result tile
      /// \return The permuted result tile
      value_type permute_task(const value_type& value) const {
        return expressions::make_permute_tensor(value, perm_);
      }

    protected:

      /// Map an index value from the unpermuted index space to the permuted index space

      /// \param i The index in the unpermuted index space
      /// \return The corresponding index in the permuted index space
      size_type perm_index(size_type i) const {
        TA_ASSERT(evaluated_);
        if(perm_.dim())
          return TensorImplBase_::range().ord(perm_ ^ trange_.tiles().idx(i));
        else
          return i;
      }

    public:
      /// Constructor

      /// \tparam TR The tiled range type
      /// \param world The world where the tensor lives
      /// \param tr The tiled range object
      /// \param shape The tensor shape bitset [ Default = 0 size bitset ]
      /// \note If the shape bitset is zero size, then the tensor is considered
      /// to be dense.
      template <typename TR>
      TensorExpressionImpl(madness::World& world, const VariableList& vars,
            const TiledRange<TR>& trange,
            const ::TiledArray::detail::Bitset<>& shape = ::TiledArray::detail::Bitset<>(0ul)) :
          TensorImplBase_(world, trange, shape),
          vars_(vars),
          trange_(trange),
          perm_(),
          evaluated_(false)
      { }

      virtual ~TensorExpressionImpl() { }

      /// Variable list accessor

      /// \return The expression variable list
      const VariableList& vars() const { return vars_; }

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
          TensorImplBase_::set(TensorImplBase_::range().ord(perm_ ^ trange_.tiles().idx(i)),
              TensorImplBase_::get_world().taskq.add(*this,
              & TensorExpressionImpl_::permute_task, value));
        else
          TensorImplBase_::set(i, value);
      }

      typename value_type::range_type make_tile_range(size_type i) const {
        return trange_.make_tile_range(i);
      }

    private:
      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spwan additional tasks that evaluate the
      /// individule result tiles.
      virtual void eval_tiles() = 0;

      /// Function for evaluating child tensors

      /// This function should return true when the child

      /// This function should evaluate all child tensors.
      /// \param vars The variable list for this tensor (may be different from
      /// the variable list used to initialize this tensor).
      /// \param pmap The process map for this tensor
      virtual madness::Future<bool> eval_children(const expressions::VariableList& vars,
          const std::shared_ptr<pmap_interface>& pmap) = 0;

      virtual TiledArray::detail::Bitset<> eval_shape() = 0;

      /// Permute the range, shape, and variable list of this tensor

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
          TensorImplBase_::trange(perm_ ^ trange_);

          // If not dense, permute the shape
          if(! TensorImplBase_::is_dense()) {
            // Construct the inverse permuted weight and size for this tensor
            typename range_type::size_array ip_weight = (-perm_) ^ TensorImplBase_::trange().tiles().weight();
            const typename range_type::index& start = trange_.tiles().start();

            // Get range iterator
            typename range_type::const_iterator range_it =
                trange_.tiles().begin();

            // Construct temp shape
            const size_type size = TensorImplBase_::size();
            TiledArray::detail::Bitset<> s0(this->eval_shape());

            // Set the new shape
            for(size_type i = 0ul; i < size; ++i, ++range_it)
              if(s0[i] != 0ul)
                TensorImplBase_::shape(TiledArray::detail::calc_ordinal(*range_it, ip_weight, start), true);
          }

        } else {
          TensorImplBase_::shape(this->eval_shape());
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

        madness::Future<bool> child_eval_done = this->eval_children(vars, pmap);

        // Initialize the data container process map
        TensorImplBase_::pmap(pmap);

        madness::Future<bool> structure_eval_done =
            TensorImplBase_::get_world().taskq.add(*this,
            & TensorExpressionImpl_::internal_eval, vars, child_eval_done);

        return structure_eval_done;
      }

    }; // class TensorExpressionImpl

  }  // namespace expressions
}  // namespace TiledArray


#endif // TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED
