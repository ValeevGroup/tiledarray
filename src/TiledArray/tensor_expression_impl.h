#ifndef TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED
#define TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED

#include <TiledArray/tensor_impl_base.h>

namespace TiledArray {
  namespace expressions {

    template <typename TRange, typename Tile>
    class TensorExpressionImpl : public TiledArray::detail::TensorImplBase<TRange, Tile> {
    public:
      typedef TensorExpressionImpl<TRange, Tile> TensorExpressionImpl_;
      typedef TiledArray::detail::TensorImplBase<TRange, Tile> TensorImplBase_;

      typedef typename TensorImplBase_::size_type size_type;
      typedef typename TensorImplBase_::trange_type trange_type;
      typedef typename TensorImplBase_::range_type range_type;
      typedef typename TensorImplBase_::pmap_interface pmap_interface;
      typedef typename TensorImplBase_::value_type value_type;

    private:
      VariableList vars_;
      Permutation perm_;
      const typename TensorImplBase_::trange_type trange_;
      volatile bool evaluated_;

      bool perm_structure(const VariableList& vars) {
        // Get the permutation to go from the current variable list to vars such
        // that:
        //   vars = perm ^ vars_
        perm_ = vars.permutation(vars_);

        // Permute the tiled range
        TensorImplBase_::trange(perm_ ^ TensorImplBase_::trange());

        // If not dense, permute the shape
        if(! TensorImplBase_::is_dense()) {
          // Construct the inverse permuted weight and size for this tensor
          typename range_type::size_array ip_weight = (-perm_) ^ TensorImplBase_::range().weight();
          const typename range_type::index& start = trange_.tiles().start();

          // Coordinated iterator for the argument object range
          typename range_type::const_iterator range_it =
              trange_.tiles().begin();

          const TiledArray::detail::Bitset<> shape = TensorImplBase_::shape();

          // permute the data
          const size_type end = TensorImplBase_::size();
          for(size_type i = 0ul; i != end; ++i, ++range_it)
            TensorImplBase_::shape(TiledArray::detail::calc_ordinal(*range_it, ip_weight, start), shape[i]);
        }

        // Store the new variable list
        vars_ = vars;

        return true;
      }

      /// Task function for permuting result tensor
      value_type permute_task(const value_type& value) const {
        return expressions::make_permute_tensor(value, perm_);
      }

    protected:

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
        const TiledArray::detail::Bitset<>& shape = TiledArray::detail::Bitset<>(0ul)) :
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
      virtual void eval_tiles() = 0;

      virtual bool eval_children(const expressions::VariableList&,
          const std::shared_ptr<pmap_interface>&) { return true; }

      bool internal_eval_children(const expressions::VariableList& vars,
          const std::shared_ptr<pmap_interface>& pmap)
      { return this->eval_children(vars, pmap); }

      madness::Void internal_eval_tiles(bool, bool) {
        evaluated_ = true;
        this->eval_tiles();
        return madness::None;
      }

    public:

      madness::Future<bool> eval(const expressions::VariableList& vars,
          const std::shared_ptr<pmap_interface>& pmap)
      {
        TA_ASSERT(! evaluated_);

        madness::Future<bool> child_eval_done =
            TensorImplBase_::get_world().taskq.add(*this,
                & TensorExpressionImpl_::internal_eval_children, vars, pmap);

        // Initialize the data container process map
        TensorImplBase_::pmap(pmap);

        madness::Future<bool> structure_eval_done =
            (vars == vars_ ?
                madness::Future<bool>(true) :
                TensorImplBase_::get_world().taskq.add(*this,
                    & TensorExpressionImpl_::perm_structure, vars));

        // Do tile evaluation step
        TensorImplBase_::get_world().taskq.add(*this,
            & TensorExpressionImpl_::internal_eval_tiles, structure_eval_done,
            child_eval_done);

        return structure_eval_done;
      }

    }; // class TensorExpressionImpl

  }  // namespace expressions
}  // namespace TiledArray


#endif // TILEDARRAY_TENSOR_EXPRESSION_IMPL_H__INCLUDED
