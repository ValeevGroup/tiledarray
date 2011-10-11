#ifndef TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/eval_task.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename>
    class ContractionTiledTensor;

    template <typename Left, typename Right>
    struct TensorTraits<ContractionTiledTensor<Left, Right> > {
      typedef DynamicTiledRange trange_type;
      typedef typename trange_type::range_type range_type;
      typedef Tensor<typename ContractionValue<typename Left::value_type::value_type,
          typename Right::value_type::value_type>::type, range_type> value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
    }; // struct TensorTraits<ContractionTiledTensor<Arg, Op> >


    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Left, typename Right>
    class ContractionTiledTensor : public ReadableTiledTensor<ContractionTiledTensor<Left, Right> > {
    public:
      typedef ContractionTiledTensor<Left, Right> ContractionTiledTensor_;
      typedef Left left_tensor_type;
      typedef Right right_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<ContractionTiledTensor_>, ContractionTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

    private:
      // Not allowed
      ContractionTiledTensor(const ContractionTiledTensor_&);
      ContractionTiledTensor_& operator=(const ContractionTiledTensor_&);


      left_tensor_type left_; ///< Left argument
      right_tensor_type right_; ///< Right argument
      trange_type trange_;
      TiledArray::detail::Bitset<> shape_;
      VariableList vars_;
      std::shared_ptr<storage_type> data_;

    public:

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      ContractionTiledTensor(const left_tensor_type& left, const right_tensor_type& right, const std::shared_ptr<math::Contraction>& cont) :
        left_(left), right_(right),
        trange_(cont->contract_trange(left.trange(), right.trange())),
        shape_((left.is_dense() || right.is_dense() ? 0 : cont->contract_shape(left.get_shape(), right.get_shape()))),
        vars_(),
        data_(new storage_type(left.get_world(), trange_.range().volume(), left.get_pmap(), false),
            madness::make_deferred_deleter<storage_type>(left.get_world()))
      {

        cont->contract_array(vars_, left.vars(), right.vars());
        data_->process_pending();
      }


      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(range() == dest.range());

        // Add result tiles to dest
        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const { return trange_.range(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return data_->size(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const { return data_->owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return data_->is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const {
        TA_ASSERT(range().includes(i));
        if(is_dense())
          return false;
        return ! (shape_[i]);
      }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return data_->get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return left_.is_dense() || right_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return shape_; }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return trange_; }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference get_local(size_type i) const {
        TA_ASSERT(left_.is_local(i));
        TA_ASSERT(right_.is_local(i));
        return op_(left_.get_local(i), right_.get_local(i));
      }


      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return data_->begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return data_->end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return vars_; }


    private:
    }; // class ContractionTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
