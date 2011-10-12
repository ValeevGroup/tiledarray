#ifndef TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/eval_task.h>
#include <TiledArray/reduce_task.h>

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
/*
    struct ArrayOpImpl {
      typedef typename StaticRange<res_packed_cs>::index res_packed_index;
      typedef typename StaticRange<left_packed_cs>::index left_packed_index;
      typedef typename StaticRange<right_packed_cs>::index right_packed_index;
      typedef madness::Range<typename Range<res_packed_cs>::const_iterator> range_type;

      ArrayOpImpl(const std::shared_ptr<cont_type>& cont, const ResArray& result,
        const left_array_type& left, const right_array_type& right) :
          world_(& result.get_world()),
          contraction_(cont),
          result_(result),
          left_(left.array()),
          right_(right.array()),
          res_range_(),
          left_range_(),
          right_range_()
      {
        // Get packed sizes
        const typename cont_type::packed_size_array size =
            cont->pack_arrays(left_.range().size(), right_.range().size());

        // Set packed range dimensions
        res_range_.resize(res_packed_index(0),
            res_packed_index(size[0], size[2], size[1], size[3]));
        left_range_.resize(left_packed_index(0),
            left_packed_index(size[0], size[4], size[1]));
        right_range_.resize(right_packed_index(0),
            right_packed_index(size[2], size[4], size[3]));
      }

      range_type range() const {
        return range_type(res_range_.begin(), res_range_.end());
      }

      void generate_tasks(const typename StaticRange<res_packed_cs>::const_iterator& it) const {
        const ordinal_index res_index = res_range_.ord(*it);

        // Check that the result tile has a value
        if(result_.is_zero(res_index))
          return;

        ordinal_index I = left_range_.size()[1];
        // Reduction objects
        std::vector<ProcessID> reduce_grp;
        reduce_grp.reserve(I);
        detail::ReduceTask<typename ResArray::value_type, addtion_op_type >
            local_reduce_op(*world_, addtion_op_type());

        for(ordinal_index i = 0; i < I; ++i) {
          // Get the a and b index
          const ordinal_index left_index = left_ord(*it, i);
          const ordinal_index right_index = right_ord(*it, i);

          // Check for non-zero contraction.
          if((! left_.is_zero(left_index)) && (! right_.is_zero(right_index))) {

            // Add to the list nodes involved in the reduction reduction group
            reduce_grp.push_back(left_.owner(left_index));

            // Do the contraction on the left node.
            if(left_.is_local(left_index)) {
              // Do the tile-tile contraction and add to local reduction list
              local_reduce_op.add(world_->taskq.add(madness::make_task(make_cont_op(res_index),
                  left_.find(left_index), right_.find(right_index))));
            }
          }
        }

        // Reduce contracted tile pairs
        if(local_reduce_op.size() != 0) {
          // Local reduction
          madness::Future<typename ResArray::value_type> local_red = local_reduce_op();

          // Remote reduction
          result_.reduce(res_index, local_red, reduce_grp.begin(),
              reduce_grp.end(), addtion_op_type());
        }
      }

    private:

      /// Contraction operation factory

      /// \param index The ordinal index of the result tile
      /// \retur The contraction operation for \c index
      contraction_op_type make_cont_op(const ordinal_index& index) const {
        return contraction_op_type(contraction_,
            result_.tiling().make_tile_range(index));
      }

      ordinal_index left_ord(const res_packed_index& res_index, ordinal_index i) const {
        const typename StaticRange<left_packed_cs>::size_array& weight = left_range_.weight();
        return res_index[0] * weight[0] + i * weight[1] + res_index[2] * weight[2];
      }

      ordinal_index right_ord(const res_packed_index& res_index, ordinal_index i) const {
        const typename StaticRange<right_packed_cs>::size_array& weight = right_range_.weight();
        return res_index[1] * weight[0] + i * weight[1] + res_index[3] * weight[2];
      }

      madness::World* world_;
      std::shared_ptr<cont_type> contraction_;
      mutable ResArray result_;
      LeftArray left_;
      RightArray right_;
      StaticRange<res_packed_cs> res_range_;
      StaticRange<left_packed_cs> left_range_;
      StaticRange<right_packed_cs> right_range_;
    }; // struct ArrayOpImpl

    struct ArrayOp {

      ArrayOp(const std::shared_ptr<cont_type>& cont, const ResArray& result,
        const left_array_type& left, const right_array_type& right) :
          pimpl_(new ArrayOpImpl(cont, result, left, right))
      { }

      ArrayOp(const ArrayOp& other) :
          pimpl_(other.pimpl_)
      { }

      ArrayOp& operator=(const ArrayOp& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      typename ArrayOpImpl::range_type range() const { return pimpl_->range(); }

      bool operator()(const typename StaticRange<res_packed_cs>::const_iterator& it) const {
        pimpl_->generate_tasks(it);
        return true;
      }

      template <typename Archive>
      void serialize(const Archive& ar) {
        TA_ASSERT(false);
      }

    private:

      std::shared_ptr<ArrayOpImpl> pimpl_;
    }; // struct ArrayOp
*/
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

      struct reduce_op {
        typedef value_type result_type;
        typedef const typename left_tensor_type::value_type& first_argument_type;
        typedef const typename right_tensor_type::value_type& second_argument_type;

        result_type operator()(first_argument_type left, second_argument_type right) {
          return left + right;
        }
      };

      static value_type contract(const typename left_tensor_type::value_type& left,
          const typename right_tensor_type::value_type& right,
          const std::shared_ptr<math::Contraction>& cont)
      {
        return ContractionTensor<typename left_tensor_type::value_type,
            typename right_tensor_type::value_type>(left, right, cont);
      }

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
        const size_type n = size();

        const typename math::Contraction::packed_size_array size =
            cont->pack_arrays(left_.range().size(), right_.range().size());

        // Construct packed size arrays
        const std::array<size_type, 4> res_start = {{ 0, 0, 0, 0 }};
        const std::array<size_type, 4> res_size = {{ size[0], size[2], size[1], size[3] }};
        const std::array<size_type, 3> left_size = {{ size[0], size[4], size[1] }};
        const std::array<size_type, 3> right_size = {{ size[2], size[4], size[3] }};

        // Construct packed weight arrays
        std::array<size_type, 4> res_weight;
        std::array<size_type, 3> left_weight;
        std::array<size_type, 3> right_weight;
        TiledArray::detail::calc_weight(res_weight, res_size, range().order());
        TiledArray::detail::calc_weight(left_weight, left_size, left.range().order());
        TiledArray::detail::calc_weight(right_weight, right_size, right.range().order());


        std::vector<ProcessID> reduce_grp;
        reduce_grp.reserve(left_size[1]);

        std::array<size_type, 4> index = {{ 0, 0, 0, 0 }};
        for(size_type i = 0ul; i < n; ++i) {
          if(! is_zero(i)) {

            // Reduction objects
            TiledArray::detail::ReduceTask<value_type, reduce_op >
                local_reduce_op(get_world(), reduce_op());

            // Iterate over common indexes
            size_type left_index =
                  left_weight[0] * index[0]
                + left_weight[2] * index[2];
            size_type right_index =
                  right_weight[0] * index[1]
                + right_weight[2] * index[3];
            for(size_type i = 0; i < left_size[1]; ++i) {

              // Check for non-zero contraction.
              if(left.is_zero(left_index) || right_.is_zero(right_index))
                continue;

              // Add to the list nodes involved in the reduction reduction group
              reduce_grp.push_back(left.owner(left_index));

              // Do the contraction on the left node.
              if(left.is_local(left_index)) {
                // Do the tile-tile contraction and add to local reduction list
                local_reduce_op.add(get_world().taskq.add(& contract,
                    left[left_index], right[right_index], cont));
              }

              left_index += left_weight[1];
              right_index += right_weight[1];
            }

            // Reduce contracted tile pairs
            if(local_reduce_op.size() != 0) {

              // Get the result tile ordinal index
              const size_type res_index =
                    res_weight[0] * index[0]
                  + res_weight[1] * index[1]
                  + res_weight[2] * index[2]
                  + res_weight[3] * index[3];

              // Start the local reduction and get the result future.
              madness::Future<value_type> local_reduction = local_reduce_op();

              // Start the remote reduction and get the result future
              madness::Future<value_type> remote_reduction = data_.reduce(res_index,
                  local_reduction, reduce_op(), reduce_grp.begin(), reduce_grp.end(),
                  data_.owner(res_index));

              // Result returned on all nodes but only the root node has the final value.
              if(is_local(res_index))
                data_.set(res_index, remote_reduction);
            }

            // clear the reduce group
            reduce_grp.clear();

          }
          TiledArray::detail::increment_coordinate_helper(index.begin(), index.end(),
              res_start.begin(), res_size.begin());
        }
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

      madness::World& get_world() const { return data_.get_world(); }


    private:
    }; // class ContractionTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
