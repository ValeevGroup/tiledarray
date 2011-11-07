#ifndef TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/binary_tensor.h>

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
      ContractionTiledTensor_& operator=(const ContractionTiledTensor_&);


      left_tensor_type left_; ///< Left argument
      right_tensor_type right_; ///< Right argument
      trange_type trange_;
      TiledArray::detail::Bitset<> shape_;
      VariableList vars_;
      std::shared_ptr<storage_type> data_;

      struct reduce_op {
        typedef value_type result_type;
        typedef const value_type& first_argument_type;
        typedef const value_type& second_argument_type;
        typedef std::plus<typename value_type::value_type> plus_op;
        typedef BinaryTensor<value_type, value_type,  plus_op> plus_tensor;

        result_type operator()(first_argument_type left, second_argument_type right) const {
          return plus_tensor(left, right, plus_op());
        }

        template <typename Archive>
        void serialize(Archive&) { TA_ASSERT(false); }
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
        shape_((left.is_dense() || right.is_dense() ? 0 : cont->contract_shape(left, right))),
        vars_(cont->contract_vars(left.vars(), right.vars())),
        data_(new storage_type(left.get_world(), trange_.tiles().volume(), left.get_pmap(), false),
            madness::make_deferred_deleter<storage_type>(left.get_world()))
      {

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
              madness::Future<value_type> remote_reduction = data_->reduce(res_index,
                  local_reduction, reduce_op(), reduce_grp.begin(), reduce_grp.end(),
                  data_->owner(res_index));

              // Result returned on all nodes but only the root node has the final value.
              if(is_local(res_index))
                data_->set(res_index, remote_reduction);
            }

            // clear the reduce group
            reduce_grp.clear();

          }
          TiledArray::detail::increment_coordinate_helper(index.begin(), index.end(),
              res_start.begin(), res_size.begin());
        }
        data_->process_pending();
      }


      ContractionTiledTensor(const ContractionTiledTensor_& other) :
        left_(other.left_), right_(other.right_),
        trange_(other.trange_),
        shape_(other.shape_),
        vars_(other.vars_),
        data_(other.data_)
      { }

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
      const range_type& range() const { return trange_.tiles(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return trange_.tiles().volume(); }

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
      bool is_dense() const { return left_.is_dense() && right_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return trange_; }

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

      madness::World& get_world() const { return data_->get_world(); }


    private:
    }; // class ContractionTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

namespace madness {
  namespace archive {

    template <typename Archive, typename T>
    struct ArchiveStoreImpl;
    template <typename Archive, typename T>
    struct ArchiveLoadImpl;

    template <typename Archive>
    struct ArchiveStoreImpl<Archive, std::shared_ptr<TiledArray::math::Contraction> > {
      static void store(const Archive& ar, const std::shared_ptr<TiledArray::math::Contraction>&) {
        TA_ASSERT(false);
      }
    };

    template <typename Archive>
    struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::math::Contraction> > {

      static void load(const Archive& ar, std::shared_ptr<TiledArray::math::Contraction>&) {
        TA_ASSERT(false);
      }
    };
  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
