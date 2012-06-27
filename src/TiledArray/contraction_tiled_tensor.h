#ifndef TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/binary_tensor.h>
#include <TiledArray/cyclic_pmap.h>
#include <TiledArray/array.h>
#include <TiledArray/summa.h>
#include <TiledArray/vspgemm.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename>
    class ContractionTiledTensor;

    template <typename LExp, typename RExp>
    ContractionTiledTensor<LExp, RExp>
    make_contraction_tiled_tensor(const ReadableTiledTensor<LExp>& left, const ReadableTiledTensor<RExp>& right) {
      return ContractionTiledTensor<LExp, RExp>(left.derived(), right.derived());
    }

    template <typename Left, typename Right>
    struct TensorTraits<ContractionTiledTensor<Left, Right> > {
      typedef DynamicTiledRange trange_type;
      typedef typename trange_type::range_type range_type;
      typedef Tensor<typename math::ContractionValue<typename Left::value_type::value_type,
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
      typedef ContractionTensorImpl<Left, Right> impl_type;
      typedef typename impl_type::left_tensor_type left_tensor_type;
      typedef typename impl_type::right_tensor_type right_tensor_type;
      typedef ReadableTiledTensor<ContractionTiledTensor_> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::pmap_interface pmap_interface;
      typedef typename base::trange_type trange_type;
      typedef typename base::value_type value_type;
      typedef typename base::const_reference const_reference;
      typedef typename base::const_iterator const_iterator;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

    private:
      std::shared_ptr<impl_type> pimpl_;


      static std::shared_ptr<impl_type> make_pimpl(const left_tensor_type& left, const right_tensor_type& right) {
        impl_type* pimpl = NULL;
        if(left.is_dense() && right.is_dense())
          pimpl = new Summa<Left, Right>(left, right);
        else
          pimpl = new VSpGemm<Left, Right>(left, right);

        return std::shared_ptr<impl_type>(pimpl,
            madness::make_deferred_deleter<impl_type>(left.get_world()));
      }

    public:

      ContractionTiledTensor() : pimpl_() { }

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      ContractionTiledTensor(const left_tensor_type& left, const right_tensor_type& right) :
        pimpl_(make_pimpl(left, right))
      { }

      /// Copy constructor

      /// Create a shallow copy of \c other.
      /// \param other The object to be copied
      ContractionTiledTensor(const ContractionTiledTensor_& other) :
        pimpl_(other.pimpl_)
      { }

      /// Assignment operator

      /// Create a shallow copy of \c other.
      /// \param other THe object to be copied
      /// \return A reference to this object
      ContractionTiledTensor_& operator=(const ContractionTiledTensor_& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(pimpl_);
        TA_ASSERT(range() == dest.range());

        // Add result tiles to dest
        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
      }


      madness::Future<bool> eval(const VariableList& v, const std::shared_ptr<pmap_interface>& pmap) {
        TA_ASSERT(pimpl_);

        // This needs to be done before eval structure.
        return pimpl_->eval(v, pmap);
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const {
        TA_ASSERT(pimpl_);
        return trange().tiles();
      }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const {
        TA_ASSERT(pimpl_);
        return range().volume();
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
        return pimpl_->get_pmap();
      }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_dense();
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const {
        TA_ASSERT(pimpl_);
        return pimpl_->shape();
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const {
        TA_ASSERT(pimpl_);
        return pimpl_->trange();
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

      /// Release tensor data

      /// Clear all tensor data from memory. This is equivalent to
      /// \c ContractionTiledTensor().swap(*this) .
      void release() {
        if(pimpl_) {
          pimpl_->clear();
          pimpl_.reset();
        }
      }

      template <typename Archive>
      void serialize(const Archive&) { TA_EXCEPTION("Serialization not supported."); }

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
      static void store(const Archive&, const std::shared_ptr<TiledArray::math::Contraction>&) {
        TA_EXCEPTION("Serialization of shared_ptr not supported.");
      }
    };

    template <typename Archive>
    struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::math::Contraction> > {

      static void load(const Archive&, std::shared_ptr<TiledArray::math::Contraction>&) {
        TA_EXCEPTION("Serialization of shared_ptr not supported.");
      }
    };
  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
