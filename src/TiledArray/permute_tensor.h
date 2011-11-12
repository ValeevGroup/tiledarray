#ifndef TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED
#define TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/tensor.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/range.h>

namespace TiledArray {
  namespace expressions {

    template <typename, typename>
    class PermuteTensor;

    template <typename Arg, typename Perm>
    struct TensorTraits<PermuteTensor<Arg, Perm> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::value_type value_type;
      typedef typename DenseStorage<value_type>::const_reference const_reference;
      typedef typename DenseStorage<value_type>::const_iterator const_iterator;
      typedef typename DenseStorage<value_type>::difference_type difference_type;
      typedef typename DenseStorage<value_type>::const_pointer const_pointer;
    }; // struct TensorTraits<PermuteTensor<Arg, Perm>> >

    template <typename Arg, typename Perm>
    struct Eval<PermuteTensor<Arg, Perm> > {
      typedef Tensor<typename Arg::value_type, typename Arg::range_type> type;
    }; // struct Eval<PermuteTensor<Arg, DIM> >

    /// A permutation of an argument tensor

    /// \tparam Arg The argument type
    /// \tparam DIM The permutation dimension.
    template <typename Arg, typename Perm>
    class PermuteTensor : public DirectReadableTensor<PermuteTensor<Arg, Perm> > {
    public:
      typedef PermuteTensor<Arg, Perm> PermuteTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF(DirectReadableTensor<PermuteTensor_>, PermuteTensor_);
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object

      typedef Perm perm_type; ///< Permutation type

    private:
      // not allowed
      PermuteTensor_& operator=(const PermuteTensor_& other);

    public:

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      PermuteTensor(const Arg& arg, const perm_type& p) :
        arg_(arg),  perm_(p), range_(p ^ arg.range()), data_()
      { }

      PermuteTensor(const PermuteTensor_& other) :
        arg_(other.arg_), perm_(other.perm_), range_(other.range_), data_(other.data_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      eval_type eval() const {
        lazy_eval();
        return Tensor<value_type, range_type>(range_, data_);
      }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        if(data_.empty())
          permute(dest);
        else
          data_.eval_to(dest);
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const {
        return range_;
      }

      /// Tile size accessor

      /// \return The number of elements in the tile
      size_type size() const {
        return range_.volume();
      }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const {
        lazy_eval();
        return data_.begin();
      }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const {
        lazy_eval();
        return data_.end();
      }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        lazy_eval();
        return data_[i];
      }

    private:

      void lazy_eval() const {
        if(data_.empty()) {
          storage_type data(size());
          permute(data);
          data_.swap(data);
        }
      }

      /// Tensor permutation

      /// \tparam The result container type
      /// \param result The container that will hold the permuted tensor data
      template <typename Res>
      void permute(Res& result) const {
        // Construct the inverse permuted weight and size for this tensor
        typename range_type::size_array ip_weight = (-perm_) ^ range_.weight();
        const typename arg_tensor_type::range_type::index& start = arg_.range().start();

        // Coordinated iterator for the argument object range
        typename arg_tensor_type::range_type::const_iterator arg_range_it =
            arg_.range().begin();

        // permute the data
        const size_type end = size();
        for(size_type arg_it = 0ul; arg_it != end; ++arg_it, ++arg_range_it)
          result[TiledArray::detail::calc_ordinal(*arg_range_it, ip_weight, start)] = arg_[arg_it];
      }

      const arg_tensor_type& arg_; ///< Argument
      perm_type perm_; ///< Transform operation
      range_type range_; ///< Result tensor range
      mutable storage_type data_; ///< Result tensor data
    }; // class PermuteTensor


  } // namespace expressions
} // namespace TiledArray

#endif // PERMUTE_TENSOR_H_
