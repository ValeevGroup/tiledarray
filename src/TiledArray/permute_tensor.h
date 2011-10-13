#ifndef TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED
#define TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/tensor.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/range.h>

namespace TiledArray {
  namespace expressions {

    template <typename, unsigned int>
    class PermuteTensor;

    template <typename Arg, unsigned int DIM>
    struct TensorTraits<PermuteTensor<Arg, DIM> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::value_type value_type;
      typedef typename DenseStorage<value_type>::const_reference const_reference;
      typedef typename DenseStorage<value_type>::const_iterator const_iterator;
      typedef typename DenseStorage<value_type>::difference_type difference_type;
      typedef typename DenseStorage<value_type>::const_pointer const_pointer;
    }; // struct TensorTraits<PermuteTensor<Arg, DIM>> >

    template <typename Arg, unsigned int DIM>
    struct Eval<PermuteTensor<Arg, DIM> > {
      typedef const Tensor<typename Arg::value_type, typename Arg::range_type>& type;
    }; // struct Eval<PermuteTensor<Arg, DIM> >

    /// A permutation of an argument tensor

    /// \tparam Arg The argument type
    /// \tparam DIM The permutation dimension.
    template <typename Arg, unsigned int DIM>
    class PermuteTensor : public DirectReadableTensor<PermuteTensor<Arg, DIM> > {
    public:
      typedef PermuteTensor<Arg, DIM> PermuteTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF(DirectReadableTensor<PermuteTensor_>, PermuteTensor_);
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object
      typedef Tensor<value_type, range_type> eval_type;

      typedef Permutation<DIM> perm_type; ///< Permutation type

    private:
      // not allowed
      PermuteTensor_& operator=(const PermuteTensor_& other);

    public:

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      PermuteTensor(const Arg& arg, const perm_type& p) :
        arg_(arg),  perm_(p), eval_()
      { }

      PermuteTensor(const PermuteTensor_& other) :
        arg_(other.arg_), perm_(other.perm_), eval_(other.eval_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      const eval_type& eval() const {
        lazy_eval();
        return *this;
      }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        lazy_eval();
        eval_.eval_to(dest);
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const {
        lazy_eval();
        return eval_.range();
      }

      /// Tile size accessor

      /// \return The number of elements in the tile
      size_type size() const {
        return arg_.size();
      }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const {
        lazy_eval();
        return eval_.begin();
      }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const {
        lazy_eval();
        return eval_.end();
      }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        lazy_eval();
        return eval_[i];
      }

    private:

      void lazy_eval() const {
        if(eval_.size()) {
          range_type range = perm_ ^ (arg_.range());
          storage_type data(range.volume());
          permute(data, range);
          eval_type(range, data.begin()).swap(eval_);
        }
      }

      /// Tensor permutation

      /// \tparam The result container type
      /// \param result The container that will hold the permuted tensor data
      template <typename Res>
      void permute(Res& result, const range_type& range) const {
        // Construct the inverse permuted weight and size for this tensor
        const perm_type ip = -perm_;
        typename range_type::size_array ip_weight = ip ^ range.weight();
        const typename range_type::index ip_start = ip ^ arg_.range().start();

        // Coordinated iterator for the argument object range
        typename arg_tensor_type::range_type::const_iterator arg_range_it =
            arg_.range().begin();

        // permute the data
        const size_type s = arg_.size();
        for(size_type arg_it = 0ul; arg_it != s; ++arg_it, ++arg_range_it)
          result[TiledArray::detail::calc_ordinal(*arg_range_it, ip_weight, ip_start)] = arg_[arg_it];
      }

      const arg_tensor_type& arg_; ///< Argument
      perm_type perm_; ///< Transform operation
      mutable eval_type eval_; ///< Evaluated tensor
    }; // class PermuteTensor


  } // namespace expressions
} // namespace TiledArray

#endif // PERMUTE_TENSOR_H_
