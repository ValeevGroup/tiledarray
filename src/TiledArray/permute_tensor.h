#ifndef TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED
#define TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/tensor.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/range.h>

namespace TiledArray {
  namespace expressions {

    template <typename>
    class PermuteTensor;

    /// Permutation tensor factory function

    /// Construct a \c PermutationTensor<T> object.
    /// \tparam Exp The argument expression type
    /// \param t The argument expression that will be permuted
    /// \param p The permutation that will be applied to \c t
    /// \return A PermuteTensor<Exp> object
    template <typename Exp>
    inline PermuteTensor<Exp> make_permute_tensor(const ReadableTensor<Exp>& t, Permutation p) {
      return PermuteTensor<Exp>(t.derived(), p);
    }

    /// No permutation factory function

    /// This function overloads the original \c make_permutation_tensor
    /// function. Since no permutation is needed in this case, a reference to
    /// the original tensor is passed through.
    /// \tparam Exp The expression type
    /// \param t The expression object
    /// \return A reference to the original expression
    template <typename Exp>
    inline const ReadableTensor<Exp>&
    make_permute_tensor(const ReadableTensor<Exp>& t, const TiledArray::detail::NoPermutation&) {
      return t;
    }

    template <typename Arg>
    struct TensorTraits<PermuteTensor<Arg> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::value_type value_type;
      typedef typename DenseStorage<value_type>::const_reference const_reference;
      typedef typename DenseStorage<value_type>::const_iterator const_iterator;
      typedef typename DenseStorage<value_type>::difference_type difference_type;
      typedef typename DenseStorage<value_type>::const_pointer const_pointer;
    }; // struct TensorTraits<PermuteTensor<Arg, Perm>> >

    template <typename Arg>
    struct Eval<PermuteTensor<Arg> > {
      typedef const Tensor<typename Arg::value_type, typename Arg::range_type>& type;
    }; // struct Eval<PermuteTensor<Arg, DIM> >

    /// A permutation of an argument tensor

    /// \tparam Arg The argument type
    /// \tparam DIM The permutation dimension.
    template <typename Arg>
    class PermuteTensor : public DirectReadableTensor<PermuteTensor<Arg> > {
    public:
      typedef PermuteTensor<Arg> PermuteTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF(DirectReadableTensor<PermuteTensor_>, PermuteTensor_);
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object

    private:
      // not allowed
      PermuteTensor_& operator=(const PermuteTensor_& other);

    public:

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      PermuteTensor(const Arg& arg, const Permutation& p) :
          t_(p ^ arg.range())
      {
        permute(arg, p);
      }

      PermuteTensor(const PermuteTensor_& other) :
          t_(other.t_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      eval_type eval() const {
        return t_;
      }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(dest.size() == t_.size());
        t_.eval_to(dest);
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const {
        return t_.range();
      }

      /// Tile size accessor

      /// \return The number of elements in the tile
      size_type size() const {
        return t_.size();
      }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const {
        return t_.begin();
      }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const {
        return t_.end();
      }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        return t_[i];
      }

    private:

      /// Tensor permutation

      /// \param arg The argument that will be permuted
      void permute(const Arg& arg, const Permutation& perm) {
        // Construct the inverse permuted weight and size for this tensor
        typename range_type::size_array ip_weight = (-perm) ^ t_.range().weight();
        const typename arg_tensor_type::range_type::index& start = arg.range().start();

        // Coordinated iterator for the argument object range
        typename arg_tensor_type::range_type::const_iterator arg_range_it =
            arg.range().begin();

        // permute the data
        const size_type end = size();
        for(size_type arg_it = 0ul; arg_it != end; ++arg_it, ++arg_range_it)
          t_[TiledArray::detail::calc_ordinal(*arg_range_it, ip_weight, start)] = arg[arg_it];
      }

      Tensor<value_type, range_type> t_; ///< Result tensor data
    }; // class PermuteTensor


  } // namespace expressions
} // namespace TiledArray

#endif // PERMUTE_TENSOR_H_
