#ifndef TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED
#define TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/eval_tensor.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/range.h>

namespace TiledArray {
  namespace expressions {

    template <typename, unsigned int>
    class PermuteTensor;

    template <typename Arg, unsigned int DIM>
    struct TensorTraits<PermuteTensor<Arg, DIM> > {
      typedef typename Arg::size_type size_type;
      typedef typename Arg::range_type range_type;
      typedef typename Arg::value_type value_type;
      typedef typename DenseStorage<value_type>::const_reference const_reference;
      typedef typename DenseStorage<value_type>::const_iterator const_iterator;
      typedef typename DenseStorage<value_type>::difference_type difference_type;
      typedef typename DenseStorage<value_type>::const_pointer const_pointer;
    }; // struct TensorTraits<PermuteTensor<Arg, DIM>> >

    template <typename Arg, unsigned int DIM>
    struct Eval<PermuteTensor<Arg, DIM> > {
      typedef const PermuteTensor<Arg, DIM>& type;
    }; // struct Eval<PermuteTensor<Arg, DIM> >

    /// A permutation of an argument tensor

    /// \tparam Arg The argument type
    /// \tparam DIM The permutation dimension.
    template <typename Arg, unsigned int DIM>
    class PermuteTensor : public DirectReadableTensor<PermuteTensor<Arg, DIM> > {
    public:
      typedef PermuteTensor<Arg, DIM> PermuteTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHEIRATE_TYPEDEF(DirectReadableTensor<PermuteTensor_>, PermuteTensor_);
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object

      typedef Permutation<DIM> perm_type; ///< Permutation type

    private:
      // not allowed
      PermuteTensor_& operator=(const PermuteTensor_& other);

    public:

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      PermuteTensor(typename TensorArg<arg_tensor_type>::type arg, const perm_type& p) :
        arg_(arg), range_(p ^ arg.range()), perm_(p), data_()
      { }

      PermuteTensor(const PermuteTensor_& other) :
        arg_(other.arg_), range_(other.range_), perm_(other.perm_), data_(other.data_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      const PermuteTensor_& eval() const {
        lazy_eval();
        return *this;
      }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(size() == dest.size());
        if(static_cast<const void*>(&arg_) != static_cast<void*>(&dest))
          permute(dest);
        else
          std::copy(begin(), end(), dest.begin());
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return range_; }

      /// Tile size accessor

      /// \return The number of elements in the tile
      size_type size() const { return range_.volume(); }

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

      void check_dependancies(madness::TaskInterface* task) const {
        arg_.check_dependancies(task);
      }

    private:

      void lazy_eval() const {
        if(size() != data_.size()) {
          storage_type temp(size());
          permute(temp);
          temp.swap(data_);
        }
      }

      template <typename CS, typename ResArray>
      void permute_helper(ResArray& result) const {
        typename range_type::size_array invp_weight = -perm_ ^ range_.weight();

        typename CS::index i(0);
        const typename CS::index start(0);

        for(typename arg_tensor_type::const_iterator it = arg_.begin(); it != arg_.end();
            ++it, TiledArray::detail::increment_coordinate(i, range_))
          result[CS::calc_ordinal(i, invp_weight)] = *it;
      }

      template <typename ResArray>
      void permute(ResArray& result) const {
        if(range_.order() == TiledArray::detail::decreasing_dimension_order) {
          permute_helper<CoordinateSystem<DIM, 0ul, TiledArray::detail::decreasing_dimension_order,
            size_type> >(result);
        } else {
          permute_helper<CoordinateSystem<DIM, 0ul, TiledArray::detail::increasing_dimension_order,
            size_type> >(result);
        }
      }

      typename TensorMem<arg_tensor_type>::type arg_; ///< Argument
      range_type range_; ///< Tensor size info
      perm_type perm_; ///< Transform operation
      mutable storage_type data_;
    }; // class PermuteTensor


  } // namespace expressions
} // namespace TiledArray

#endif // PERMUTE_TENSOR_H_
