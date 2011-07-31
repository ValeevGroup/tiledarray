#ifndef TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED
#define TILEDARRAY_PERMUTE_TENSOR_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/eval_tensor.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace expressions {

    template <typename, unsigned int>
    class PermuteTensor;

    template <typename Arg, unsigned int DIM>
    struct TensorTraits<PermuteTensor<Arg, DIM> > {
      typedef typename TensorSize::size_type size_type;
      typedef typename TensorSize::size_array size_array;
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

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      PermuteTensor(const arg_tensor_type& arg, const perm_type& p) :
        arg_(&arg), size_(permute_size(p, arg.size()), arg.order()), perm_(p), data_()
      { }

      PermuteTensor(const PermuteTensor_& other) :
        arg_(other.arg_), size_(other.size_), perm_(other.perm_), data_(other.data_)
      { }

      PermuteTensor_& operator=(const PermuteTensor_& other) {
        arg_ = other.arg_;
        size_ = other.size_;
        perm_ = other.perm_;
        data_ = other.data_;

        return *this;
      }

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
        TA_ASSERT(volume() == dest.volume());
        if(static_cast<const void*>(arg_) != static_cast<void*>(&dest))
          permute(dest);
        else
          std::copy(begin(), end(), dest.begin());
      }

      /// Tensor dimension accessor

      /// \return The number of dimensions
      static unsigned int dim() { return DIM; }

      /// Data ordering

      /// \return The data ordering type
      TiledArray::detail::DimensionOrderType order() const { return size_.order(); }

      /// Tensor dimension size accessor

      /// \return An array that contains the sizes of each tensor dimension
      const size_array& size() const { return size_.size(); }

      /// Tensor volume

      /// \return The total number of elements in the tensor
      size_type volume() const { return size_.volume(); }

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

      /// Make a permuted size array

      /// \tparam SizeArray The input size array type
      /// \param p The permutation that will be used to permute \c s
      /// \param s The size array to be permuted
      /// \return A permuted copy of \c s
      template <typename SizeArray>
      static size_array permute_size(const perm_type& p, const SizeArray& s) {
        size_array result(DIM);
        TiledArray::detail::permute_array(p.begin(), p.end(), s.begin(), result.begin());
        return result;
      }

      void lazy_eval() const {
        if(volume() != data_.volume()) {
          storage_type temp(volume());
          permute(temp);
          temp.swap(data_);
        }
      }

      template <typename CS, typename EvalArg, typename ResArray>
      void permute_helper(const EvalArg& arg, ResArray& result) const {
        typename CS::size_array p_size;
        TiledArray::detail::permute_array(perm_.begin(), perm_.end(), size_.size().begin(), p_size.begin());
        typename CS::size_array invp_weight = -perm_ ^ CS::calc_weight(p_size);

        typename CS::index i(0);
        const typename CS::index start(0);

        for(typename arg_tensor_type::const_iterator it = arg.begin(); it != arg.end();
            ++it, CS::increment_coordinate(i, start, size_.size()))
          result[CS::calc_ordinal(i, invp_weight)] = *it;
      }

      template <typename ResArray>
      void permute(ResArray& result) const {
        if(order() == TiledArray::detail::decreasing_dimension_order) {
          permute_helper<CoordinateSystem<DIM, 0ul, TiledArray::detail::decreasing_dimension_order,
            size_type> >(arg_->eval(), result);
        } else {
          permute_helper<CoordinateSystem<DIM, 0ul, TiledArray::detail::increasing_dimension_order,
            size_type> >(arg_->eval(), result);
        }
      }

      const arg_tensor_type* arg_; ///< Argument
      TensorSize size_; ///< Tensor size info
      perm_type perm_; ///< Transform operation
      mutable storage_type data_;
    }; // class PermuteTensor


  } // namespace expressions
} // namespace TiledArray

#endif // PERMUTE_TENSOR_H_
