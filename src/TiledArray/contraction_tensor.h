#ifndef TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED

#include <TiledArray/tensor.h>
#include <TiledArray/contraction.h>
#include <TiledArray/permute_tensor.h>
#include <world/shared_ptr.h>

namespace TiledArray {
  namespace expressions {

    template <typename, typename>
    class ContractionTensor;

    /// Contraction tensor factory function

    /// Construct a ContractionTensor object.
    /// \tparam LExp Left expression type
    /// \tparam RExp Right expression type
    /// \param left The left expression object
    /// \param right The right expression object
    /// \param c A shared pointer to the contraction definition object
    /// \return A \c ContractionTensor<LExp, RExp> object
    template <typename LExp, typename RExp>
    inline ContractionTensor<LExp, RExp> make_contraction_tensor(const ReadableTensor<LExp>& left,
        const ReadableTensor<RExp>& right, const std::shared_ptr<math::Contraction>& c)
    { return ContractionTensor<LExp, RExp>(left.derived(), right.derived(), c); }

    template <typename LeftArg, typename RightArg>
    struct TensorTraits<ContractionTensor<LeftArg, RightArg> > {
      typedef std::size_t size_type;
      typedef DynamicRange range_type;
      typedef typename math::ContractionValue<typename LeftArg::value_type,
          typename RightArg::value_type>::type value_type;
      typedef typename DenseStorage<value_type>::const_reference const_reference;
      typedef typename DenseStorage<value_type>::const_iterator const_iterator;
      typedef typename DenseStorage<value_type>::difference_type difference_type;
      typedef typename DenseStorage<value_type>::const_pointer const_pointer;
    }; // struct TensorTraits<ContractionTensor<LeftArg, RightArg> >

    template <typename LeftArg, typename RightArg>
    struct Eval<ContractionTensor<LeftArg, RightArg> > {
      typedef const Tensor<typename math::ContractionValue<typename LeftArg::value_type,
          typename RightArg::value_type>::type>& type;
    }; // struct Eval<ContractionTensor<LeftArg, RightArg> >

    /// Tensor that is composed from two contracted argument tensors

    /// The tensor elements are constructed using a binary transformation
    /// operation.
    /// \tparam LeftArg The left-hand argument type
    /// \tparam RightArg The right-hand argument type
    /// \tparam Op The binary transform operator type.
    template <typename LeftArg, typename RightArg>
    class ContractionTensor : public DirectReadableTensor<ContractionTensor<LeftArg, RightArg> > {
    public:
      typedef ContractionTensor<LeftArg, RightArg> ContractionTensor_;
      typedef LeftArg left_tensor_type;
      typedef RightArg right_tensor_type;
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHERIT_TYPEDEF(DirectReadableTensor<ContractionTensor_>, ContractionTensor_);
      typedef math::Contraction contract_type; ///< Contraction type
      typedef typename Tensor<value_type, range_type>::storage_type storage_type;

    private:
      // not allowed
      ContractionTensor_& operator=(const ContractionTensor_&);

    public:

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param c Shared pointer to contraction object
      ContractionTensor(const left_tensor_type& left, const right_tensor_type& right, const std::shared_ptr<contract_type>& c) :
        left_(left),
        right_(right),
        eval_(),
        contraction_(c)
      { }

      /// Copy constructor
      ContractionTensor(const ContractionTensor_& other) :
        left_(other.left_),
        right_(other.right_),
        eval_(other.eval_),
        contraction_(other.contraction_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      eval_type eval() const {
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

      /// Tensor size

      /// \return The number of elements in the tensor
      size_type size() const { return range().volume(); }


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



      /// Evaluate the tensor only when the data is needed
      void lazy_eval() const {
        if(! eval_.size()) {
          contraction_->permute_contract_tensor(left_, right_).swap(eval_);
        }
      }

      const left_tensor_type& left_; ///< Left argument
      const right_tensor_type& right_; ///< Right argument
      mutable Tensor<value_type, range_type> eval_; ///< The evaluated tensor data
      std::shared_ptr<contract_type> contraction_; ///< Contraction definition
    }; // class ContractionTensor

  } // namespace expressions



  namespace math {

    // This function needs to be here to break cyclic dependencies.
    template <typename LeftTensor, typename RightTensor>
    inline detail::Bitset<> Contraction::contract_shape(const LeftTensor& left, const RightTensor& right) {
      typedef expressions::Tensor<TiledArray::detail::Bitset<>::value_type> eval_tensor;
      eval_tensor left_map(left.range(), left.get_shape().begin());
      eval_tensor right_map(right.range(), right.get_shape().begin());

      expressions::ContractionTensor<eval_tensor, eval_tensor>
      contract_tensor(left_map, right_map,
          std::shared_ptr<Contraction>(this, & madness::detail::no_delete<Contraction>));

      return detail::Bitset<>(contract_tensor.begin(), contract_tensor.end());
    }

  } // namespace math


} // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
