#ifndef TILEDARRAY_ARG_TENSOR_H__INCLUDED
#define TILEDARRAY_ARG_TENSOR_H__INCLUDED

#include <TiledArray/tensor_base.h>
#include <TiledArray/future_tensor.h>

namespace TiledArray {
  namespace expressions {

    template <typename> class ArgTensor;

    template <typename T>
    struct TensorTraits<ArgTensor<T> > {
      typedef typename T::range_type range_type;
      typedef typename T::value_type value_type;
      typedef typename T::const_reference const_reference;
    }; // struct TensorTraits<ArgTensor<T> >

    template <typename T>
    struct Eval<ArgTensor<T> > {
      typedef typename Eval<T>::type type;
    }; // struct Eval<ArgTensor<T> >

    /// A tensor that is given as an argument to a transformation tensor.

    /// This object is a wrapper around a tensor argument. It is designed to
    /// handle the logic of a tensor that may either be local or remote in a
    /// transparent manner.
    /// \tparam T The arg tensor type.
    template <typename T>
    class ArgTensor : public ReadableTensor<ArgTensor<T> > {
    public:
      typedef ArgTensor<T> ArgTensor_;
      TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF(ReadableTensor<ArgTensor_>, ArgTensor_);
      typedef T arg_tensor_type;
      typedef FutureTensor<EvalTensor<typename T::value_type, typename T::range_type> > future_tensor;
      typedef typename arg_tensor_type::storage_type storage_type; /// The storage type for this object

      /// Construct a tensor from a local argument

      /// \param t The local tensor
      ArgTensor(const arg_tensor_type& t) :
        local_(NULL), remote_(NULL)
      {
        local_ = new (&memory_) arg_tensor_type(t);
      }

      /// Construct a tensor from a local argument

      /// \param t The local tensor
      ArgTensor(const future_tensor& t) :
        local_(NULL), remote_(NULL)
      {
        remote_ = new (&memory_) future_tensor(t);
      }

      /// Copy constructor

      /// \param other The object to be copied
      ArgTensor(const ArgTensor<T>& other) :
        local_(NULL), remote_(NULL)
      {
        if(other.is_local())
          local_ = new (&memory_) arg_tensor_type(* other.local());
        else
          remote_ = new (&memory_) future_tensor(* other.remote());
      }

      ~ArgTensor() {
        if(is_local())
          local_->~arg_tensor_type();
        else
          remote_->~future_tensor();
      }

      /// Tensor range object accessor

      /// \return The range object of the tensor
      /// \throw TiledArray::Exception If the remote has not been evaluated.
      /// \throw TiledArray::Exception If the tensor is zero.
      const range_type& range() const {
        return (is_local() ? local()->range() : remote()->range());
      }

      /// Tensor size accessor

      /// \return The number of elements in the tensor
      /// \throw TiledArray::Exception If the remote has not been evaluated.
      /// \throw TiledArray::Exception If the tensor is zero.
      size_type size() const {
        return (is_local() ? local()->size() : remote()->size());
      }


      /// \throw TiledArray::Exception If the future has not been evaluated.
      typename Eval<arg_tensor_type>::type eval() const {
        return (is_local() ? local()->eval() : remote()->eval());
      }


      /// \throw TiledArray::Exception If the future has not been evaluated.
      template<typename Dest>
      void eval_to(Dest& dest) const {
        if(is_local())
          local()->eval_to(dest);
        remote()->eval_to(dest);
      }

      // element access

      /// \throw TiledArray::Exception If the future has not been evaluated.
      const_reference operator[](size_type i) const {
        return (is_local() ? (* local())[i] : (* remote())[i]);
      }

      /// Check and add dependencies for \c task .

      /// If the future has not been evaluated, the task dependency counter is
      /// incremented and a callback is registered. If \c task is \c NULL ,
      /// nothing is done.
      /// \param task The task that depends on this future tensor.
      void check_dependency(madness::TaskInterface* task) const {
        if(is_local())
          local()->check_dependency(task);
        else
          remote()->check_dependency(task);
      }

      /// Check if the tensor is stored locally.

      /// \return \c true if the tensor is stored locally, \c false otherwise.
      bool is_local() const { return local_ != NULL; }

    private:

      // This is here to simulate a union

      const arg_tensor_type* local() const {
        TA_ASSERT(local_)
        return local_;
      }

      arg_tensor_type* local() {
        TA_ASSERT(local_)
        return local_;
      }

      const future_tensor* remote() const {
        TA_ASSERT(remote_);
        return remote_;
      }

      future_tensor* remote() {
        TA_ASSERT(remote_);
        return remote_;
      }

      char memory_[(sizeof(arg_tensor_type) > sizeof(future_tensor) ? sizeof(arg_tensor_type) : sizeof(future_tensor))];
      arg_tensor_type* local_;
      future_tensor* remote_;
    };

  }  // namespace expressions
}  // namespace TiledArray


#endif // TILEDARRAY_ARG_TENSOR_H__INCLUDED
