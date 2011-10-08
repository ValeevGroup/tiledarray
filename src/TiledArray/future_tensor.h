#ifndef TILEDARRAY_FUTURE_TENSOR_H__INCLUDED
#define TILEDARRAY_FUTURE_TENSOR_H__INCLUDED

#include <TiledArray/tensor_base.h>
#include <world/worldtask.h> // for TaskInterface and Future

namespace TiledArray {
  namespace expressions {

    template <typename> class FutureTensor;

    template <typename T>
    struct TensorTraits<FutureTensor<T> > {
      typedef typename T::range_type range_type;
      typedef typename T::value_type value_type;
      typedef typename T::const_reference const_reference;
    }; // struct TensorTraits<FutureTensor<T> >

    template <typename T>
    struct Eval<FutureTensor<T> > {
      typedef typename Eval<T>::type type;
    }; // struct Eval<FutureTensor<T> >

    /// Wrapper object for tensors held by \c Futures

    /// \tparam T The tensor type
    template <typename T>
    class FutureTensor : public ReadableTensor<FutureTensor<T> > {
    public:
      typedef FutureTensor<T> FutureTensor_;
      typedef typename T::eval_type eval_type;
      TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF(ReadableTensor<FutureTensor_>, FutureTensor_);
      typedef T tensor_type;
      typedef madness::Future<tensor_type> future;

      /// Constructor

      /// \param f The future to the tensor
      FutureTensor(const future& f) :
        tensor_(f)
      { }

      /// Copy constructor

      /// \param other The object to be copied
      FutureTensor(const FutureTensor<T>& other) :
        tensor_(other.tensor_)
      { }

      /// Tensor range object accessor

      /// \return The range object of the tensor
      /// \throw TiledArray::Exception If the future has not been evaluated.
      const range_type& range() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().range();
      }

      /// Tensor volume accessor

      /// \return The volume of the tensor
      /// \throw TiledArray::Exception If the future has not been evaluated.
      size_type size() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().size();
      }


      /// \throw TiledArray::Exception If the future has not been evaluated.
      typename Eval<tensor_type>::type eval() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().eval();
      }


      /// \throw TiledArray::Exception If the future has not been evaluated.
      template<typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().eval_to(dest);
      }

      // element access

      /// \throw TiledArray::Exception If the future has not been evaluated.
      const_reference operator[](size_type i) const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get()[i];
      }

      /// Check and add dependencies for \c task .

      /// If the future has not been evaluated, the task dependency counter is
      /// incremented and a callback is registered. If \c task is \c NULL ,
      /// nothing is done.
      /// \param task The task that depends on this future tensor.
      void check_dependency(madness::TaskInterface* task) const {
        if(task && (! tensor_.probe())) {
          task->inc();
          tensor_->register_callback(task);
        }
      }

//      /// Future accessor
//
//      /// \return A const reference to the tensor future
//      const future& get_future() const { return tensor_; }
//
//      /// Future accessor
//
//      /// \return A reference to the tensor future
//      future& get_future() { return tensor_; }
//
//      operator future&() { return tensor_; }
//
//      operator const future&() const { return tensor_; }

      /// Check if the tensor future has been evaluated.

      /// \return \c true if the future has been evaluated, \c false otherwise.
      bool probe() const { return tensor_.probe(); }

    private:
      future tensor_; ///< The future of the tensor
    };


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_FUTURE_TENSOR_H__INCLUDED
