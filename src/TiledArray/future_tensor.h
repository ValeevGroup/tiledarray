#ifndef TILEDARRAY_FUTURE_TENSOR_H__INCLUDED
#define TILEDARRAY_FUTURE_TENSOR_H__INCLUDED

#include <TiledArray/tensor_base.h>
#include <world/worldtask.h> // for TaskInterface and Future

namespace TiledArray {
  namespace expressions {

    template <typename> class FutureTensor;

    template <typename T>
    struct TensorTraits<FutureTensor<T> > {
      typedef typename T::size_type size_type;
      typedef typename T::size_array size_array;
      typedef typename T::value_type value_type;
      typedef typename T::const_iterator const_iterator;
      typedef typename T::const_reference const_reference;
      typedef typename T::const_pointer const_pointer;
      typedef typename T::difference_type difference_type;
    }; // struct TensorTraits<FutureTensor<T> >

    template <typename T>
    struct Eval<FutureTensor<T> > {
      typedef typename Eval<T>::type type;
    }; // struct Eval<FutureTensor<T> >

    template <typename T>
    class FutureTensor : public DirectReadableTensor<FutureTensor<T> > {
    public:
      typedef FutureTensor<T> FutureTensor_;
      TILEDARRAY_DIRECT_READABLE_TENSOR_INHEIRATE_TYPEDEF(DirectReadableTensor<FutureTensor_>, FutureTensor_);
      typedef T tensor_type;
      typedef madness::Future<tensor_type> future;
      typedef typename tensor_type::storage_type storage_type; /// The storage type for this object

      FutureTensor(const future& f) :
        tensor_(f)
      { }

      FutureTensor(const FutureTensor<T>& other) :
        tensor_(other.tensor_)
      { }

      // dimension information
      unsigned int dim() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().dim();
      }

      TiledArray::detail::DimensionOrderType order() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().order();
      }

      const size_array& size() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().size();
      }

      size_type volume() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().volume();
      }

      typename Eval<tensor_type>::type eval() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().eval();
      }

      template<typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().eval_to(dest);
      }

      // element access
      const_reference operator[](size_type i) const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get()[i];
      }

      // iterator factory
      const_iterator begin() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().begin();
      }

      const_iterator end() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().end();
      }

      const_pointer data() const {
        TA_ASSERT(tensor_.probe());
        return tensor_.get().data();
      }

      void check_dependency(madness::TaskInterface* task) const {
        TA_ASSERT(task);
        if(! tensor_.probe()) {
          task->inc();
          tensor_->register_callback(task);
        }
      }

      const future& get_future() const { return tensor_; }

      future& get_future() { return tensor_; }

    private:
      future tensor_;
    };


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_FUTURE_TENSOR_H__INCLUDED
