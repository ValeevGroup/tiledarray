#ifndef TILEDARRAY_EVAL_TASK_H__INCLUDED
#define TILEDARRAY_EVAL_TASK_H__INCLUDED

#include <world/worldtask.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace expressions {
    namespace detail {


      template <typename Res>
      class TensorTask : public madness::TaskInterface {
      public:
        typedef TensorTask<Res> TensorTask_;

        using madness::TaskInterface::get_world;

        TensorTask(int ndep = 0) :
          madness::TaskInterface(ndep), res_(new Res()), ref_()
        { }

        const std::shared_ptr<Res>& result() const { return res_; }

      protected:
        madness::Future<Res> res_;
      };

      template <typename Arg>
      class UnaryTensorTask : public TensorTask<Tensor<typename Arg::value_type, typename Arg::range_type> > {
      public:
        typedef UnaryTensorTask<Arg> UnaryTensorTask_;
        typedef Arg argument_type;
        typedef Tensor<typename argument_type::value_type,
            typename argument_type::range_type> result_type;
        typedef TensorTask<result_type> base;

        template <typename T>
        UnaryTensorTask(const T& tile) :
          base(1), arg_(tile.result())
        {
          tile.register_callback(this);
        }

      protected:
        std::shared_ptr<const argument_type> arg_;
      };

      template <typename Left, typename Right>
      class BinaryTensorTask : public TensorTask<Tensor<typename Left::value_type, typename Left::range_type> > {
      public:
        typedef BinaryTensorTask<Arg> BinaryTensorTask_;
        typedef Left left_argument_type;
        typedef Right right_argument_type;
        typedef Tensor<typename left_argument_type::value_type,
            typename left_argument_type::range_type> result_type;
        typedef TensorTask<result_type> base;

        template <typename LT, typename RT>
        BinaryTensorTask(LT& ltile, RT& rtile) :
          base(2), left_(ltile.result()), right_(rtile.result())
        {
          ltile.register_callback(this);
          rtile.register_callback(this);
        }

        using base::result;

      protected:
        std::shared_ptr<const left_argument_type> left_;
        std::shared_ptr<const right_argument_type> right_;
      };

      template <typename Arg, typename Op>
      class ReductionTensorTask : public TensorTask<Tensor<typename Arg::value_type, typename Arg::range_type> > {
      public:
        typedef ReductionTensorTask<Arg> ReductionTensorTask_;
        typedef Arg argument_type;
        typedef Tensor<typename argument_type::value_type,
            typename argument_type::range_type> result_type;
        typedef TensorTask<result_type> base;

        template <typename T>
        ReductionTensorTask(const T& args) :
          base(args.size()), args_(), op_(result())
        {
          for(typename T::const_iterator it = args.begin(); it != args.end(); ++it) {
            args_.push_back(it->result());
            it->register_callback(this);
          }
        }

        using base::result;

        virtual void run(madness::World& w) {
          typedef typename std::vector<std::shared_ptr<const argument_type> >::const_iterator iterator;

        }

      protected:
        std::vector<std::shared_ptr<const argument_type> > args_;
        Op op_;
      };



    } // namespace detail
  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EVAL_TASK_H__INCLUDED
