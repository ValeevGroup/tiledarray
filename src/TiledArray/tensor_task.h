#ifndef TILEDARRAY_TENSOR_TASK_H__INCLUDED
#define TILEDARRAY_TENSOR_TASK_H__INCLUDED

#include <world/worldtask.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace expressions {
    namespace detail {


      template <typename Res>
      class TensorTask : public madness::TaskInterface {
      public:
        typedef TensorTask<Res> TensorTask_;

        TensorTask(int ndep = 0) :
          madness::TaskInterface(ndep), res_(new Res())
        { }

        const std::shared_ptr<Res>& result() const { return res_; }

      protected:
        madness::Future<Res> res_;
      };

      template <typename Arg, typename Op>
      class UnaryTensorTask : public TensorTask<Tensor<typename Arg::value_type, typename Arg::range_type> > {
      public:
        typedef UnaryTensorTask<Arg, Op> UnaryTensorTask_;
        typedef Arg argument_type;
        typedef Tensor<typename argument_type::value_type,
            typename argument_type::range_type> result_type;
        typedef TensorTask<result_type> base;

        template <typename T>
        UnaryTensorTask(const std::shared_ptr<argument_type>& tile, const Op& op = Op()) :
          base(1), arg_(tile.result()), op_(op)
        {
          arg_.register_callback(this);
        }

        virtual void run(madness::World&) {
          *result_ = op_(*arg_);
          result_->notify();
        }

      protected:

        using base::result_;

        std::shared_ptr<argument_type> arg_;
        Op op_;
      };

      template <typename Left, typename Right, typename Op>
      class BinaryTensorTask : public TensorTask<Tensor<typename Left::value_type, typename Left::range_type> > {
      public:
        typedef BinaryTensorTask<Left, Right, Op> BinaryTensorTask_;
        typedef Left left_argument_type;
        typedef Right right_argument_type;
        typedef Tensor<typename left_argument_type::value_type,
            typename left_argument_type::range_type> result_type;
        typedef TensorTask<result_type> base;

        template <typename LT, typename RT>
        BinaryTensorTask(std::shared_ptr<left_argument_type>& left,
            const std::shared_ptr<right_argument_type>& right, Op op = Op()) :
          base(2), left_(left), right_(right), op_(op)
        {
          left_.register_callback(this);
          right_.register_callback(this);
        }

        virtual void run(madness::World&) {
          *result_ = op_(*left_, *right_);
          result_->notify();
        }

      protected:

        using base::result_;

        std::shared_ptr<const left_argument_type> left_;
        std::shared_ptr<const right_argument_type> right_;
        Op op_;
      };

      template <typename Arg, typename Op>
      class ReductionTensorTask : public TensorTask<Tensor<typename Arg::value_type, typename Arg::range_type> > {
      public:
        typedef ReductionTensorTask<Arg, Op> ReductionTensorTask_;
        typedef Arg argument_type;
        typedef Tensor<typename argument_type::value_type,
            typename argument_type::range_type> result_type;
        typedef TensorTask<result_type> base;

        template <typename T>
        ReductionTensorTask(const T& args, const Op& op = Op()) :
          base(args.size()), args_(args.begin(), args.end()), op_(op)
        {
          for(arg_array::iterator it = args_.begin(); it != args_.end(); ++it)
            it->register_callback(this);
        }

        using base::result;

        virtual void run(madness::World&) {
          for(typename arg_array::const_iterator it = args_.begin(); it != args_.end(); ++it)
            *result_ = op_(*result_, *(*it));
          result_->notify();
        }

      protected:

        using base::result_;

        typedef std::vector<std::shared_ptr<const argument_type> > arg_array;
        arg_array args_;
        Op op_;
      };


    } // namespace detail
  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TASK_H__INCLUDED
