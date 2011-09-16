#ifndef TILEDARRAY_EVAL_TASK_H__INCLUDED
#define TILEDARRAY_EVAL_TASK_H__INCLUDED

#include <world/worldtask.h>

namespace TiledArray {
  namespace expressions {
    namespace detail {

      template <typename Res, typename Arg>
      class EvalTask : public madness::TaskInterface {
      public:
        EvalTask(const typename Res::range_type& range, const Arg& arg) :
          res_(), range_(range), arg_(arg)
        {
          arg_.check_dependencies(this);
        }

        virtual void run(madness::World&) {
          res_.set(Res(range_, arg_.begin()));
        }

        const madness::Future<Res>& result() const { return res_; }

      private:
        madness::Future<Res> res_;
        typename Res::range_type range_;
        Arg arg_;
      };

      /// Task generator for tensor eval_to functions

      /// \tparam Dest The destination tiled tensor type
      /// \tparam It The iterator for input tile types
      template <typename Dest, typename It>
      class EvalTo {
      private:
        typedef EvalTask<typename Dest::value_type,
            typename std::iterator_traits<It>::value_type> eval_task;
                                                    ///< The evaluation task type

      public:
        /// Constructor

        /// \param world The world which will evaluate the tasks.
        /// \param dest The destination tensor
        EvalTo(madness::World& world, Dest& dest) :
          world_(world), dest_(dest)
        { }

        /// Generate the tile evaluation task

        /// \param it The input tensor for the tile
        /// \return \c true if the task was successfully submitted
        bool operator()(It it) {
          eval_task* task = new eval_task(dest_.tiling().make_tile_range(it.index()), *it);
          try {
            dest_.set(it.index(), task->result());
            world_.taskq.add(task);
          } catch(...) {
            delete task;
            throw;
          }

          return true;
        }

      private:
        madness::World& world_;
        Dest& dest_;
      }; // class EvalTo

    } // namespace detail
  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EVAL_TASK_H__INCLUDED
