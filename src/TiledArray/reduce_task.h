#ifndef TILEDARRAY_REDUCE_TASK_H__INCLUDED
#define TILEDARRAY_REDUCE_TASK_H__INCLUDED

#include <world/worldfut.h>
#include <world/worldrange.h>
#include <world/make_task.h>
#include <list>

namespace TiledArray {
  namespace detail {

    template <typename T, typename Op>
    class ReduceTaskImpl {
    public:
      typedef T value_type;
      typedef std::list<madness::Future<value_type> > container_type;
      typedef madness::Range<typename container_type::iterator> range_type;
      typedef Op op_type;

      ReduceTaskImpl(madness::World& world, const op_type& op) :
          world_(world), list_(), op_(op)
      { }

      void add(madness::Future<value_type> f) {
        list_.push_back(f);
      }

      std::size_t size() const { return list_.size(); }

      range_type range(const int chunk) const {
        return range_type(list_.begin(), list_.end(), chunk);
      }

      value_type reduce(const value_type& value1, const value_type& value2) const {
        return op_(value1, value2);
      }

      madness::World& get_world() const { return world_; }

    private:
      madness::World& world_;
      mutable container_type list_;
      op_type op_;

    }; // class ReduceTaskImpl

    template <typename T, typename Op>
    class ReduceTask {
    private:
      typedef ReduceTaskImpl<T, Op> impl_type;
      typedef ReduceTask<T, Op> ReduceTask_;

    public:
      typedef typename impl_type::value_type value_type;
      typedef typename impl_type::range_type range_type;
      typedef typename impl_type::op_type op_type;

      typedef value_type result_type;

      ReduceTask(madness::World& world, const op_type& op) :
          pimpl_(new impl_type(world, op))
      { }

      ReduceTask(const ReduceTask_& other) :
          pimpl_(other.pimpl_)
      { }

      ReduceTask_& operator=(const ReduceTask_& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      void add(const madness::Future<value_type>& f) {
        pimpl_->add(f);
      }

      void add(const value_type& value) {
        add(madness::Future<value_type>(value));
      }

      std::size_t size() const { return pimpl_->size(); }

      madness::Future<value_type> operator()() const {
        return (*this)(pimpl_->range(8));
      }

      value_type operator()(const value_type& value1, const value_type& value2) const {
        return pimpl_->reduce(value1, value2);
      }

      madness::Future<value_type> operator()(const range_type& range) const {
        if (range.size() <= range.get_chunksize()) {
          value_type result = value_type();
          for(typename range_type::iterator it = range.begin(); it != range.end(); ++it) {
            TA_ASSERT(it->probe(), std::runtime_error, "Future is not ready.");
            result = (*this)(result, *it);
          }
          return madness::Future<value_type>(result);
        } else {
          range_type left = range;
          range_type right(left, madness::Split());

          madness::Future<value_type> left_red = make_task(left);
          madness::Future<value_type> right_red = make_task(right);
          return pimpl_->get_world().taskq.add(madness::make_task(*this, left_red, right_red));
        }
      }

    private:

      madness::Future<value_type> make_task(range_type range) const {
        madness::TaskFn<ReduceTask, range_type>* task = madness::make_task(*this, range);

        if(range.size() <= range.get_chunksize()) {
          for(typename range_type::iterator it = range.begin(); it != range.end(); ++it) {
            if(! (it->probe())) {
                task->inc();
                it->register_callback(task);
            }
          }
        }

        return pimpl_->get_world().taskq.add(task);
      }

      std::shared_ptr<impl_type> pimpl_;

    }; // class ReduceTask

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REDUCE_TASK_H__INCLUDED
