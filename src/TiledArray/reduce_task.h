#ifndef TILEDARRAY_REDUCE_TASK_H__INCLUDED
#define TILEDARRAY_REDUCE_TASK_H__INCLUDED

#include <world/worldfut.h>
#include <world/worldrange.h>
#include <list>

namespace TiledArray {
  namespace detail {

    template <typename T, typename Op>
    class ReduceTaskImpl {
    public:
      typedef ReduceTaskImpl<T, Op> ReduceTaskImpl_;  ///< This object type
      typedef T value_type;                           ///< The type to be reduced
      typedef madness::Future<value_type> data_type;  ///< The data type that is added
      typedef std::list<data_type> container_type;    ///< The reduction object container
      typedef typename container_type::const_iterator iterator; ///< Reduction iterator
      typedef Op op_type;                             ///< The reduction function type

    private:
      // Not allowed
      ReduceTaskImpl(const ReduceTaskImpl_&);
      ReduceTaskImpl_& operator=(const ReduceTaskImpl_&);

    public:

      /// Constructor

      /// \param op The reduction operation
      ReduceTaskImpl(const op_type& op) :
          data_(), op_(op)
      { }

      /// Add an element to the reduction

      /// \param data The element to be added
      void add(const data_type& data) {
        data_.push_back(data);
      }

      /// Add an element to the reduction

      /// \param chunk The chunk size of the reduction operation
      /// \return An iterator range for reduction task
      Range<typename std::list<data_type>::const_iterator>
      range(int chunk) const {
        return Range<typename std::vector<data_type>::const_iterator>(data_.begin(),
            data_.end(), chunk);
      }

      std::size_t size() const { return data_.size(); }

      /// Reduction operation function
      value_type reduce(const value_type& left, const value_type& right) const {
        return op_(left, right);
      }

    private:
      std::list<data_type> data_;   ///< The data to be reduced
      op_type op_;                  ///< The reduction operation
    }; // class ReduceTaskImpl

    template <typename T, typename Op>
    class ReduceTask {
    public:
      typedef ReduceTask<T, Op> ReduceTask_;          ///< This object type
      typedef T value_type;                           ///< The type to be reduced
      typedef madness::Future<value_type> data_type;  ///< The data type that is added
      typedef Op op_type;                             ///< The reduction function type

      /// Constructor

      /// \param op The reduction operation
      ReduceTask(const op_type& op = op_type()) :
          pimpl_(new ReduceTaskImpl<value_type, op_type>(op))
      { }

      /// Copy constructor
      ReduceTask(const ReduceTask_& other) :
          pimpl_(other.pimpl_)
      { }

      ReduceTask_& operator=(const ReduceTask_& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Add an element to the reduction

      /// \param data The element to be added
      void add(const data_type& data) {
        pimpl_->add(data);
      }

      /// Add an element to the reduction

      /// \param chunk The chunk size of the reduction operation
      /// \return An iterator range for reduction task
      Range<typename std::list<data_type>::const_iterator>
      range(int chunk = 1) const {
        return pimpl_->range(chunk);
      }

      /// Return the number of object in the reduction.
      std::size_t size() const { return pimpl_->size(); }

      /// Reduction operation function

      /// \param left The left object to reduce
      /// \param right The right object to reduce
      /// \return The reduction of left and right
      value_type operator()(const value_type& left, const value_type& right) const {
        return pimpl_->reduce(left, right);
      }

    private:
      std::shared_ptr<ReduceTaskImpl<value_type, op_type> > pimpl_; ///< The implementation object
    }; // class ReduceTaskImpl

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REDUCE_TASK_H__INCLUDED
