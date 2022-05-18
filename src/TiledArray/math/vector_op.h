/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  vector_op.h
 *  Nov 17, 2013
 *
 */

#ifndef TILEDARRAY_MATH_VECTOR_OP_H__INCLUDED
#define TILEDARRAY_MATH_VECTOR_OP_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/external/madness.h>

#if HAVE_INTEL_TBB
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/tbb_stddef.h>
#endif

#include <TiledArray/type_traits.h>

#define TILEDARRAY_LOOP_UNWIND ::TiledArray::math::LoopUnwind::value

namespace TiledArray {
namespace math {

// Import param_type into
using ::TiledArray::detail::param_type;

// Define compile time constant for loop unwinding.
typedef std::integral_constant<std::size_t,
                               TILEDARRAY_CACHELINE_SIZE / sizeof(double)>
    LoopUnwind;
typedef std::integral_constant<std::size_t,
                               ~std::size_t(TILEDARRAY_LOOP_UNWIND - 1ul)>
    index_mask;

template <std::size_t>
struct VectorOpUnwind;

/// Vector loop unwind helper class

/// This object will unwind \c 1 step of a vector operation loop, and
/// terminate the loop
template <>
struct VectorOpUnwind<0ul> {
  static constexpr std::size_t offset = TILEDARRAY_LOOP_UNWIND - 1ul;

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void for_each(
      Op&& op, Result* MADNESS_RESTRICT const result,
      const Args* MADNESS_RESTRICT const... args) {
    op(result[offset], args[offset]...);
  }

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void for_each_ptr(
      Op&& op, Result* MADNESS_RESTRICT const result,
      const Args* MADNESS_RESTRICT const... args) {
    op(result + offset, args[offset]...);
  }

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void reduce(
      Op&& op, Result& MADNESS_RESTRICT result,
      const Args* MADNESS_RESTRICT const... args) {
    op(result, args[offset]...);
  }

  template <typename Result, typename Arg>
  static TILEDARRAY_FORCE_INLINE void scatter(
      Result* MADNESS_RESTRICT const result,
      const Arg* MADNESS_RESTRICT const arg,
      const std::size_t /*result_stride*/) {
    *result = arg[offset];
  }

  template <typename Result, typename Arg>
  static TILEDARRAY_FORCE_INLINE void gather(
      Result* MADNESS_RESTRICT const result,
      const Arg* MADNESS_RESTRICT const arg, std::size_t /*arg_stride*/) {
    result[offset] = *arg;
  }

};  //  struct VectorOpUnwind

/// Vector loop unwind helper class

/// This object will unwind \c N steps of a vector operation loop.
/// \tparam N The number of steps to unwind
template <std::size_t N>
struct VectorOpUnwind : public VectorOpUnwind<N - 1ul> {
  typedef VectorOpUnwind<N - 1ul> VectorOpUnwindN1;

  static constexpr std::size_t offset = TILEDARRAY_LOOP_UNWIND - N - 1ul;

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void for_each(
      Op&& op, Result* MADNESS_RESTRICT const result,
      const Args* MADNESS_RESTRICT const... args) {
    op(result[offset], args[offset]...);
    VectorOpUnwindN1::for_each(op, result, args...);
  }

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void for_each_ptr(
      Op&& op, Result* MADNESS_RESTRICT const result,
      const Args* MADNESS_RESTRICT const... args) {
    op(result + offset, args[offset]...);
    VectorOpUnwindN1::for_each_ptr(op, result, args...);
  }

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void reduce(
      Op&& op, Result& MADNESS_RESTRICT result,
      const Args* MADNESS_RESTRICT const... args) {
    op(result, args[offset]...);
    VectorOpUnwindN1::reduce(op, result, args...);
  }

  template <typename Result, typename Arg>
  static TILEDARRAY_FORCE_INLINE void scatter(
      Result* MADNESS_RESTRICT const result,
      const Arg* MADNESS_RESTRICT const arg, const std::size_t result_stride) {
    *result = arg[offset];
    VectorOpUnwindN1::scatter(result + result_stride, arg, result_stride);
  }

  template <typename Result, typename Arg>
  static TILEDARRAY_FORCE_INLINE void gather(
      Result* MADNESS_RESTRICT const result,
      const Arg* MADNESS_RESTRICT const arg, std::size_t arg_stride) {
    result[offset] = *arg;
    VectorOpUnwindN1::gather(result, arg + arg_stride, arg_stride);
  }

};  //  struct VectorOpUnwind

typedef VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1> VecOpUnwindN;
template <typename>
class Block;

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE void for_each_block(Op&& op, Result* const result,
                                            const Args* const... args) {
  VecOpUnwindN::for_each(op, result, args...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE void for_each_block(Op&& op, Block<Result>& result,
                                            Block<Args>&&... args) {
  VecOpUnwindN::for_each(op, result.data(), args.data()...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE void for_each_block_n(
    Op&& op, const std::size_t n, Result* MADNESS_RESTRICT const result,
    const Args* MADNESS_RESTRICT const... args) {
  for (std::size_t i = 0ul; i < n; ++i) op(result[i], args[i]...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE typename std::enable_if<(sizeof...(Args) >= 0)>::type
for_each_block_ptr(Op&& op, Result* const result, const Args* const... args) {
  VecOpUnwindN::for_each_ptr(op, result, args...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE typename std::enable_if<(sizeof...(Args) > 0)>::type
for_each_block_ptr(Op&& op, Result* const result, Block<Args>&&... args) {
  VecOpUnwindN::for_each_ptr(op, result, args.data()...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE void for_each_block_ptr_n(
    Op&& op, const std::size_t n, Result* MADNESS_RESTRICT const result,
    const Args* MADNESS_RESTRICT const... args) {
  for (std::size_t i = 0ul; i < n; ++i) op(result + i, args[i]...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE void reduce_block(Op&& op, Result& result,
                                          const Args* const... args) {
  VecOpUnwindN::reduce(op, result, args...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE void reduce_block(Op&& op, Result& result,
                                          Block<Args>&&... args) {
  VecOpUnwindN::reduce(op, result, args.data()...);
}

template <typename Op, typename Result, typename... Args>
TILEDARRAY_FORCE_INLINE void reduce_block_n(
    Op&& op, const std::size_t n, Result& MADNESS_RESTRICT result,
    const Args* MADNESS_RESTRICT const... args) {
  for (std::size_t i = 0ul; i < n; ++i) op(result, args[i]...);
}

template <typename Result, typename Arg>
TILEDARRAY_FORCE_INLINE void copy_block(Result* const result,
                                        const Arg* const arg) {
  for_each_block([](Result& lhs, param_type<Arg> rhs) { lhs = rhs; }, result,
                 arg);
}

template <typename Arg, typename Result>
TILEDARRAY_FORCE_INLINE void copy_block_n(std::size_t n, Result* const result,
                                          const Arg* const arg) {
  for_each_block_n([](Result& lhs, param_type<Arg> rhs) { lhs = rhs; }, n,
                   result, arg);
}

template <typename Arg, typename Result>
TILEDARRAY_FORCE_INLINE void scatter_block(Result* const result,
                                           const std::size_t stride,
                                           const Arg* const arg) {
  VecOpUnwindN::scatter(result, arg, stride);
}

template <typename Result, typename Arg>
TILEDARRAY_FORCE_INLINE void scatter_block_n(const std::size_t n,
                                             Result* result,
                                             const std::size_t stride,
                                             const Arg* const arg) {
  for (std::size_t i = 0; i < n; ++i, result += stride) *result = arg[i];
}

template <typename Result, typename Arg>
TILEDARRAY_FORCE_INLINE void gather_block(Result* const result,
                                          const Arg* const arg,
                                          const std::size_t stride) {
  VecOpUnwindN::gather(result, arg, stride);
}

template <typename Arg, typename Result>
TILEDARRAY_FORCE_INLINE void gather_block_n(const std::size_t n,
                                            Result* const result,
                                            const Arg* const arg,
                                            const std::size_t stride) {
  for (std::size_t i = 0; i < n; ++i, arg += stride) result[i] = *arg;
}

template <typename T>
class Block {
  TILEDARRAY_ALIGNED_STORAGE T block_[TILEDARRAY_LOOP_UNWIND];

 public:
  Block() {}
  explicit Block(const T* const data) { load(data); }

  void load(const T* const data) { copy_block(block_, data); }

  void store(T* const data) const { copy_block(data, block_); }

  Block<T>& gather(const T* const data, const std::size_t stride) {
    gather_block(block_, data, stride);
    return *this;
  }

  void scatter_to(T* const data, std::size_t stride) const {
    scatter_block(data, stride, block_);
  }

  T* data() { return block_; }
  const T* data() const { return block_; }

};  // class Block

#ifdef HAVE_INTEL_TBB

struct SizeTRange {
  static constexpr std::size_t block_size = TILEDARRAY_LOOP_UNWIND;

  // GRAIN_SIZE is set to 8 to trigger TiledArray Unit Test
  // in reality, partition is controled by tbb::auto_partitioner instead of
  // GRAIN_SIZE
  static constexpr std::size_t GRAIN_SIZE = 8ul;

  size_t lower;
  size_t upper;

  SizeTRange(const size_t start, const size_t end) : lower(start), upper(end) {}

  //      SizeTRange(const size_t n, const size_t g_size)
  //              : lower(0), upper(n - 1), grain_size(g_size) { }

  SizeTRange() = default;
  SizeTRange(const SizeTRange& r) = default;

  ~SizeTRange() {}

  //      void set_grain_size(std::size_t grain_size){
  //        GRAIN_SIZE = grain_size;
  //      }

  bool empty() const { return lower > upper; }

  bool is_divisible() const { return size() >= 2 * GRAIN_SIZE; }

  size_t begin() const { return lower; }

  size_t end() const { return upper; }

  size_t size() const { return upper - lower; }

  SizeTRange(SizeTRange& r, tbb::split) {
    size_t nblock = (r.upper - r.lower) / block_size;
    nblock = (nblock + 1) / 2;
    lower = r.lower + nblock * block_size;
    upper = r.upper;
    r.upper = lower;
  }
};

//    SizeTRange::set_grain_size(1024ul);

#endif

template <typename Op, typename Result, typename... Args,
          std::enable_if_t<std::is_void_v<
              std::invoke_result_t<Op, Result&, Args...>>>* = nullptr>
void inplace_vector_op_serial(Op&& op, const std::size_t n,
                              Result* const result, const Args* const... args) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  constexpr std::size_t index_mask = ~std::size_t(TILEDARRAY_LOOP_UNWIND - 1ul);
  const std::size_t nx = n & index_mask;

  for (; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
    Block<Result> result_block(result + i);
    for_each_block(op, result_block, Block<Args>(args + i)...);
    result_block.store(result + i);
  }

  for_each_block_n(op, n - i, result + i, (args + i)...);
}

#ifdef HAVE_INTEL_TBB

template <typename Op, typename Result, typename... Args>
class ApplyInplaceVectorOp {
 public:
  ApplyInplaceVectorOp(Op& op, Result* const result, const Args* const... args)
      : op_(op), result_(result), args_(args...) {}

  ~ApplyInplaceVectorOp() {}

  template <std::size_t... Is>
  void helper(SizeTRange& range, const std::index_sequence<Is...>&) const {
    std::size_t offset = range.begin();
    std::size_t n_range = range.size();
    inplace_vector_op_serial(op_, n_range, result_ + offset,
                             (std::get<Is>(args_) + offset)...);
  }

  void operator()(SizeTRange& range) const {
    helper(range, std::make_index_sequence<sizeof...(Args)>());
  }

 private:
  Op& op_;
  Result* const result_;
  std::tuple<const Args* const...> args_;
};

#endif

template <typename Op, typename Result, typename... Args,
          std::enable_if_t<std::is_void_v<
              std::invoke_result_t<Op, Result&, Args...>>>* = nullptr>
void inplace_vector_op(Op&& op, const std::size_t n, Result* const result,
                       const Args* const... args) {
#ifdef HAVE_INTEL_TBB
  //        std::cout << "INPLACE_TBB_VECTOR_OP" << std::endl;
  SizeTRange range(0, n);

  // if support lambda variadic
  //      auto apply_inplace_vector_op = [op, result, args...](SizeTRange
  //      &range) {
  //          size_t offset = range.begin();
  //          size_t n_range = range.size();
  //          inplace_vector_op_serial(op, n_range, result + offset, (args +
  //          offset)...);
  //        };
  // else
  auto apply_inplace_vector_op =
      ApplyInplaceVectorOp<Op, Result, Args...>(op, result, args...);

  tbb::parallel_for(range, apply_inplace_vector_op, tbb::auto_partitioner());
#else
  inplace_vector_op_serial(op, n, result, args...);
#endif
}

template <typename Op, typename Result, typename... Args,
          std::enable_if_t<
              !std::is_void_v<std::invoke_result_t<Op, Args...>>>* = nullptr>
void vector_op_serial(Op&& op, const std::size_t n, Result* const result,
                      const Args* const... args) {
  auto wrapper_op = [&op](Result& res, param_type<Args>... a) {
    res = op(a...);
  };

  std::size_t i = 0ul;

  // Compute block iteration limit
  constexpr std::size_t index_mask = ~std::size_t(TILEDARRAY_LOOP_UNWIND - 1ul);
  const std::size_t nx = n & index_mask;

  for (; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
    Block<Result> result_block;
    for_each_block(wrapper_op, result_block, Block<Args>(args + i)...);
    result_block.store(result + i);
  }

  for_each_block_n(wrapper_op, n - i, result + i, (args + i)...);
}

#ifdef HAVE_INTEL_TBB

template <typename Op, typename Result, typename... Args>
class ApplyVectorOp {
 public:
  ApplyVectorOp(Op& op, Result* const result, const Args* const... args)
      : op_(op), result_(result), args_(args...) {}

  ~ApplyVectorOp() {}

  template <std::size_t... Is>
  void helper(SizeTRange& range, const std::index_sequence<Is...>&) const {
    std::size_t offset = range.begin();
    std::size_t n_range = range.size();
    vector_op_serial(op_, n_range, result_ + offset,
                     (std::get<Is>(args_) + offset)...);
  }

  void operator()(SizeTRange& range) const {
    helper(range, std::make_index_sequence<sizeof...(Args)>());
  }

 private:
  Op& op_;
  Result* const result_;
  std::tuple<const Args* const...> args_;
};

#endif

template <typename Op, typename Result, typename... Args,
          std::enable_if_t<
              !std::is_void_v<std::invoke_result_t<Op, Args...>>>* = nullptr>
void vector_op(Op&& op, const std::size_t n, Result* const result,
               const Args* const... args) {
#ifdef HAVE_INTEL_TBB
  //        std::cout << "TBB_VECTOR_OP" << std::endl;
  SizeTRange range(0, n);

  // if support lambda variadic
  //        auto apply_vector_op = [op, result, args...](SizeTRange &range) {
  //          size_t offset = range.begin();
  //          size_t n_range = range.size();
  //          vector_op_serial(op, n_range, result + offset, (args +
  //          offset)...);
  //        };
  // else
  auto apply_vector_op =
      ApplyVectorOp<Op, Result, Args...>(op, result, args...);

  tbb::parallel_for(range, apply_vector_op, tbb::auto_partitioner());
#else
  vector_op_serial(op, n, result, args...);
#endif
}

template <typename Op, typename Result, typename... Args>
void vector_ptr_op_serial(Op&& op, const std::size_t n, Result* const result,
                          const Args* const... args) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  constexpr std::size_t index_mask = ~std::size_t(TILEDARRAY_LOOP_UNWIND - 1ul);
  const std::size_t nx = n & index_mask;

  for (; i < nx; i += TILEDARRAY_LOOP_UNWIND)
    for_each_block_ptr(op, result + i, Block<Args>(args + i)...);
  for_each_block_ptr_n(op, n - i, result + i, (args + i)...);
}

#ifdef HAVE_INTEL_TBB
template <typename Op, typename Result, typename... Args>
class ApplyVectorPtrOp {
 public:
  ApplyVectorPtrOp(Op& op, Result* const result, const Args* const... args)
      : op_(op), result_(result), args_(args...) {}

  ~ApplyVectorPtrOp() {}

  template <std::size_t... Is>
  void helper(SizeTRange& range, const std::index_sequence<Is...>&) const {
    std::size_t offset = range.begin();
    std::size_t n_range = range.size();
    vector_ptr_op_serial(op_, n_range, result_ + offset,
                         (std::get<Is>(args_) + offset)...);
  }

  void operator()(SizeTRange& range) const {
    helper(range, std::make_index_sequence<sizeof...(Args)>());
  }

 private:
  Op& op_;
  Result* const result_;
  std::tuple<const Args* const...> args_;
};
#endif

template <typename Op, typename Result, typename... Args>
void vector_ptr_op(Op&& op, const std::size_t n, Result* const result,
                   const Args* const... args) {
#ifdef HAVE_INTEL_TBB
  //        std::cout << "TBB_VECTOR_PTR_OP" << std::endl;
  SizeTRange range(0, n);

  // if support lambda variadic
  //        auto apply_vector_ptr_op = [op, result, args...](SizeTRange &range)
  //        {
  //          size_t offset = range.begin();
  //          size_t n_range = range.size();
  //          vector_ptr_op_serial(op, n_range, result + offset, (args +
  //          offset)...);
  //        };
  // else
  auto apply_vector_ptr_op =
      ApplyVectorPtrOp<Op, Result, Args...>(op, result, args...);
  tbb::parallel_for(range, apply_vector_ptr_op, tbb::auto_partitioner());
#else
  vector_ptr_op_serial(op, n, result, args...);
#endif
}

template <typename Op, typename Result, typename... Args>
void reduce_op_serial(Op&& op, const std::size_t n, Result& result,
                      const Args* const... args) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  constexpr std::size_t index_mask = ~std::size_t(TILEDARRAY_LOOP_UNWIND - 1ul);
  const std::size_t nx = n & index_mask;

  for (; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
    Result temp = result;
    reduce_block(op, temp, Block<Args>(args + i)...);
    result = temp;
  }

  reduce_block_n(op, n - i, result, (args + i)...);
}

#ifdef HAVE_INTEL_TBB
/// Helper class for composing TBB parallel reductions. Meets the \c Body
/// concept used for the imperative form of \c tbb::parallel_reduce .
template <typename ReduceOp, typename JoinOp, typename Result, typename... Args>
class ApplyReduceOp {
 public:
  ApplyReduceOp(ReduceOp& reduce_op, JoinOp& join_op, const Result& identity,
                const Result& result, const Args* const... args)
      : reduce_op_(reduce_op),
        join_op_(join_op),
        identity_(identity),
        result_(result),
        args_(args...) {}

  ApplyReduceOp(ApplyReduceOp& rhs, tbb::split)
      : reduce_op_(rhs.reduce_op_),
        join_op_(rhs.join_op_),
        identity_(rhs.identity_),
        result_(rhs.identity_),
        args_(rhs.args_) {}

  ~ApplyReduceOp() {}

  template <std::size_t... Is>
  void helper(SizeTRange& range, const std::index_sequence<Is...>&) {
    std::size_t offset = range.begin();
    std::size_t n_range = range.size();
    reduce_op_serial(reduce_op_, n_range, result_,
                     (std::get<Is>(args_) + offset)...);
  }

  void operator()(SizeTRange& range) {
    helper(range, std::make_index_sequence<sizeof...(Args)>());
  }

  void join(const ApplyReduceOp& rhs) { join_op_(result_, rhs.result_); }

  const Result result() const { return result_; }

 private:
  ReduceOp& reduce_op_;
  JoinOp& join_op_;
  const Result identity_;
  Result result_;
  std::tuple<const Args* const...> args_;
};
#endif

template <typename ReduceOp, typename JoinOp, typename Result, typename... Args>
void reduce_op(ReduceOp&& reduce_op, JoinOp&& join_op, const Result& identity,
               const std::size_t n, Result& result, const Args* const... args) {
  // TODO implement reduce operation with TBB
#ifdef HAVE_INTEL_TBB
  SizeTRange range(0, n);

  auto apply_reduce_op = ApplyReduceOp<ReduceOp, JoinOp, Result, Args...>(
      reduce_op, join_op, identity, result, args...);

  tbb::parallel_reduce(range, apply_reduce_op, tbb::auto_partitioner());

  result = apply_reduce_op.result();
#else
  reduce_op_serial(reduce_op, n, result, args...);
#endif
}

template <typename Arg, typename Result>
typename std::enable_if<!(std::is_same<Arg, Result>::value &&
                          detail::is_scalar_v<Arg>)>::type
copy_vector(const std::size_t n, const Arg* const arg, Result* const result) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  constexpr std::size_t index_mask = ~std::size_t(TILEDARRAY_LOOP_UNWIND - 1ul);
  const std::size_t nx = n & index_mask;

  for (; i < nx; i += TILEDARRAY_LOOP_UNWIND) copy_block(result + i, arg + i);
  copy_block_n(n - i, result + i, arg + i);
}

template <typename T>
inline typename std::enable_if<detail::is_scalar_v<T>>::type copy_vector(
    const std::size_t n, const T* const arg, T* const result) {
  std::memcpy(result, arg, n * sizeof(T));
}

template <typename Arg, typename Result>
void fill_vector(const std::size_t n, const Arg& arg, Result* const result) {
  auto fill_op = [arg](Result& res) { res = arg; };
  vector_op(fill_op, n, result);
}

template <typename Arg, typename Result>
typename std::enable_if<!(detail::is_scalar_v<Arg> &&
                          detail::is_scalar_v<Result>)>::type
uninitialized_copy_vector(const std::size_t n, const Arg* const arg,
                          Result* const result) {
  auto op = [](Result* const res, param_type<Arg> a) { new (res) Result(a); };
  vector_ptr_op(op, n, result, arg);
}

template <typename Arg, typename Result>
inline typename std::enable_if<detail::is_scalar_v<Arg> &&
                               detail::is_scalar_v<Result>>::type
uninitialized_copy_vector(const std::size_t n, const Arg* const arg,
                          Result* const result) {
  copy_vector(n, arg, result);
}

template <typename Arg, typename Result>
typename std::enable_if<!(detail::is_scalar_v<Arg> &&
                          detail::is_scalar_v<Result>)>::type
uninitialized_fill_vector(const std::size_t n, const Arg& arg,
                          Result* const result) {
  auto op = [arg](Result* const res) { new (res) Result(arg); };
  vector_ptr_op(op, n, result);
}

template <typename Arg, typename Result>
inline typename std::enable_if<detail::is_scalar_v<Arg> &&
                               detail::is_scalar_v<Result>>::type
uninitialized_fill_vector(const std::size_t n, const Arg& arg,
                          Result* const result) {
  fill_vector(n, arg, result);
}

template <typename Arg>
typename std::enable_if<!detail::is_scalar_v<Arg>>::type destroy_vector(
    const std::size_t n, Arg* const arg) {
  auto op = [](Arg* const a) { a->~Arg(); };
  vector_ptr_op(op, n, arg);
}

template <typename Arg>
inline typename std::enable_if<detail::is_scalar_v<Arg>>::type destroy_vector(
    const std::size_t, const Arg* const) {}

template <typename Arg, typename Result, typename Op>
typename std::enable_if<!(detail::is_scalar_v<Arg> &&
                          detail::is_scalar_v<Result>)>::type
uninitialized_unary_vector_op(const std::size_t n, const Arg* const arg,
                              Result* const result, Op&& op) {
  auto wrapper_op = [&op](Result* const res, param_type<Arg> a) {
    new (res) Result(op(a));
  };
  vector_ptr_op(wrapper_op, n, result, arg);
}

template <typename Arg, typename Result, typename Op>
inline typename std::enable_if<detail::is_scalar_v<Arg> &&
                               detail::is_scalar_v<Result>>::type
uninitialized_unary_vector_op(const std::size_t n, const Arg* const arg,
                              Result* const result, Op&& op) {
  vector_op(op, n, result, arg);
}

template <typename Left, typename Right, typename Result, typename Op>
typename std::enable_if<!(detail::is_scalar_v<Left> &&
                          detail::is_scalar_v<Right> &&
                          detail::is_scalar_v<Result>)>::type
uninitialized_binary_vector_op(const std::size_t n, const Left* const left,
                               const Right* const right, Result* const result,
                               Op&& op) {
  auto wrapper_op = [&op](Result* const res, param_type<Left> l,
                          param_type<Right> r) { new (res) Result(op(l, r)); };

  vector_ptr_op(op, n, result, left, right);
}

template <typename Left, typename Right, typename Result, typename Op>
typename std::enable_if<detail::is_scalar_v<Left> &&
                        detail::is_scalar_v<Right> &&
                        detail::is_scalar_v<Result>>::type
uninitialized_binary_vector_op(const std::size_t n, const Left* const left,
                               const Right* const right, Result* const result,
                               Op&& op) {
  vector_op(op, n, result, left, right);
}

}  // namespace math
}  // namespace TiledArray

#endif  // TILEDARRAY_MATH_VECTOR_OP_H__INCLUDED
