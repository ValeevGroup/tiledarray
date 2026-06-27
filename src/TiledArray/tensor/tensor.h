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
 */

#ifndef TILEDARRAY_TENSOR_TENSOR_H__INCLUDED
#define TILEDARRAY_TENSOR_TENSOR_H__INCLUDED

#include "TiledArray/config.h"

#include "TiledArray/host/env.h"
#include "TiledArray/platform.h"

#include "TiledArray/math/blas.h"
#include "TiledArray/math/gemm_helper.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/complex.h"
#include "TiledArray/tensor/kernels.h"
#include "TiledArray/tile_interface/clone.h"
#include "TiledArray/tile_interface/permute.h"
#include "TiledArray/tile_interface/trace.h"
#include "TiledArray/util/logger.h"
#include "TiledArray/util/ptr_registry.h"

#include <umpire_cxx_allocator.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace TiledArray {

namespace detail {

/// Signals that we can take the trace of a Tensor<T, A> (for numeric \c T)
template <typename T, typename A>
struct TraceIsDefined<Tensor<T, A>, enable_if_numeric_t<T>> : std::true_type {};

template <typename To, typename From,
          typename = std::enable_if_t<
              detail::is_nested_tensor_v<To, detail::remove_cvr_t<From>>>>
To clone_or_cast(From&& f) {
  if constexpr (std::is_same_v<To, detail::remove_cvr_t<From>>)
    return std::forward<From>(f).clone();
  else if constexpr (detail::is_convertible_v<From, To>) {
    return static_cast<To>(std::forward<From>(f));
  } else if constexpr (detail::is_range_v<To> &&
                       detail::is_range_v<detail::remove_cvr_t<From>>) {
    using std::begin;
    using std::data;
    using std::end;

    To t(f.range());
    if constexpr (detail::is_contiguous_tensor_v<detail::remove_cvr_t<From>>) {
      const auto n = f.range().volume();
      if constexpr (detail::is_contiguous_tensor_v<To>) {
        std::copy(data(f), data(f) + n, data(t));
      } else {
        std::copy(data(f), data(f) + n, begin(t));
      }
    } else {
      if constexpr (detail::is_contiguous_tensor_v<To>) {
        std::copy(begin(f), end(f), data(t));
      } else
        std::copy(begin(f), end(f), begin(t));
    }
    return t;
  } else {
    static_assert(
        !std::is_void_v<To>,
        "clone_or_cast<To,From>: could not figure out how to convert From to "
        "To, either overload of a member function of Tensor is missing or From "
        "need to provide a conversion operator to To");
  }
}

/// ---------------------------------------------------------------------------
/// Env-gated timing probe for the strided-GEMM ToT "scale" outer-contraction
/// path (Tensor::gemm, commit 266f0a48): measures how much of the scale work
/// runs on the fast strided BLAS GEMM vs. how much reverts to the per-cell
/// AXPY fallback. Mirrors the ce+e / ce+ce probes in arena_einsum.h: switched
/// on by the SAME env var TA_GEMM_TIMING=1 (single master switch); takes no
/// clock samples and touches no atomics when unset (zero production overhead).
/// Two regimes are tracked separately:
///   [0] tot_x_t: left ToT x plain scalar, "m,k;a * k,n -> m,n;a" (per-row m)
///   [1] t_x_tot: plain scalar x right ToT, "m,k * k,n;a -> m,n;a" (per-col n)
/// Per-regime totals print to stderr at process exit.
inline bool scale_gemm_timing_enabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("TA_GEMM_TIMING");
    return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
  }();
  return enabled;
}

/// Counters for one scale regime. `{0}` member-init gives well-defined zero.
struct ScaleRegimeCounters {
  std::atomic<std::uint64_t> gemm_ns{0};  // wall ns inside the strided gemm
  std::atomic<std::uint64_t> fb_ns{0};    // wall ns inside the AXPY fallback
  std::atomic<std::uint64_t> gemm_runs{
      0};                                 // clean rows/cols (one strided GEMM)
  std::atomic<std::uint64_t> fb_runs{0};  // rows/cols that fell back to AXPY
  std::atomic<std::uint64_t> gemm_flop{0};  // 2*K*N*A (clean), summed
  std::atomic<std::uint64_t> fb_flop{0};  // exact 2*K*Sum(cellsize) (fallback)
  std::atomic<std::uint64_t> fb_absent{0};  // fallback reason: an empty cell
  std::atomic<std::uint64_t> fb_ragged{
      0};  // fallback reason: ragged inner size
  std::atomic<std::uint64_t> fb_stride{
      0};  // fallback reason: multi-page stride
  // --- phase breakdown of the per-(b,m) loop (Amdahl of the 75% overhead) ---
  std::atomic<std::uint64_t> kernel_ns{0};      // whole for-b/for-m loop body
  std::atomic<std::uint64_t> check_pres_ns{0};  // per-row presence + size scan
  std::atomic<std::uint64_t> check_str_ns{0};   // per-row constant-stride walk
  // beta-eligibility: how many Tensor::gemm CALLS land on a freshly-allocated
  // (this->empty()) output tile -- where beta=0 would be valid -- vs an
  // accumulation into an existing tile (beta=1 required for correctness).
  std::atomic<std::uint64_t> calls_firstwrite{0};
  std::atomic<std::uint64_t> calls_accum{0};
  // loop-residual := kernel_ns - check_pres - check_str - gemm_ns - fb_ns
  //   (row pointer setup, loop control, A<=0 skips; absorbs probe clock cost)
};
inline ScaleRegimeCounters g_scale[2];  // [0]=tot_x_t, [1]=t_x_tot

#ifdef TA_STRIDED_DGEMM_COUNT
// Test-observable strided scale-GEMM fire counters: [0]=tot_x_t, [1]=t_x_tot.
// Incremented once per strided scale GEMM (independent of TA_GEMM_TIMING).
// Type matches the arena counters (g_strided_dgemm_ce_*_calls in arena_einsum.h).
inline std::atomic<std::size_t> g_scale_strided_calls[2]{};
#endif

/// Manual (non-scoped) phase clock for regions that set locals used later, so a
/// timed scope can't wrap them. No-op unless TA_GEMM_TIMING is set. Mirrors the
/// arena_einsum.h phase_start/phase_stop pattern.
inline std::chrono::steady_clock::time_point scale_phase_start() {
  return scale_gemm_timing_enabled() ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
}
inline void scale_phase_stop(std::atomic<std::uint64_t>& acc,
                             std::chrono::steady_clock::time_point t0) {
  if (!scale_gemm_timing_enabled()) return;
  acc.fetch_add(static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count()),
                std::memory_order_relaxed);
}

/// RAII timer over one strided GEMM / one AXPY-fallback run; no-op (no clock
/// read, no atomic touch) unless TA_GEMM_TIMING is set. Mirrors the
/// arena_einsum.h ScopedPhaseTimer.
class ScopedScaleTimer {
 public:
  explicit ScopedScaleTimer(std::atomic<std::uint64_t>& acc)
      : acc_(scale_gemm_timing_enabled() ? &acc : nullptr) {
    if (acc_) t0_ = std::chrono::steady_clock::now();
  }
  ~ScopedScaleTimer() {
    if (!acc_) return;
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0_)
                        .count();
    acc_->fetch_add(static_cast<std::uint64_t>(dt), std::memory_order_relaxed);
  }
  ScopedScaleTimer(const ScopedScaleTimer&) = delete;
  ScopedScaleTimer& operator=(const ScopedScaleTimer&) = delete;

 private:
  std::atomic<std::uint64_t>* acc_;
  std::chrono::steady_clock::time_point t0_;
};

/// Prints the scale-path coverage at process exit (only if TA_GEMM_TIMING set).
struct ScaleGemmTimingDumper {
  ~ScaleGemmTimingDumper() {
    if (!scale_gemm_timing_enabled()) return;
    auto L = [](std::atomic<std::uint64_t>& a) {
      return a.load(std::memory_order_relaxed);
    };
    const char* names[2] = {"tot_x_t (left ToT x scalar, per-row)",
                            "t_x_tot (scalar x right ToT, per-col)"};
    std::uint64_t tg_ns = 0, tf_ns = 0, tg_fl = 0, tf_fl = 0;
    for (int r = 0; r < 2; ++r) {
      const auto gns = L(g_scale[r].gemm_ns), fns = L(g_scale[r].fb_ns);
      const auto gr = L(g_scale[r].gemm_runs), fr = L(g_scale[r].fb_runs);
      const auto gf = L(g_scale[r].gemm_flop), ff = L(g_scale[r].fb_flop);
      tg_ns += gns;
      tf_ns += fns;
      tg_fl += gf;
      tf_fl += ff;
      const double tt = static_cast<double>(gns + fns);
      const double ftot = static_cast<double>(gf + ff);
      std::cerr << "[scale-timing] " << names[r] << ":\n";
      std::cerr << "[scale-timing]   strided GEMM : " << gns / 1e9 << " s  ("
                << gr << " runs)\n";
      std::cerr << "[scale-timing]   fallback AXPY: " << fns / 1e9 << " s  ("
                << fr << " runs)\n";
      std::cerr << "[scale-timing]   time coverage (GEMM / total) : "
                << (tt > 0 ? 100.0 * gns / tt : 0.0) << "%\n";
      std::cerr << "[scale-timing]   FLOP coverage (GEMM / total) : "
                << (ftot > 0 ? 100.0 * gf / ftot : 0.0) << "%  (" << gf / 1e9
                << " GFLOP gemm / " << ftot / 1e9 << " GFLOP)\n";
      std::cerr << "[scale-timing]   GFLOP/s strided="
                << (gns > 0 ? gf / static_cast<double>(gns) : 0.0)
                << "  fallback="
                << (fns > 0 ? ff / static_cast<double>(fns) : 0.0) << "\n";
      std::cerr << "[scale-timing]   fallback runs by reason: absent="
                << L(g_scale[r].fb_absent)
                << " ragged=" << L(g_scale[r].fb_ragged)
                << " multipage-stride=" << L(g_scale[r].fb_stride) << "\n";
      // Phase breakdown of the per-(b,m) loop = where the non-GEMM overhead
      // goes.
      const auto kn = L(g_scale[r].kernel_ns);
      const auto cp = L(g_scale[r].check_pres_ns);
      const auto cs = L(g_scale[r].check_str_ns);
      const auto resid =
          (kn > gns + fns + cp + cs) ? kn - gns - fns - cp - cs : 0;
      auto pc = [kn](std::uint64_t x) {
        return kn > 0 ? 100.0 * static_cast<double>(x) / static_cast<double>(kn)
                      : 0.0;
      };
      std::cerr << "[scale-phases] kernel total (for-b/for-m): " << kn / 1e9
                << " s\n";
      std::cerr << "[scale-phases]   strided GEMM        : " << gns / 1e9
                << " s  (" << pc(gns) << "%)\n";
      std::cerr << "[scale-phases]   fallback AXPY       : " << fns / 1e9
                << " s  (" << pc(fns) << "%)\n";
      std::cerr << "[scale-phases]   clean-check presence: " << cp / 1e9
                << " s  (" << pc(cp) << "%)\n";
      std::cerr << "[scale-phases]   clean-check STRIDE walk: " << cs / 1e9
                << " s  (" << pc(cs) << "%)\n";
      std::cerr << "[scale-phases]   loop residual       : " << resid / 1e9
                << " s  (" << pc(resid) << "%)\n";
      const auto fw = L(g_scale[r].calls_firstwrite);
      const auto ac = L(g_scale[r].calls_accum);
      std::cerr << "[scale-beta] gemm CALLS: first-write (beta=0 ok)=" << fw
                << "  accumulate (beta=1 needed)=" << ac << "  ("
                << (fw + ac > 0 ? 100.0 * fw / (fw + ac) : 0.0)
                << "% beta=0-eligible)\n";
    }
    const double allt = static_cast<double>(tg_ns + tf_ns);
    const double allf = static_cast<double>(tg_fl + tf_fl);
    std::cerr << "[scale-timing] SCALE TOTAL: strided GEMM " << tg_ns / 1e9
              << " s, fallback AXPY " << tf_ns / 1e9 << " s, time coverage "
              << (allt > 0 ? 100.0 * tg_ns / allt : 0.0) << "%, FLOP coverage "
              << (allf > 0 ? 100.0 * tg_fl / allf : 0.0) << "%\n";
  }
};
inline ScaleGemmTimingDumper g_scale_gemm_timing_dumper;

}  // namespace detail

/// An N-dimensional tensor object

/// A contiguous row-major tensor with __shallow-copy__ semantics.
/// As of TiledArray 1.1 Tensor represents a batch of tensors with same Range
/// (the default batch size = 1).
/// \tparam T The value type of this tensor
/// \tparam A The allocator type for the data; only default-constructible
/// allocators are supported to save space
template <typename T, typename Allocator>
class Tensor {
  // meaningful error if T& is not assignable, see
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=48101
  static_assert(std::is_assignable<std::add_lvalue_reference_t<T>, T>::value,
                "Tensor<T,Allocator>: T must be an assignable type (e.g. "
                "cannot be const)");
  // default-constructible Allocator allows to reduce the size of default Tensor
  // and minimize the overhead of null elements in Tensor<Tensor<T>>
  static_assert(
      std::is_default_constructible_v<Allocator>,
      "Tensor<T,Allocator>: only default-constructible Allocator is supported");

#ifdef TA_TENSOR_MEM_TRACE
  template <typename... Ts>
  std::string make_string(Ts&&... ts) {
    std::ostringstream oss;
    (oss << ... << ts);
    return oss.str();
  }
#endif

 public:
  typedef Range range_type;                              ///< Tensor range type
  typedef typename range_type::index1_type index1_type;  ///< 1-index type
  typedef typename range_type::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename range_type::ordinal_type
      size_type;  ///< Size type (to meet the container concept)
  typedef Allocator allocator_type;  ///< Allocator type
  typedef typename std::allocator_traits<allocator_type>::value_type
      value_type;  ///< Array element type
  typedef std::add_lvalue_reference_t<value_type>
      reference;  ///< Element (lvalue) reference type
  typedef std::add_lvalue_reference_t<std::add_const_t<value_type>>
      const_reference;  ///< Element (const lvalue) reference type
  typedef typename std::allocator_traits<allocator_type>::pointer
      pointer;  ///< Element pointer type
  typedef typename std::allocator_traits<allocator_type>::const_pointer
      const_pointer;  ///< Element const pointer type
  typedef typename std::allocator_traits<allocator_type>::difference_type
      difference_type;                   ///< Difference type
  typedef pointer iterator;              ///< Element iterator type
  typedef const_pointer const_iterator;  ///< Element const iterator type
  typedef typename TiledArray::detail::numeric_type<T>::type
      numeric_type;  ///< the numeric type that supports T
  typedef typename TiledArray::detail::scalar_type<T>::type
      scalar_type;  ///< the scalar type that supports T

 private:
  template <typename X>
  using value_t = typename X::value_type;
  template <typename X>
  using numeric_t = typename TiledArray::detail::numeric_type<X>::type;

  template <typename... Ts>
  struct is_tensor {
    static constexpr bool value = detail::is_tensor<Ts...>::value ||
                                  detail::is_tensor_of_tensor<Ts...>::value;
  };

 public:
  /// compute type of Tensor with different element type
  template <typename U,
            typename OtherAllocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<U>>
  using rebind_t = Tensor<U, OtherAllocator>;

  template <typename U, typename V = value_type, typename = void>
  struct rebind_numeric;
  template <typename U, typename V>
  struct rebind_numeric<U, V, std::enable_if_t<is_tensor<V>::value>> {
    using VU = typename V::template rebind_numeric<U>::type;
    using type = Tensor<VU, typename std::allocator_traits<
                                Allocator>::template rebind_alloc<VU>>;
  };
  template <typename U, typename V>
  struct rebind_numeric<U, V, std::enable_if_t<!is_tensor<V>::value>> {
    using type = Tensor<
        U, typename std::allocator_traits<Allocator>::template rebind_alloc<U>>;
  };

  /// compute type of Tensor with different numeric type
  template <typename U>
  using rebind_numeric_t = typename rebind_numeric<U, value_type>::type;

 private:
  using default_construct = bool;

  Tensor(const range_type& range, size_t nbatch, bool default_construct)
      : range_(range), nbatch_(nbatch) {
    size_t size = range_.volume() * nbatch;
    allocator_type allocator;
    auto* ptr = allocator.allocate(size);
    // default construct elements of data only if can have any effect ...
    if constexpr (!std::is_trivially_default_constructible_v<T>) {
      // .. and requested
      if (default_construct) {
        std::uninitialized_default_construct_n(ptr, size);
      }
    }
    auto deleter = [
#ifdef TA_TENSOR_MEM_TRACE
                       this,
#endif
                       allocator = std::move(allocator),
                       size](auto&& ptr) mutable {
      std::destroy_n(ptr, size);
      // N.B. deregister ptr *before* deallocating to avoid possible race
      // between reallocation and deregistering
#ifdef TA_TENSOR_MEM_TRACE
      const auto nbytes = size * sizeof(T);
      if (nbytes >= trace_if_larger_than_) {
        ptr_registry()->erase(ptr, nbytes,
                              make_string("created by TA::Tensor*=", this));
      }
#endif
      allocator.deallocate(ptr, size);
    };
    this->data_ = std::shared_ptr<value_type[]>(ptr, std::move(deleter));
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::data_.get()=", data_.get()));
      ptr_registry()->insert(data_.get(), nbytes(),
                             make_string("created by TA::Tensor*=", this));
    }
#endif
  }

  Tensor(range_type&& range, size_t nbatch, bool default_construct)
      : range_(std::move(range)), nbatch_(nbatch) {
    size_t size = range_.volume() * nbatch;
    allocator_type allocator;
    auto* ptr = allocator.allocate(size);
    // default construct elements of data only if can have any effect ...
    if constexpr (!std::is_trivially_default_constructible_v<T>) {
      // .. and requested
      if (default_construct) {
        std::uninitialized_default_construct_n(ptr, size);
      }
    }
    auto deleter = [
#ifdef TA_TENSOR_MEM_TRACE
                       this,
#endif
                       allocator = std::move(allocator),
                       size](auto&& ptr) mutable {
      std::destroy_n(ptr, size);
      // N.B. deregister ptr *before* deallocating to avoid possible race
      // between reallocation and deregistering
#ifdef TA_TENSOR_MEM_TRACE
      const auto nbytes = size * sizeof(T);
      if (nbytes >= trace_if_larger_than_) {
        ptr_registry()->erase(ptr, nbytes,
                              make_string("created by TA::Tensor*=", this));
      }
#endif
      allocator.deallocate(ptr, size);
    };
    this->data_ = std::shared_ptr<value_type[]>(ptr, std::move(deleter));
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::data_.get()=", data_.get()));
      ptr_registry()->insert(data_.get(), nbytes(),
                             make_string("created by TA::Tensor*=", this));
    }
#endif
  }

  template <typename T_>
  static decltype(auto) value_converter(const T_& arg) {
    using arg_type = detail::remove_cvr_t<decltype(arg)>;
    if constexpr (detail::is_tensor_v<arg_type> &&
                  !is_tensor_view_v<arg_type>)  // clone owning nested tensors
      return arg.clone();
    else if constexpr (!std::is_same_v<arg_type, value_type>) {  // convert
      if constexpr (std::is_convertible_v<arg_type, value_type>)
        return static_cast<value_type>(arg);
      else
        return conversions::to<value_type, arg_type>()(arg);
    } else
      return arg;  // identity (for views, copy = rebind, no deep clone)
  };

  range_type range_;  ///< Range
  /// Number of `range_`-sized blocks in `data_`
  /// \note this is not used for (in)equality comparison
  size_t nbatch_ = 1;
  std::shared_ptr<value_type[]> data_;  ///< Shared pointer to the data

 public:
  /// constructs an empty (null) Tensor
  /// \post `this->empty()`
  Tensor() = default;

  /// copy constructor

  /// \param[in] other an object to copy data from
  /// \post `*this` is a shallow copy of \p other,
  /// i.e. `*this == other && this->data()==other.data()`
  Tensor(const Tensor& other)
      : range_(other.range_), nbatch_(other.nbatch_), data_(other.data_) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor(const Tensor& other)::data_.get()=",
                            data_.get()));
    }
#endif
  }

  /// move constructor

  /// \param[in,out] other an object to move data from;
  ///                      on return \p other is in empty (null) but not
  ///                      necessarily default state
  /// \post `other.empty()`
  Tensor(Tensor&& other)
      : range_(std::move(other.range_)),
        nbatch_(std::move(other.nbatch_)),
        data_(std::move(other.data_)) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          &other,
          make_string("TA::Tensor(Tensor&& other)::data_.get()=", data_.get()));
      ptr_registry()->insert(
          this,
          make_string("TA::Tensor(Tensor&& other)::data_.get()=", data_.get()));
    }
#endif
  }

  ~Tensor() {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          this, make_string("TA::~Tensor()::data_.get()=", data_.get()));
    }
#endif
  }

  struct nbatches {
    template <typename Int,
              typename = std::enable_if_t<std::is_integral_v<Int>>>
    nbatches(Int n) : n(n) {}
    template <typename Int,
              typename = std::enable_if_t<std::is_integral_v<Int>>>
    nbatches& operator=(Int n) {
      this->n = n;
      return *this;
    }

    size_type n = 1;
  };

  /// Construct a tensor with a range equal to \c range. The data is
  /// default-initialized (which, for `T` with trivial default constructor,
  /// means data is uninitialized).
  /// \param range The range of the tensor
  /// \param nbatch The number of batches (default is 1)
  explicit Tensor(const range_type& range, nbatches nb = 1)
      : Tensor(range, nb.n, default_construct{true}) {}

  /// Construct a tensor of tensor values, setting all elements to the same
  /// value

  /// \param range An array with the size of of each dimension
  /// \param value The value of the tensor elements
  template <
      typename Value,
      typename std::enable_if<std::is_same<Value, value_type>::value &&
                              detail::is_tensor<Value>::value>::type* = nullptr>
  Tensor(const range_type& range, const Value& value)
      : Tensor(range, 1, default_construct{false}) {
    const auto n = this->size();
    pointer MADNESS_RESTRICT const data = this->data();
    if constexpr (is_tensor_view_v<Value>) {
      // Views are rebind-on-copy and lack member `clone`; just copy each.
      for (size_type i = 0ul; i < n; ++i) new (data + i) value_type(value);
    } else {
      Clone<Value, Value> cloner;
      for (size_type i = 0ul; i < n; ++i)
        new (data + i) value_type(cloner(value));
    }
  }

  /// Construct a tensor of scalars, setting all elements to the same value

  /// \param range An array with the size of of each dimension
  /// \param value The value of the tensor elements
  template <typename Value,
            typename std::enable_if<std::is_convertible_v<Value, value_type> &&
                                    !detail::is_tensor<Value>::value>::type* =
                nullptr>
  Tensor(const range_type& range, const Value& value)
      : Tensor(range, 1, default_construct{false}) {
    detail::tensor_init([value]() -> Value { return value; }, *this);
  }

  /// Construct a tensor with a fill op that takes an element index

  /// \tparam ElementIndexOp callable of signature
  /// `value_type(const Range::index_type&)`
  /// \param range An array with the size of of each dimension
  /// \param element_idx_op a callable of type ElementIndexOp
  template <typename ElementIndexOp,
            typename = std::enable_if_t<std::is_invocable_r_v<
                value_type, ElementIndexOp, const Range::index_type&>>>
  Tensor(const range_type& range, const ElementIndexOp& element_idx_op)
      : Tensor(range, 1, default_construct{false}) {
    pointer MADNESS_RESTRICT const data = this->data();
    for (auto&& element_idx : range) {
      const auto ord = range.ordinal(element_idx);
      new (data + ord) value_type(element_idx_op(element_idx));
    }
  }

  /// Construct an evaluated tensor
  template <typename InIter,
            typename std::enable_if<
                TiledArray::detail::is_input_iterator<InIter>::value &&
                !std::is_pointer<InIter>::value>::type* = nullptr>
  Tensor(const range_type& range, InIter it)
      : Tensor(range, 1, default_construct{false}) {
    auto n = range.volume();
    pointer MADNESS_RESTRICT data = this->data();
    for (size_type i = 0ul; i < n; ++i, ++it, ++data)
      new (data) value_type(*it);
  }

  template <typename U>
  Tensor(const Range& range, const U* u)
      : Tensor(range, 1, default_construct{false}) {
    math::uninitialized_copy_vector(range.volume(), u, this->data());
  }

  explicit Tensor(const Range& range, std::initializer_list<T> il)
      : Tensor(range, il.begin()) {}

  /// Construct a copy of a tensor interface object

  /// \tparam T1 A tensor type
  /// \param other The tensor to be copied
  /// \note this constructor is disabled if \p T1 already has a conversion
  ///       operator to this type
  /// \warning if `T1` is a tensor of tensors its elements are _cloned_ rather
  ///          than copied to make the semantics of  this to be consistent
  ///          between tensors of scalars and tensors of scalars; specifically,
  ///          if `T1` is a tensor of scalars the constructed tensor is
  ///          is independent of \p other, thus should apply clone to inner
  ///          tensor nests to behave similarly for nested tensors
  template <
      typename T1,
      typename std::enable_if<
          is_tensor<T1>::value && !std::is_same<T1, Tensor>::value &&
          !detail::has_conversion_operator_v<T1, Tensor>>::type* = nullptr>
  explicit Tensor(const T1& other)
      : Tensor(detail::clone_range(other), 1, default_construct{false}) {
    detail::tensor_init(value_converter<typename T1::value_type>, *this, other);
  }

  /// Construct a permuted tensor copy

  /// \tparam T1 A tensor type
  /// \tparam Perm A permutation type
  /// \param other The tensor to be copied
  /// \param perm The permutation that will be applied to the copy
  /// \warning if `T1` is a tensor of tensors its elements are _cloned_ rather
  ///          than copied to make the semantics of  this to be consistent
  ///          between tensors of scalars and tensors of tensors; specifically,
  ///          if `T1` is a tensor of scalars the constructed tensor is
  ///          is independent of \p other, thus should apply clone to inner
  ///          tensor nests to behave similarly for nested tensors
  template <
      typename T1, typename Perm,
      typename std::enable_if<detail::is_nested_tensor_v<T1> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor(const T1& other, const Perm& perm)
      : Tensor(outer(perm) * other.range(), other.nbatch(),
               default_construct{false}) {
    const auto outer_perm = outer(perm);
    // The outer permute kernel (tensor_init -> permute) writes through
    // at_ordinal, which requires nbatch()==1. Apply it per batch slice when
    // batched: batch(b) is a writable nbatch==1 view sharing this/other's
    // storage, so each slice permutes in place. For nbatch()==1 (every
    // production/expression path -- see plan Background) the loop body runs
    // exactly once and is identical to the prior code. nbatch()>1 only
    // arises on the deprecated legacy subworld einsum route.
    if (this->nbatch() == 1) {
      if (outer_perm) {
        detail::tensor_init(value_converter<typename T1::value_type>,
                            outer_perm, *this, other);
      } else {
        detail::tensor_init(value_converter<typename T1::value_type>, *this,
                            other);
      }
    } else {
      for (std::size_t b = 0; b < this->nbatch(); ++b) {
        auto this_b = this->batch(b);
        auto other_b = other.batch(b);
        if (outer_perm) {
          detail::tensor_init(value_converter<typename T1::value_type>,
                              outer_perm, this_b, other_b);
        } else {
          detail::tensor_init(value_converter<typename T1::value_type>, this_b,
                              other_b);
        }
      }
    }

    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    constexpr bool is_view = is_tensor_view_v<value_type>;
    // tile ops pass bipartite permutations here even if this is a plain tensor.
    // For view inners, the cell has fixed layout that can't be permuted in
    // place -- skip the inner-permute pass and rely on callers to arrange
    // canonical inner indexing (regime-A einsum's `do_perm.{A,B,C}` bailout
    // guarantees no inner permutation is needed for our paths).
    if constexpr (is_tot && is_bperm && !is_view) {
      if (inner_size(perm) != 0) {
        const auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;

        auto volume = total_size();
        for (decltype(volume) i = 0; i < volume; ++i) {
          auto& el = *(data() + i);
          if (!el.empty()) el = p(el, inner_perm);
        }
      }
    } else if constexpr (is_tot && is_bperm && is_view) {
      if (inner_size(perm) != 0) {
        TA_EXCEPTION(
            "Tensor<View>: inner permutation requested but view "
            "cells cannot be permuted in place");
      }
    }
  }

  /// "Element-wise" unary transform of \c other

  /// \tparam T1 A tensor type
  /// \tparam Op A unary callable
  /// \param other The tensor argument
  /// \param op Unary operation that can be invoked on elements of \p other;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
  template <typename T1, typename Op,
            typename std::enable_if_t<
                is_tensor<T1>::value &&
                !detail::is_permutation_v<std::decay_t<Op>>>* = nullptr>
  Tensor(const T1& other, Op&& op)
      : Tensor(detail::clone_range(other), 1, default_construct{false}) {
    detail::tensor_init(op, *this, other);
  }

  /// "Element-wise" unary transform of \c other fused with permutation

  /// equivalent, but more efficient, than `Tensor(other, op).permute(perm)`
  /// \tparam T1 A tensor type
  /// \tparam Op A unary callable
  /// \tparam Perm A permutation type
  /// \param other The tensor argument
  /// \param op Unary operation that can be invoked as` op(other[i]))`;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
  template <
      typename T1, typename Op, typename Perm,
      typename std::enable_if_t<is_tensor<T1>::value &&
                                detail::is_permutation_v<Perm>>* = nullptr>
  Tensor(const T1& other, Op&& op, const Perm& perm)
      : Tensor(outer(perm) * other.range(), 1, default_construct{false}) {
    detail::tensor_init(op, outer(perm), *this, other);
    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;
        for (auto& x : *this) x = p(x, inner_perm);
      }
    }
  }

  /// "Element-wise" binary transform of \c {left,right}

  /// \tparam T1 A tensor type
  /// \tparam T2 A tensor type
  /// \tparam Op A binary callable
  /// \param left The left-hand tensor argument
  /// \param right The right-hand tensor argument
  /// \param op Binary operation that can be invoked as `op(left[i],right[i]))`;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
  template <typename T1, typename T2, typename Op,
            typename = std::enable_if_t<detail::is_nested_tensor_v<T1, T2>>>
  Tensor(const T1& left, const T2& right, Op&& op)
      : Tensor(detail::clone_range(left), 1, default_construct{false}) {
    detail::tensor_init(op, *this, left, right);
  }

  /// "Element-wise" binary transform of \c {left,right} fused with permutation

  /// \tparam T1 A tensor type
  /// \tparam T2 A tensor type
  /// \tparam Op A binary callable
  /// \tparam Perm A permutation tile
  /// \param left The left-hand tensor argument
  /// \param right The right-hand tensor argument
  /// \param op Binary operation that can be invoked as `op(left[i],right[i]))`;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
  /// \param perm The permutation that will be applied to the arguments
  template <
      typename T1, typename T2, typename Op, typename Perm,
      typename std::enable_if<detail::is_nested_tensor<T1, T2>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor(const T1& left, const T2& right, Op&& op, const Perm& perm)
      : Tensor(outer(perm) * left.range(), 1, default_construct{false}) {
    detail::tensor_init(op, outer(perm), *this, left, right);
    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;
        for (auto& x : *this) x = p(x, inner_perm);
      }
    }
  }

  /// Construct a tensor with a range equal to \c range using existing data
  /// \param range The range of the tensor
  /// \param nbatch The number of batches
  /// \param data shared pointer to the data
  Tensor(const range_type& range, size_t nbatch,
         std::shared_ptr<value_type[]> data)
      : range_(range), nbatch_(nbatch), data_(std::move(data)) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor(range, nbatch, data)::data_.get()=",
                            data_.get()));
    }
#endif
  }

  /// Construct a tensor with a range equal to \c range using existing data
  /// assuming unit batch size \param range The range of the tensor \param data
  /// shared pointer to the data
  Tensor(const range_type& range, std::shared_ptr<value_type[]> data)
      : range_(range), nbatch_(1), data_(std::move(data)) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this,
          make_string("TA::Tensor(range, data)::data_.get()=", data_.get()));
    }
#endif
  }

  /// The batch size accessor

  /// @return the size of tensor batch represented by `*this`
  size_t nbatch() const { return this->nbatch_; }

  /// @param[in] idx the batch index
  /// @pre `idx < this->nbatch()`
  /// @return (plain, i.e. nbatch=1) Tensor representing element \p idx of
  /// the batch
  Tensor batch(size_t idx) const {
    TA_ASSERT(idx < this->nbatch());
    std::shared_ptr<value_type[]> data(this->data_,
                                       this->data_.get() + idx * this->size());
    return Tensor(this->range(), 1, data);
  }

  /// Returns Tensor representing the data using another range and batch size

  /// @param[in] range the Range of the result
  /// @param[in] nbatch the number of batches of the result
  /// @return Tensor object representing `this->data()` using @p range and @p
  /// nbatch
  auto reshape(const range_type& range, size_t nbatch = 1) const {
    TA_ASSERT(this->range().volume() * this->nbatch() ==
              range.volume() * nbatch);
    return Tensor(range, nbatch, this->data_);
  }

  /// @return a deep copy of `*this`
  Tensor clone() const& {
    Tensor result;
    if (data_) {
      if constexpr (detail::is_tensor_of_tensor_v<Tensor> &&
                    detail::is_ta_tensor_v<value_type>) {
        auto fill = [](typename value_type::value_type* dst,
                       const typename value_type::value_type* src,
                       std::size_t n) {
          for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
        };
        result = detail::arena_trivial_unary<Tensor>(*this, fill);
      } else if constexpr (is_arena_tensor_v<value_type>) {
        auto fill = [](typename value_type::value_type* dst,
                       const typename value_type::value_type* src,
                       std::size_t n) {
          for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
        };
        result = detail::arena_trivial_unary<Tensor>(*this, fill);
      } else {
        result = detail::tensor_op<Tensor>(
            [](const numeric_type value) -> numeric_type { return value; },
            *this);
      }
    } else if (range_) {  // corner case: data_ = null implies range_.volume()
                          // == 0;
      TA_ASSERT(range_.volume() == 0);
      result = Tensor(range_);
    }
    return result;
  }

  /// cloning an rvalue ref forwards the contents of this
  /// @return a deep copy of `*this`
  /// @post this is in a moved-from state
  Tensor clone() && { return std::move(*this); }

  template <typename T1,
            typename std::enable_if<is_tensor<T1>::value>::type* = nullptr>
  Tensor& operator=(const T1& other) {
    *this = Tensor(detail::clone_range(other), 1, default_construct{false});
    detail::inplace_tensor_op(
        [](reference MADNESS_RESTRICT tr,
           typename T1::const_reference MADNESS_RESTRICT t1) { tr = t1; },
        *this, other);

    return *this;
  }

  /// copy assignment operator

  /// \param[in] other an object to copy data from
  /// \post `*this` is a shallow copy of \p other,
  /// i.e. `*this == other && this->data()==other.data()`
  Tensor& operator=(const Tensor& other) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          this,
          make_string("TA::Tensor::operator=(const Tensor&)::data_.get()=",
                      data_.get()));
    }
#endif
    range_ = other.range_;
    nbatch_ = other.nbatch_;
    data_ = other.data_;
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this,
          make_string("TA::Tensor::operator=(const Tensor&)::data_.get()=",
                      data_.get()));
    }
#endif
    return *this;
  }

  /// move assignment operator

  /// \param[in,out] other an object to move data from;
  ///                      on return \p other is in empty (null) but not
  ///                      necessarily default state
  /// \post `other.empty()`
  Tensor& operator=(Tensor&& other) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          this, make_string("TA::Tensor::operator=(Tensor&&)::data_.get()=",
                            data_.get()));
    }
    if (other.nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          &other, make_string("TA::Tensor::operator=(Tensor&&)::data_.get()=",
                              data_.get()));
    }
#endif
    range_ = std::move(other.range_);
    nbatch_ = std::move(other.nbatch_);
    data_ = std::move(other.data_);
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::operator=(Tensor&&)::data_.get()=",
                            data_.get()));
    }
#endif
    return *this;
  }

  /// Tensor range object accessor

  /// \return The tensor range object
  const range_type& range() const { return range_; }

  /// Tensor dimension size accessor

  /// \return The number of elements in the tensor
  ordinal_type size() const { return (this->range().volume()); }

  /// \return The number of elements in the tensor by summing up the sizes of
  /// the batches.
  ordinal_type total_size() const { return size() * nbatch(); }

  /// Tensor data size (in bytes) accessor

  /// \return The number of bytes occupied by this tensor's data
  /// \warning this only returns valid value if this is a tensor of scalars
  std::size_t nbytes() const {
    return this->range().volume() * this->nbatch_ * sizeof(T);
  }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  const_reference operator[](const Ordinal ord) const {
    TA_ASSERT(!this->empty());
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator[](index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  reference operator[](const Ordinal ord) {
    TA_ASSERT(!this->empty());
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator[](index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  const_reference at_ordinal(const Ordinal ord) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  reference at_ordinal(const Ordinal ord) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator[](const Index& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator[](const Index& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  const_reference operator[](const std::initializer_list<Integer>& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  reference operator[](const std::initializer_list<Integer>& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral_v<Ordinal>>* = nullptr>
  const_reference operator()(const Ordinal& ord) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral_v<Ordinal>>* = nullptr>
  reference operator()(const Ordinal& ord) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator()(const Index& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator()(const Index& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  const_reference operator()(const std::initializer_list<Integer>& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  reference operator()(const std::initializer_list<Integer>& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  ///)
  /// \param[in] i an index \return Const reference to the element at position
  /// \c i.
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <
      typename... Index,
      std::enable_if_t<(sizeof...(Index) > 1ul) &&
                       detail::is_integral_list<Index...>::value>* = nullptr>
  const_reference operator()(const Index&... i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range().rank() == sizeof...(Index));
    // can't distinguish between operator()(Index...) and operator()(ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    using Int = std::common_type_t<Index...>;
    const auto iord = this->range_.ordinal(
        std::array<Int, sizeof...(Index)>{{static_cast<Int>(i)...}});
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  ///)
  /// \param[in] i an index \return Reference to the element at position \c i
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <
      typename... Index,
      std::enable_if_t<(sizeof...(Index) > 1ul) &&
                       detail::is_integral_list<Index...>::value>* = nullptr>
  reference operator()(const Index&... i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range().rank() == sizeof...(Index));
    // can't distinguish between operator()(Index...) and operator()(ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    using Int = std::common_type_t<Index...>;
    const auto iord = this->range_.ordinal(
        std::array<Int, sizeof...(Index)>{{static_cast<Int>(i)...}});
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Iterator factory

  /// \return A const iterator to the first data element
  const_iterator begin() const { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return An iterator to the first data element
  iterator begin() { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the last data element
  const_iterator end() const {
    return (this->data() ? this->data() + this->size() : NULL);
  }

  /// Iterator factory

  /// \return An iterator to the last data element
  iterator end() { return (this->data() ? this->data() + this->size() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the first data element
  const_iterator cbegin() const { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the first data element
  const_iterator cbegin() { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the last data element
  const_iterator cend() const {
    return (this->data() ? this->data() + this->size() : NULL);
  }

  /// Iterator factory

  /// \return A const iterator to the last data element
  const_iterator cend() {
    return (this->data() ? this->data() + this->size() : NULL);
  }

  /// Read-only access to the data

  /// \return A const pointer to the tensor data
  const_pointer data() const { return this->data_.get(); }

  /// Mutable access to the data

  /// \return A mutable pointer to the tensor data
  pointer data() { return this->data_.get(); }

  /// @param[in] batch_idx the batch index
  /// @pre `batch_idx < this->nbatch()`
  /// @return A const pointer to the tensor data of the batch \p batch_idx
  const_pointer batch_data(size_t batch_idx) const {
    TA_ASSERT(batch_idx < this->nbatch());
    return data() + batch_idx * size();
  }

  /// @param[in] batch_idx the batch index
  /// @pre `batch_idx < this->nbatch()`
  /// @return A const pointer to the tensor data of the batch \p batch_idx
  pointer batch_data(size_t batch_idx) {
    TA_ASSERT(batch_idx < this->nbatch());
    return data() + batch_idx * size();
  }

  /// Read-only shared_ptr to the data

  /// \return A const shared_ptr to the tensor data
  std::shared_ptr<const value_type[]> data_shared() const {
    return this->data_;
  }

  /// Mutable shared_ptr to the data

  /// \return A mutable shared_ptr to the tensor data
  std::shared_ptr<value_type[]> data_shared() { return this->data_; }

  /// Test if the tensor is empty

  /// \return \c true if this tensor contains no
  ///         data, otherwise \c false.
  /// \note Empty Tensor is defaul_ish_, i.e. it is *equal* to
  ///       a default-constructed Tensor
  ///       (`this->empty()` is equivalent to `*this == Tensor{}`),
  ///       but is not identical
  ///       to a default-constructed Tensor (e.g., `this->empty()` does not
  ///       imply `this->nbatch() == Tensor{}.nbatch()`)
  bool empty() const {
    // empty data_ implies default values for range_ (but NOT nbatch_)
    TA_ASSERT(
        (this->data_.use_count() == 0 && !this->range_) ||
        (this->data_.use_count() != 0 && this->range_));  // range is empty
    return this->data_.use_count() == 0;
  }

  /// MADNESS serialization function

  /// This function enables serialization within MADNESS
  /// \tparam Archive A MADNESS archive type
  /// \param[out] ar An input/output archive
  template <typename Archive>
  void serialize(Archive& ar) {
    bool empty = this->empty();
    auto range = this->range_;
    auto nbatch = this->nbatch_;
    ar & empty;
    if (!empty) {
      ar & range;
      ar & nbatch;
      if constexpr (is_arena_tensor_v<value_type>) {
        // ArenaTensor inner cells own no storage themselves; their data
        // lives in a per-outer-tile arena slab. Bypass the generic
        // wrap(value_type*, N) path (which would try to serialize bare
        // Cell* pointers across processes) and manage cell storage at
        // this outer-tile boundary instead. The slab is rebuilt on load.
        serialize_arena_inner_cells(ar, std::move(range), nbatch);
      } else {
        if constexpr (madness::is_input_archive_v<Archive>) {
          *this = Tensor(std::move(range), nbatch, default_construct{true});
        }
        ar& madness::archive::wrap(this->data_.get(),
                                   this->range_.volume() * nbatch);
      }
    } else {
      if constexpr (madness::is_input_archive_v<Archive>) {
        *this = Tensor{};
      }
    }
  }

 private:
  /// ArenaTensor-aware inner-cell serialization. Writes per-cell metadata
  /// (null flag + range) then element bytes; on load, rebuilds the outer
  /// via `arena_outer_init` so the slab is reconstructed in one
  /// allocation and the outer-data deleter keeps it alive.
  template <typename Archive>
  void serialize_arena_inner_cells(Archive& ar, range_type range,
                                   std::size_t nbatch) {
    using InnerT = value_type;
    using InnerRange = typename InnerT::range_type;
    const std::size_t N = range.volume() * nbatch;
    if constexpr (madness::is_output_archive_v<Archive>) {
      // Per-cell null flags.
      for (std::size_t i = 0; i < N; ++i) {
        bool not_null = bool(this->data_.get()[i]);
        ar & not_null;
      }
      // Inner ranges for non-null cells only.
      for (std::size_t i = 0; i < N; ++i) {
        const InnerT& cell = this->data_.get()[i];
        if (cell) ar & cell.range();
      }
      // Element bytes for non-null cells only.
      for (std::size_t i = 0; i < N; ++i) {
        const InnerT& cell = this->data_.get()[i];
        if (cell) ar& madness::archive::wrap(cell.data(), cell.size());
      }
    } else {
      // Load: read all metadata, plan + allocate slab via the factory,
      // then read element bytes into each placed cell's data().
      std::vector<bool> flags(N);
      for (std::size_t i = 0; i < N; ++i) {
        bool f;
        ar & f;
        flags[i] = f;
      }
      std::vector<InnerRange> ranges(N);
      for (std::size_t i = 0; i < N; ++i) {
        if (flags[i]) ar& ranges[i];
      }
      *this = detail::arena_outer_init<Tensor>(
          range, nbatch, [&](std::size_t ord) -> InnerRange {
            return flags[ord] ? ranges[ord] : InnerRange{};
          });
      for (std::size_t i = 0; i < N; ++i) {
        if (flags[i]) {
          InnerT& cell = this->data_.get()[i];
          ar& madness::archive::wrap(cell.data(), cell.size());
        }
      }
    }
  }

 public:
  /// Swap tensor data

  /// \param other The tensor to swap with this
  void swap(Tensor& other) {
#ifdef TA_TENSOR_MEM_TRACE
    bool this_to_be_traced = false;
    bool other_to_be_traced = false;
    if (nbytes() >= trace_if_larger_than_) {
      this_to_be_traced = true;
      ptr_registry()->erase(
          this, make_string("TA::Tensor::swap()::data_.get()=", data_.get()));
    }
    if (other.nbytes() >= trace_if_larger_than_) {
      other_to_be_traced = true;
      ptr_registry()->erase(
          &other,
          make_string("TA::Tensor::swap()::data_.get()=", other.data_.get()));
    }
#endif
    std::swap(data_, other.data_);
    std::swap(range_, other.range_);
    std::swap(nbatch_, other.nbatch_);
#ifdef TA_TENSOR_MEM_TRACE
    if (other_to_be_traced) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::swap()::data_.get()=", data_.get()));
    }
    if (this_to_be_traced) {
      ptr_registry()->insert(
          &other,
          make_string("TA::Tensor::swap()::data_.get()=", other.data_.get()));
    }
#endif
  }

  // clang-format off
  /// Constructs a view of the block defined by \p lower_bound and \p upper_bound.

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///   auto tview = t.block(lobounds, upbounds);
  ///   assert(tview.range().includes(lobounds));
  ///   assert(tview(lobounds) == t(lobounds));
  /// \endcode
  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound
  /// \param upper_bound The upper bound
  /// \return a {const,mutable} view of the block defined by \p lower_bound and \p upper_bound
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  /// @{
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  detail::TensorInterface<T, BlockRange> block(const Index1& lower_bound,
                                               const Index2& upper_bound) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  detail::TensorInterface<const T, BlockRange> block(
      const Index1& lower_bound, const Index2& upper_bound) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by \p lower_bound and \p upper_bound.

  /// Examples of using this:
  /// \code
  ///   auto tview = t.block({0, 1, 2}, {4, 6, 8});
  ///   assert(tview.range().includes(lobounds));
  ///   assert(tview(lobounds) == t(lobounds));
  /// \endcode
  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower bound
  /// \param upper_bound The upper bound
  /// \return a {const,mutable} view of the block defined by \p lower_bound and \p upper_bound
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  /// @{
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  detail::TensorInterface<T, BlockRange> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  detail::TensorInterface<const T, BlockRange> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by its \p bounds.

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///
  ///   // using vector of pairs
  ///   std::vector<std::pair<size_t,size_t>> vpbounds{{0,4}, {1,6}, {2,8}};
  ///   auto tview0 = t.block(vpbounds);
  ///   // using vector of tuples
  ///   std::vector<std::tuple<size_t,size_t>> vtbounds{{0,4}, {1,6}, {2,8}};
  ///   auto tview1 = t.block(vtbounds);
  ///   assert(tview0 == tview1);
  ///
  ///   // using zipped ranges of bounds (using Boost.Range)
  ///   // need to #include <boost/range/combine.hpp>
  ///   auto tview2 = t.block(boost::combine(lobounds, upbounds));
  ///   assert(tview0 == tview2);
  ///
  ///   // using zipped ranges of bounds (using Ranges-V3)
  ///   // need to #include <range/v3/view/zip.hpp>
  ///   auto tview3 = t.block(ranges::views::zip(lobounds, upbounds));
  ///   assert(tview0 == tview3);
  /// \endcode
  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v)
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange> &&
                                        !std::is_same_v<PairRange, Range>>>
  detail::TensorInterface<const T, BlockRange> block(
      const PairRange& bounds) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange> &&
                                        !std::is_same_v<PairRange, Range>>>
  detail::TensorInterface<T, BlockRange> block(const PairRange& bounds) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by its \p bounds.

  /// Examples of using this:
  /// \code
  ///   auto tview0 = t.block({{0,4}, {1,6}, {2,8}});
  /// \endcode
  /// \tparam Index An integral type
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  detail::TensorInterface<const T, BlockRange> block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }

  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  detail::TensorInterface<T, BlockRange> block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by a TiledArray::Range.

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///
  ///   auto tview = t.block(TiledArray::Range(lobounds, upbounds));
  /// \endcode
  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v)
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  detail::TensorInterface<const T, BlockRange> block(
      const Range& bounds) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, bounds.lobound(), bounds.upbound()),
        this->data());
  }

  detail::TensorInterface<T, BlockRange> block(const Range& bounds) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, bounds.lobound(), bounds.upbound()),
        this->data());
  }
  /// @}

  /// Create a permuted copy of this tensor

  /// \tparam Perm A permutation tile
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor permute(const Perm& perm) const {
    if constexpr (is_arena_tensor_v<value_type>) {
      // View inner cells cannot be permuted in place; the owning tile
      // rewrites its slab(s). The outer cells reorder shallowly (the 8-byte
      // views are reindexed, the slab is shared via keep-alive); a
      // non-trivial inner permutation rewrites every cell into a fresh slab.
      // The generic Tensor(other, perm) ctor's allocate-then-fill shape does
      // not fit the arena slab model, so route around it.
      const auto outer_perm = outer(perm);
      Tensor result =
          (outer_perm && !outer_perm.is_identity())
              ? detail::arena_permute_shallow<Tensor>(*this, outer_perm)
              : *this;
      if constexpr (detail::is_bipartite_permutation_v<Perm>) {
        const auto inner_perm = inner(perm);
        if (inner_perm && !inner_perm.is_identity())
          result = detail::arena_inner_permute<Tensor>(result, inner_perm);
      }
      return result;
    } else {
      return Tensor(*this, perm);
    }
  }

  /// Shift the lower and upper bound of this tensor

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A reference to this tensor
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  Tensor& shift_to(const Index& bound_shift) {
// although shift_to is currently fine on shared objects since ranges are
// not shared, this will change in the future
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
    this->range_.inplace_shift(bound_shift);
    return *this;
  }

  /// Shift the lower and upper bound of this tensor

  /// \tparam Integer An integral type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A reference to this tensor
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  Tensor& shift_to(const std::initializer_list<Integer>& bound_shift) {
    // although shift_to is currently fine on shared objects since ranges are
    // not shared, this will change in the future
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
    this->range_.template inplace_shift<std::initializer_list<Integer>>(
        bound_shift);
    return *this;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A shifted copy of this tensor
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  Tensor shift(const Index& bound_shift) const {
    Tensor result = clone();
    result.shift_to(bound_shift);
    return result;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Integer An integral type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A shifted copy of this tensor
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  Tensor shift(const std::initializer_list<Integer>& bound_shift) const {
    Tensor result = clone();
    result.template shift_to<std::initializer_list<Integer>>(bound_shift);
    return result;
  }

  // Generic vector operations

  /// Use a binary, element wise operation to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary, element-wise operation
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i],other[i])
  template <typename Right, typename Op,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  auto binary(const Right& right, Op&& op) const {
    using result_value_type = decltype(op(
        std::declval<const T&>(), std::declval<const value_t<Right>&>()));
    using result_allocator_type = typename std::allocator_traits<
        Allocator>::template rebind_alloc<result_value_type>;
    using ResultTensor = Tensor<result_value_type, result_allocator_type>;
    return ResultTensor(*this, right, op);
  }

  /// Use a binary, element wise operation to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \tparam Perm A permutation tile
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary element-wise operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i],other[i])
  template <typename Right, typename Op, typename Perm,
            typename std::enable_if<is_tensor<Right>::value &&
                                    detail::is_permutation_v<
                                        std::remove_reference_t<Perm>>>::type* =
                nullptr>
  auto binary(const Right& right, Op&& op, Perm&& perm) const {
    using result_value_type = decltype(op(
        std::declval<const T&>(), std::declval<const value_t<Right>&>()));
    using result_allocator_type = typename std::allocator_traits<
        Allocator>::template rebind_alloc<result_value_type>;
    using ResultTensor = Tensor<result_value_type, result_allocator_type>;
    // tile ops pass bipartite permutations here even if the result is a plain
    // tensor
    [[maybe_unused]] constexpr bool is_bperm =
        detail::is_bipartite_permutation_v<Perm>;
    constexpr bool result_is_tot = detail::is_tensor_of_tensor_v<ResultTensor>;

    if constexpr (!result_is_tot) {
      if constexpr (is_bperm) {
        TA_ASSERT(!inner(perm));  // ensure this is a plain permutation since
                                  // ResultTensor is plain
        return ResultTensor(*this, right, op, outer(std::forward<Perm>(perm)));
      } else
        return ResultTensor(*this, right, op, std::forward<Perm>(perm));
    } else {
      // AFAIK the other branch fundamentally relies on raw pointer arithmetic,
      // which won't work for ToTs.
      auto temp = binary(right, std::forward<Op>(op));
      Permute<decltype(temp), decltype(temp)> p;
      return p(temp, std::forward<Perm>(perm));
    }
    abort();  // unreachable
  }

  /// Use a binary, element wise operation to modify this tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary element-wise operation
  /// \return A reference to this object
  /// \throw TiledArray::Exception When this tensor is empty.
  /// \throw TiledArray::Exception When \c other is empty.
  /// \throw TiledArray::Exception When the range of this tensor is not equal
  /// to the range of \c other.
  /// \throw TiledArray::Exception When this and \c other are the same.
  template <typename Right, typename Op,
            typename std::enable_if<detail::is_nested_tensor_v<Right>>::type* =
                nullptr>
  Tensor& inplace_binary(const Right& right, Op&& op) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
    detail::inplace_tensor_op(op, *this, right);
    return *this;
  }

  /// Use a unary, element wise operation to construct a new tensor

  /// \tparam Op The unary operation type
  /// \param op The unary element-wise operation
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i])
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  Tensor unary(Op&& op) const& {
    return Tensor(*this, op);
  }

  /// Use a unary, element wise operation to construct a new tensor

  /// \tparam Op The unary operation type
  /// \param op The unary element-wise operation
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i])
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  Tensor unary(Op&& op) && {
    inplace_unary(std::forward<Op>(op));
    return std::move(*this);
  }

  /// Use a unary, element wise operation to construct a new, permuted tensor

  /// \tparam Op The unary operation type
  /// \tparam Perm A permutation tile
  /// \param op The unary element-wise operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted tensor with elements that have been modified by \c op
  /// \throw TiledArray::Exception When this tensor is empty.
  /// \throw TiledArray::Exception The dimension of \c perm does not match
  /// that of this tensor.
  template <typename Op, typename Perm,
            typename = std::enable_if_t<
                detail::is_permutation_v<std::remove_reference_t<Perm>>>>
  Tensor unary(Op&& op, Perm&& perm) const {
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    [[maybe_unused]] constexpr bool is_bperm =
        detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    if constexpr (!is_tot) {
      if (empty()) return {};
      if constexpr (is_bperm) {
        TA_ASSERT(inner_size(perm) == 0);  // ensure this is a plain permutation
        return Tensor(*this, op, outer(std::forward<Perm>(perm)));
      } else
        return Tensor(*this, op, std::forward<Perm>(perm));
    } else {
      auto temp = unary(std::forward<Op>(op));
      Permute<Tensor, Tensor> p;
      return p(temp, std::forward<Perm>(perm));
    }
    abort();  // unreachable
  }

  /// Use a unary, element wise operation to modify this tensor

  /// \tparam Op The unary operation type
  /// \param op The unary, element-wise operation
  /// \return A reference to this object
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  Tensor& inplace_unary(Op&& op) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
    detail::inplace_tensor_op(op, *this);
    return *this;
  }

  // Scale operation

  /// Construct a scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor scale(const Scalar factor) const& {
    // early exit for empty this
    if (empty()) return {};

    if constexpr (detail::is_tensor_of_tensor_v<Tensor> &&
                  detail::is_ta_tensor_v<value_type>) {
      auto fill = [factor](typename value_type::value_type* dst,
                           const typename value_type::value_type* src,
                           std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] * factor;
      };
      return detail::arena_trivial_unary<Tensor>(*this, fill);
    } else if constexpr (is_arena_tensor_v<value_type>) {
      auto fill = [factor](typename value_type::value_type* dst,
                           const typename value_type::value_type* src,
                           std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] * factor;
      };
      return detail::arena_trivial_unary<Tensor>(*this, fill);
    } else {
      return unary([factor](const value_type& a) {
        using namespace TiledArray::detail;
        return a * factor;
      });
    }
  }

  /// Construct a scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor scale(const Scalar factor) && {
    scale_to(factor);
    return std::move(*this);
  }

  /// Construct a scaled and permuted copy of this tensor

  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation tile
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor and permuted
  template <typename Scalar, typename Perm,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar> &&
                                        detail::is_permutation_v<Perm>>>
  Tensor scale(const Scalar factor, const Perm& perm) const {
    // early exit for empty this
    if (empty()) return {};

    if constexpr (is_arena_tensor_v<value_type>) {
      // Arena inner cells: scale via the arena kernel (which manages the slab),
      // then apply the result permutation if non-trivial. Mirrors the arena
      // add(right, perm) overload above. ArenaTensor is also a view, so this
      // branch must precede the view branch below.
      auto result = scale(factor);
      return arena_perm_is_trivial(perm) ? result : result.permute(perm);
    } else if constexpr (is_tensor_view_v<value_type>) {
      TA_EXCEPTION(
          "Tensor<View>::scale(factor, perm): permutation is not "
          "supported for view inner cells");
      return Tensor{};
    } else {
      return unary(
          [factor](const value_type& a) {
            using namespace TiledArray::detail;
            return a * factor;
          },
          perm);
    }
  }

  /// Scale this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& scale_to(const Scalar factor) {
    // early exit for empty this
    if (empty()) return *this;

    if constexpr (is_arena_tensor_v<value_type>) {
      // Arena inner cells: route through each cell's own in-place scale_to (the
      // free arena kernel), which handles a ComplexConjugate factor by
      // conjugating each arena scalar in place. Going through `cell *= factor`
      // would instead select the generic operator*=(.., ComplexConjugate) ->
      // detail::conj(cell), which has no value-returning conj for ArenaTensor.
      return inplace_unary(
          [factor](value_type& MADNESS_RESTRICT c) { c.scale_to(factor); });
    } else {
      return inplace_unary(
          [factor](value_type& MADNESS_RESTRICT res) { res *= factor; });
    }
  }

  // Addition operations

  /// Element-wise add for `Tensor<ArenaTensor>` ToT operands. Routes through
  /// the arena binary kernel; inner cells have no `operator+` of their own.
  template <typename Right>
    requires(is_arena_tensor_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type>)
  Tensor add(const Right& right) const {
    if (empty()) return detail::clone_or_cast<Tensor>(right);
    if (right.empty()) return this->clone();
    auto fill = [](typename value_type::value_type* dst,
                   const typename value_type::value_type* l,
                   const typename value_type::value_type* r, std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = l[i] + r[i];
    };
    return detail::arena_trivial_binary<Tensor>(*this, right, fill);
  }

  /// Mixed `Tensor<ArenaTensor> + Tensor<scalar>`: each inner element is
  /// offset by the corresponding outer-cell scalar. Routes through the
  /// arena scaled kernel; no operator+ between ArenaTensor and scalar.
  template <typename Right>
    requires(is_arena_tensor_v<value_type> &&
             detail::is_numeric_v<typename Right::value_type>)
  Tensor add(const Right& right) const {
    if (empty() || right.empty()) return {};
    using ElemT = typename value_type::value_type;
    using Scalar = typename Right::value_type;
    auto fill = [](ElemT* dst, const ElemT* arena, const Scalar& s,
                   std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = arena[i] + s;
    };
    return detail::arena_trivial_scaled<Tensor>(*this, right, fill);
  }

  /// Mixed `Tensor<scalar> + Tensor<ArenaTensor>`: symmetric to above,
  /// result has the same ToT layout as the right operand.
  template <typename Right>
    requires(detail::is_numeric_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type>)
  Right add(const Right& right) const {
    if (empty() || right.empty()) return {};
    using ArenaInner = typename Right::value_type;
    using ElemT = typename ArenaInner::value_type;
    using Scalar = value_type;
    auto fill = [](ElemT* dst, const ElemT* arena, const Scalar& s,
                   std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = s + arena[i];
    };
    return detail::arena_trivial_scaled<Right>(right, *this, fill);
  }

  /// Scaled element-wise add for `Tensor<ArenaTensor>` ToT operands:
  /// `(this + right) * factor`. Routes through the arena binary kernel.
  template <typename Right, typename Scalar>
    requires(is_arena_tensor_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type> &&
             detail::is_numeric_v<Scalar>)
  Tensor add(const Right& right, const Scalar factor) const {
    using ElemT = typename value_type::value_type;
    auto fill = [factor](ElemT* dst, const ElemT* l, const ElemT* r,
                         std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = (l[i] + r[i]) * factor;
    };
    return detail::arena_trivial_binary<Tensor>(*this, right, fill);
  }

  /// True if \p perm reorders nothing -- empty or identity. Handles a plain
  /// Permutation and a (bipartite) ToT permutation alike.
  template <typename Perm>
  static bool arena_perm_is_trivial(const Perm& perm) {
    if constexpr (std::is_same_v<Perm, BipartitePermutation>)
      return !static_cast<bool>(perm) ||
             (perm.first().is_identity() && perm.second().is_identity());
    else
      return !static_cast<bool>(perm) || perm.is_identity();
  }

  /// Permuted add for `Tensor<ArenaTensor>` ToT operands. The operands are
  /// congruent by the time a permuted product reaches a tile op, so the
  /// elementwise `add(right)` is valid and `perm` is the result permutation;
  /// `permute` applies it (shallow outer reindex + inner-slab rewrite).
  template <typename Right, typename Perm>
    requires(is_arena_tensor_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type> &&
             detail::is_permutation_v<Perm>)
  Tensor add(const Right& right, const Perm& perm) const {
    auto result = add(right);
    return arena_perm_is_trivial(perm) ? result : result.permute(perm);
  }

  /// Permuted scaled add for `Tensor<ArenaTensor>` ToT operands; see the
  /// permuted-add overload above for the congruent-operand rationale.
  template <typename Right, typename Scalar, typename Perm>
    requires(is_arena_tensor_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type> &&
             detail::is_numeric_v<Scalar> && detail::is_permutation_v<Perm>)
  Tensor add(const Right& right, const Scalar factor, const Perm& perm) const {
    auto result = add(right, factor);
    return arena_perm_is_trivial(perm) ? result : result.permute(perm);
  }

  /// Add this and \c other to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <typename Right>
    requires(is_tensor<Right>::value &&
             detail::sum_convertible_to<value_type, const value_type&,
                                        const value_t<Right>&> &&
             !(is_arena_tensor_v<value_type> &&
               detail::is_numeric_v<typename Right::value_type>) &&
             !(detail::is_numeric_v<value_type> &&
               is_arena_tensor_v<typename Right::value_type>))
  Tensor add(const Right& right) const {
    // early exit for empty right
    if (right.empty()) return this->clone();

    // early exit for empty this
    if (empty()) detail::clone_or_cast<Tensor>(right);

    if constexpr (detail::is_tensor_of_tensor_v<Tensor> &&
                  detail::is_ta_tensor_v<value_type> &&
                  detail::is_ta_tensor_v<typename Right::value_type>) {
      auto fill = [](typename value_type::value_type* dst,
                     const typename value_type::value_type* l,
                     const typename value_type::value_type* r, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = l[i] + r[i];
      };
      return detail::arena_trivial_binary<Tensor>(*this, right, fill);
    } else {
      return binary(
          right,
          [](const value_type& l, const value_t<Right>& r) -> decltype(l + r) {
            if constexpr (detail::is_tensor_v<value_type>) {
              if (l.empty()) {
                if (r.empty())
                  return {};
                else
                  return r.clone();
              } else {
                if (r.empty())
                  return l.clone();
                else
                  return l + r;
              }
            }
            return l + r;
          });
    }
  }

  /// Add this and \c other to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <typename Right>
    requires(is_tensor<Right>::value &&
             detail::addable_to<value_type&, const value_t<Right>&>)
  Tensor add(const Right& right) && {
    add_to(right);
    return std::move(*this);
  }

  /// Add this and \c other to construct a new tensor of type that differs from
  /// this

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <typename Right>
    requires(detail::is_tensor_v<Right> &&
             !detail::sum_convertible_to<value_type, const value_type&,
                                         const value_t<Right>&> &&
             !(is_arena_tensor_v<value_type> &&
               is_arena_tensor_v<typename Right::value_type>) &&
             !(is_arena_tensor_v<value_type> &&
               detail::is_numeric_v<typename Right::value_type>) &&
             !(detail::is_numeric_v<value_type> &&
               is_arena_tensor_v<typename Right::value_type>))
  auto add(const Right& right) const {
    return binary(right, [](const value_type& l, const value_t<Right>& r) {
      return l + r;
    });
  }

  /// Add this and \c other to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Perm A permutation tile
  /// \param right The tensor that will be added to this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <typename Right, typename Perm>
    requires(is_tensor<Right>::value && detail::is_permutation_v<Perm> &&
             detail::addable<const value_type&, const value_t<Right>&>)
  auto add(const Right& right, const Perm& perm) const {
    return binary(
        right,
        [](const value_type& l, const value_t<Right>& r) { return l + r; },
        perm);
  }

  /// Scale and add this and \c other to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be added to this tensor
  /// \param factor The scaling factor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other, scaled by \c factor
  template <typename Right, typename Scalar>
    requires(is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
             detail::addable<const value_type&, const value_t<Right>&>)
  auto add(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const value_type& l, const value_t<Right>& r) {
                    return (l + r) * factor;
                  });
  }

  /// Scale and add this and \c other to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation tile
  /// \param right The tensor that will be added to this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm>
    requires(is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
             detail::is_permutation_v<Perm> &&
             detail::addable<const value_type&, const value_t<Right>&>)
  auto add(const Right& right, const Scalar factor, const Perm& perm) const {
    return binary(
        right,
        [factor](const value_type& l, const value_t<Right>& r) {
          return (l + r) * factor;
        },
        perm);
  }

  /// Add a constant to a copy of this tensor

  /// \param value The constant to be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c value
  Tensor add(const numeric_type value) const {
    // early exit for empty this
    if (empty()) return {};

    return unary([value](const value_type& a) { return a + value; });
  }

  /// Add a constant to a permuted copy of this tensor

  /// \tparam Perm A permutation tile
  /// \param value The constant to be added to this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c value
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor add(const numeric_type value, const Perm& perm) const {
    // early exit for empty this
    if (empty()) return {};

    return unary([value](const value_type& a) { return a + value; }, perm);
  }

  /// Add \c other to this tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A reference to this tensor
  template <typename Right>
    requires(is_tensor<Right>::value &&
             detail::addable_to<value_type&, const value_t<Right>&>)
  Tensor& add_to(const Right& right) {
    // early exit for empty right
    if (right.empty()) return *this;

    // early exit for empty this
    if (empty()) {
      *this = detail::clone_or_cast<Tensor>(right);
      return *this;
    }

    return inplace_binary(right, [](value_type& MADNESS_RESTRICT l,
                                    const value_t<Right> r) { l += r; });
  }

  /// Add \c other to this tensor, and scale the result

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be added to this tensor
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Right, typename Scalar>
    requires(is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
             detail::addable_to<value_type&, const value_t<Right>&>)
  Tensor& add_to(const Right& right, const Scalar factor) {
    return inplace_binary(
        right, [factor](value_type& MADNESS_RESTRICT l,
                        const value_t<Right> r) { (l += r) *= factor; });
  }

  /// axpy: <tt>result[i] += arg[i] * factor</tt> (factor scales only the
  /// added operand, not the existing result). Distinct from
  /// `add_to(arg, factor)` which has the legacy `(result + arg) * factor`
  /// semantics. Useful as a fused replacement for
  /// `add_to(result, scale(arg, factor))` when the intermediate
  /// materialization is undesirable (e.g. when `value_type` is a view).
  ///
  /// The lambda body dispatches by element type so the same body works
  /// for flat and ToT tensors -- at the leaf (scalar) level it uses
  /// `l += r * factor`; at the cell level it delegates to the cell's
  /// `axpy_to` member (free or member, found via ADL).
  template <typename Right, typename Scalar>
    requires(is_tensor<Right>::value && detail::is_numeric_v<Scalar>)
  Tensor& axpy_to(const Right& right, const Scalar factor) {
    if (right.empty()) return *this;
    if (empty()) {
      *this = detail::clone_or_cast<Tensor>(right);
      this->scale_to(factor);
      return *this;
    }
    return inplace_binary(right,
                          [factor](auto& MADNESS_RESTRICT l, const auto& r) {
                            using L = std::remove_reference_t<decltype(l)>;
                            if constexpr (detail::is_tensor_helper<L>::value) {
                              l.axpy_to(r, factor);
                            } else {
                              l += r * factor;
                            }
                          });
  }

  /// axpy with fused permutation on the added operand:
  /// <tt>result[i] += (perm ^ arg)[i] * factor</tt>.
  ///
  /// Bails for view inner cells (which cannot be permuted in place).
  template <typename Right, typename Scalar, typename Perm>
    requires(is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
             detail::is_permutation_v<Perm>)
  Tensor& axpy_to(const Right& right, const Scalar factor, const Perm& perm) {
    if (right.empty()) return *this;
    if constexpr (is_tensor_view_v<value_type>) {
      TA_EXCEPTION(
          "Tensor<View>::axpy_to(right, factor, perm): inner "
          "permutation is not supported for view inner cells");
      return *this;
    } else {
      auto permuted = right.permute(perm);
      if (empty()) {
        // first contribution into an unallocated target (e.g. a contraction
        // result inner cell): initialize to factor * (perm ^ arg) rather
        // than asserting non-empty in inplace_binary -- mirrors the
        // non-permuting axpy_to overload above.
        *this = detail::clone_or_cast<Tensor>(permuted);
        this->scale_to(factor);
        return *this;
      }
      return inplace_binary(
          permuted, [factor](auto& MADNESS_RESTRICT l, const auto& r) {
            using L = std::remove_reference_t<decltype(l)>;
            if constexpr (detail::is_tensor_helper<L>::value) {
              l.axpy_to(r, factor);
            } else {
              l += r * factor;
            }
          });
    }
  }

  /// Add a constant to this tensor

  /// \param value The constant to be added
  /// \return A reference to this tensor
  template <typename Scalar>
    requires(detail::is_numeric_v<Scalar> &&
             detail::addable_to<value_type&, const Scalar>)
  Tensor& add_to(const Scalar value) {
    return inplace_unary(
        [value](value_type& MADNESS_RESTRICT res) { res += value; });
  }

  // Subtraction operations

  /// Subtract \c right from this and return the result

  /// Element-wise subtraction for `Tensor<ArenaTensor>` ToT operands. Routes
  /// through the arena binary kernel; inner cells have no `operator-`.
  template <typename Right>
    requires(is_arena_tensor_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type>)
  Tensor subt(const Right& right) const {
    auto fill = [](typename value_type::value_type* dst,
                   const typename value_type::value_type* l,
                   const typename value_type::value_type* r, std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = l[i] - r[i];
    };
    return detail::arena_trivial_binary<Tensor>(*this, right, fill);
  }

  /// Mixed `Tensor<ArenaTensor> - Tensor<scalar>`: subtract per-cell scalar
  /// from every inner element. Routes through the arena scaled kernel.
  template <typename Right>
    requires(is_arena_tensor_v<value_type> &&
             detail::is_numeric_v<typename Right::value_type>)
  Tensor subt(const Right& right) const {
    if (empty() || right.empty()) return {};
    using ElemT = typename value_type::value_type;
    using Scalar = typename Right::value_type;
    auto fill = [](ElemT* dst, const ElemT* arena, const Scalar& s,
                   std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = arena[i] - s;
    };
    return detail::arena_trivial_scaled<Tensor>(*this, right, fill);
  }

  /// Mixed `Tensor<scalar> - Tensor<ArenaTensor>`: for each outer cell,
  /// broadcast the scalar minus each inner element of the arena side.
  template <typename Right>
    requires(detail::is_numeric_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type>)
  Right subt(const Right& right) const {
    if (empty() || right.empty()) return {};
    using ArenaInner = typename Right::value_type;
    using ElemT = typename ArenaInner::value_type;
    using Scalar = value_type;
    auto fill = [](ElemT* dst, const ElemT* arena, const Scalar& s,
                   std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = s - arena[i];
    };
    return detail::arena_trivial_scaled<Right>(right, *this, fill);
  }

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be subtracted from this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right
  template <typename Right,
            typename = std::enable_if_t<
                detail::tensors_have_equal_nested_rank_v<Tensor, Right> &&
                !(is_arena_tensor_v<value_type> &&
                  is_arena_tensor_v<typename Right::value_type>)>>
  Tensor subt(const Right& right) const {
    if constexpr (detail::is_tensor_of_tensor_v<Tensor> &&
                  detail::is_ta_tensor_v<value_type> &&
                  detail::is_ta_tensor_v<typename Right::value_type>) {
      auto fill = [](typename value_type::value_type* dst,
                     const typename value_type::value_type* l,
                     const typename value_type::value_type* r, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = l[i] - r[i];
      };
      return detail::arena_trivial_binary<Tensor>(*this, right, fill);
    } else {
      return binary(
          right,
          [](const value_type& l, const value_t<Right>& r) -> decltype(l - r) {
            if constexpr (detail::is_tensor_v<value_type>) {
              if (l.empty()) {
                if (r.empty())
                  return {};
                else
                  return -r;
              } else {
                if (r.empty())
                  return l.clone();
                else
                  return l - r;
              }
            } else {
              return l - r;
            }
          });
    }
  }

  /// Subtract \c right from this and return the result permuted by \c perm

  /// \tparam Right The right-hand tensor type
  /// \tparam Perm A permutation type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right
  template <
      typename Right, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor subt(const Right& right, const Perm& perm) const {
    if constexpr (is_arena_tensor_v<value_type> &&
                  is_arena_tensor_v<typename Right::value_type>) {
      // arena ToT x arena ToT: operands are congruent at tile-op time, so the
      // elementwise `subt(right)` is valid; apply the result permutation as a
      // post-pass (shallow outer reindex + inner-slab rewrite).
      auto result = subt(right);
      return arena_perm_is_trivial(perm) ? result : result.permute(perm);
    } else if constexpr (is_tensor_view_v<value_type>) {
      // Permutation isn't supported for other view inner cells (fixed storage
      // layout). Subt+permute would require materialization.
      TA_EXCEPTION(
          "Tensor<View>::subt(right, perm): permutation is not "
          "supported for view inner cells");
      return Tensor{};
    } else {
      return binary(
          right, [](const value_type& l, const value_type& r) { return l - r; },
          perm);
    }
  }

  /// Subtract \c right from this and return the result scaled by a scaling \c
  /// factor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param factor The scaling factor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right, scaled by \c factor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor subt(const Right& right, const Scalar factor) const {
    if constexpr (is_arena_tensor_v<value_type> &&
                  is_arena_tensor_v<typename Right::value_type>) {
      using ElemT = typename value_type::value_type;
      auto fill = [factor](ElemT* dst, const ElemT* l, const ElemT* r,
                           std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = (l[i] - r[i]) * factor;
      };
      return detail::arena_trivial_binary<Tensor>(*this, right, fill);
    } else {
      return binary(right, [factor](const value_type& l, const value_type& r) {
        return (l - r) * factor;
      });
    }
  }

  /// Subtract \c right from this and return the result scaled by a scaling \c
  /// factor and permuted by \c perm

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm,
            typename std::enable_if<
                is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
                detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor subt(const Right& right, const Scalar factor, const Perm& perm) const {
    if constexpr (is_arena_tensor_v<value_type> &&
                  is_arena_tensor_v<typename Right::value_type>) {
      // arena ToT x arena ToT scaled subtraction; see the unscaled permuted
      // subt overload above for the congruent-operand rationale.
      auto result = subt(right, factor);
      return arena_perm_is_trivial(perm) ? result : result.permute(perm);
    } else {
      return binary(
          right,
          [factor](const value_type& l, const value_type& r) {
            return (l - r) * factor;
          },
          perm);
    }
  }

  /// Subtract a constant from a copy of this tensor

  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c value
  Tensor subt(const numeric_type value) const { return add(-value); }

  /// Subtract a constant from a permuted copy of this tensor

  /// \tparam Perm A permutation tile
  /// \param value The constant to be subtracted
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c value
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor subt(const numeric_type value, const Perm& perm) const {
    return add(-value, perm);
  }

  /// Subtract \c right from this tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be subtracted from this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  Tensor& subt_to(const Right& right) {
    // early exit for empty right
    if (right.empty()) return *this;

    return inplace_binary(
        right, [](auto& MADNESS_RESTRICT l, const auto& r) { l -= r; });
  }

  /// Subtract \c right from and scale this tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& subt_to(const Right& right, const Scalar factor) {
    // early exit for empty right
    if (right.empty()) {
      return this->scale_to(factor);
    }

    return inplace_binary(right,
                          [factor](auto& MADNESS_RESTRICT l, const auto& r) {
                            (l -= r) *= factor;
                          });
  }

  /// Subtract a constant from this tensor

  /// \return A reference to this tensor
  Tensor& subt_to(const numeric_type value) { return add_to(-value); }

  // Multiplication operations

  /// Multiply this by \c right to create a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be multiplied by this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right
  /// Element-wise mult for `Tensor<ArenaTensor>` ToT operands. Routes
  /// through the arena binary kernel; inner cells have no `operator*`.
  template <typename Right>
    requires(is_arena_tensor_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type>)
  Tensor mult(const Right& right) const {
    if (empty() || right.empty()) return {};
    auto fill = [](typename value_type::value_type* dst,
                   const typename value_type::value_type* l,
                   const typename value_type::value_type* r, std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = l[i] * r[i];
    };
    return detail::arena_trivial_binary<Tensor>(*this, right, fill);
  }

  /// Mixed `Tensor<ArenaTensor> * Tensor<scalar>`: outer Hadamard, each
  /// inner cell scaled by the corresponding scalar. Routes through the
  /// arena scaled kernel; no operator* between ArenaTensor and scalar.
  template <typename Right>
    requires(is_arena_tensor_v<value_type> &&
             detail::is_numeric_v<typename Right::value_type>)
  Tensor mult(const Right& right) const {
    if (empty() || right.empty()) return {};
    using ElemT = typename value_type::value_type;
    using Scalar = typename Right::value_type;
    auto fill = [](ElemT* dst, const ElemT* arena, const Scalar& s,
                   std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = arena[i] * s;
    };
    return detail::arena_trivial_scaled<Tensor>(*this, right, fill);
  }

  /// Mixed `Tensor<scalar> * Tensor<ArenaTensor>`: symmetric to above,
  /// result has the same ToT layout as the right operand.
  template <typename Right>
    requires(detail::is_numeric_v<value_type> &&
             is_arena_tensor_v<typename Right::value_type>)
  Right mult(const Right& right) const {
    if (empty() || right.empty()) return {};
    using ArenaInner = typename Right::value_type;
    using ElemT = typename ArenaInner::value_type;
    using Scalar = value_type;
    auto fill = [](ElemT* dst, const ElemT* arena, const Scalar& s,
                   std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = s * arena[i];
    };
    return detail::arena_trivial_scaled<Right>(right, *this, fill);
  }

  template <
      typename Right,
      typename std::enable_if<
          detail::is_nested_tensor_v<Right> && !is_arena_tensor_v<value_type> &&
          !is_arena_tensor_v<typename Right::value_type>>::type* = nullptr>
  decltype(auto) mult(const Right& right) const {
    auto mult_op = [](const value_type& l, const value_t<Right>& r) {
      return l * r;
    };

    if (empty() || right.empty()) {
      using res_t = decltype(std::declval<Tensor>().binary(
          std::declval<Right>(), mult_op));
      return res_t{};
    }

    if constexpr (detail::is_tensor_of_tensor_v<Tensor> &&
                  detail::is_ta_tensor_v<value_type> &&
                  detail::is_ta_tensor_v<typename Right::value_type>) {
      auto fill = [](typename value_type::value_type* dst,
                     const typename value_type::value_type* l,
                     const typename value_type::value_type* r, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = l[i] * r[i];
      };
      return detail::arena_trivial_binary<Tensor>(*this, right, fill);
    } else {
      return binary(right, mult_op);
    }
  }

  /// Multiply this by \c right to create a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Perm a permutation type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right
  template <
      typename Right, typename Perm,
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  decltype(auto) mult(const Right& right, const Perm& perm) const {
    if constexpr (is_arena_tensor_v<value_type> &&
                  is_arena_tensor_v<typename Right::value_type>) {
      // arena ToT x arena ToT Hadamard product. By the time a permuted product
      // reaches a tile op, the engine has already brought both operands to a
      // common (congruent) layout, so the elementwise `mult(right)` is valid;
      // `perm` is the result permutation (common layout -> target). Apply it
      // as a post-pass: `permute` reindexes the outer cells shallowly
      // (arena_permute_shallow) and rewrites the inner slab if the inner part
      // of the permutation is non-trivial (arena_inner_permute).
      auto result = mult(right);
      return arena_perm_is_trivial(perm) ? result : result.permute(perm);
    } else if constexpr (detail::is_numeric_v<value_type> &&
                         is_arena_tensor_v<typename Right::value_type>) {
      // t x tot: a plain scalar tile times an arena ToT tile. The 2-arg
      // arena overload scales each inner cell into a fresh slab; a
      // non-trivial result permutation is then a shallow outer reindex of
      // that slab (the inner part is identity for a Hadamard t x tot).
      auto result = mult(right);
      return arena_perm_is_trivial(perm) ? result : result.permute(perm);
    } else if constexpr (is_arena_tensor_v<value_type> &&
                         detail::is_numeric_v<typename Right::value_type>) {
      // tot x t: the mirror of the above -- an arena ToT tile times a plain
      // scalar tile. Same slab-then-reindex handling.
      auto result = mult(right);
      return arena_perm_is_trivial(perm) ? result : result.permute(perm);
    } else {
      return binary(
          right,
          [](const value_type& l, const value_t<Right>& r) { return l * r; },
          perm);
    }
  }

  /// Scale and multiply this by \c right to create a new tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right, scaled by \c factor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  decltype(auto) mult(const Right& right, const Scalar factor) const {
    if constexpr (is_arena_tensor_v<value_type> &&
                  is_arena_tensor_v<typename Right::value_type>) {
      using ElemT = typename value_type::value_type;
      auto fill = [factor](ElemT* dst, const ElemT* l, const ElemT* r,
                           std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = (l[i] * r[i]) * factor;
      };
      return detail::arena_trivial_binary<Tensor>(*this, right, fill);
    } else {
      return binary(right,
                    [factor](const value_type& l, const value_t<Right>& r) {
                      return (l * r) * factor;
                    });
    }
  }

  /// Scale and multiply this by \c right to create a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right, scaled by \c factor
  template <
      typename Right, typename Scalar, typename Perm,
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_numeric_v<Scalar> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  decltype(auto) mult(const Right& right, const Scalar factor,
                      const Perm& perm) const {
    if constexpr (is_arena_tensor_v<value_type> &&
                  is_arena_tensor_v<typename Right::value_type>) {
      // arena ToT x arena ToT scaled Hadamard product; see the unscaled
      // permuted mult overload above for the congruent-operand rationale.
      // Scale during the elementwise product, then permute the result.
      auto result = mult(right, factor);
      return arena_perm_is_trivial(perm) ? result : result.permute(perm);
    } else {
      return binary(
          right,
          [factor](const value_type& l, const value_t<Right>& r) {
            return (l * r) * factor;
          },
          perm);
    }
  }

  /// Multiply this tensor by \c right

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be multiplied by this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<detail::is_nested_tensor_v<Right>>::type* =
                nullptr>
  Tensor& mult_to(const Right& right) {
    // early exit for empty right
    if (right.empty()) {
      *this = Tensor{};
      return *this;
    }

    // early exit for empty this
    if (empty()) return *this;

    return inplace_binary(right, [](value_type& MADNESS_RESTRICT l,
                                    const value_t<Right>& r) { l *= r; });
  }

  /// Scale and multiply this tensor by \c right

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& mult_to(const Right& right, const Scalar factor) {
    // early exit for empty this
    if (empty()) return *this;

    return inplace_binary(
        right, [factor](value_type& MADNESS_RESTRICT l,
                        const value_t<Right>& r) { (l *= r) *= factor; });
  }

  // Negation operations

  /// Create a negated copy of this tensor

  /// \return A new tensor that contains the negative values of this tensor
  Tensor neg() const {
    // early exit for empty this
    if (empty()) return this->clone();

    if constexpr (is_arena_tensor_v<value_type>) {
      Tensor result = this->clone();
      result.scale_to(numeric_type(-1));
      return result;
    } else {
      return unary([](const value_type r) { return -r; });
    }
  }

  /// Create a negated and permuted copy of this tensor

  /// \tparam Perm A permutation type
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor that contains the negative values of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor neg(const Perm& perm) const {
    // early exit for empty this
    if (empty()) return this->clone();

    if constexpr (is_tensor_view_v<value_type>) {
      // View cells cannot be permuted in place (size-fixed); permute is
      // intentionally not supported here.
      TA_EXCEPTION(
          "Tensor<View>::neg(perm): permutation is not supported "
          "for view inner cells");
      return Tensor{};
    } else {
      return unary([](const value_type l) { return -l; }, perm);
    }
  }

  /// Negate elements of this tensor

  /// \return A reference to this tensor
  Tensor& neg_to() {
    // early exit for empty this
    if (empty()) return *this;

    if constexpr (is_tensor_view_v<value_type>) {
      return this->scale_to(numeric_type(-1));
    } else {
      return inplace_unary([](value_type& MADNESS_RESTRICT l) { l = -l; });
    }
  }

  /// Create a complex conjugated copy of this tensor

  /// \return A copy of this tensor that contains the complex conjugate the
  /// values
  Tensor conj() const { return scale(detail::conj_op()); }

  /// Create a complex conjugated and scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A copy of this tensor that contains the scaled complex
  /// conjugate the values
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor conj(const Scalar factor) const {
    return scale(detail::conj_op(factor));
  }

  /// Create a complex conjugated and permuted copy of this tensor

  /// \tparam Perm A permutation type
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor that contains the complex
  /// conjugate values
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor conj(const Perm& perm) const {
    return scale(detail::conj_op(), perm);
  }

  /// Create a complex conjugated, scaled, and permuted copy of this tensor

  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation type
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor that contains the complex
  /// conjugate values
  template <
      typename Scalar, typename Perm,
      typename std::enable_if<detail::is_numeric_v<Scalar> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor conj(const Scalar factor, const Perm& perm) const {
    return scale(detail::conj_op(factor), perm);
  }

  /// Complex conjugate this tensor

  /// \return A reference to this tensor
  Tensor& conj_to() { return scale_to(detail::conj_op()); }

  /// Complex conjugate and scale this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& conj_to(const Scalar factor) {
    return scale_to(detail::conj_op(factor));
  }

  // GEMM operations

  /// Contract this tensor with another tensor

  /// \tparam As Template parameter pack of a tensor type
  /// \tparam V The type of \c alpha scalar
  /// \param A The tensor that will be contracted with this tensor
  /// \param alpha Multiply the result by this constant
  /// \param gemm_helper The *GEMM operation meta data
  /// \return A new tensor which is the result of contracting this tensor with
  /// \c A and scaled by \c alpha
  template <typename... As, typename V>
  Tensor gemm(const Tensor<As...>& A, const V alpha,
              const math::GemmHelper& gemm_helper) const {
    Tensor result;
    result.gemm(*this, A, alpha, gemm_helper);
    return result;
  }

  /// Contract two tensors and accumulate the scaled result to this tensor

  /// GEMM is limited to matrix like contractions. For example, the following
  /// contractions are supported:
  /// \code
  /// C[a,b] = A[a,i,j] * B[i,j,b]
  /// C[a,b] = A[a,i,j] * B[b,i,j]
  /// C[a,b] = A[i,j,a] * B[i,j,b]
  /// C[a,b] = A[i,j,a] * B[b,i,j]
  ///
  /// C[a,b,c,d] = A[a,b,i,j] * B[i,j,c,d]
  /// C[a,b,c,d] = A[a,b,i,j] * B[c,d,i,j]
  /// C[a,b,c,d] = A[i,j,a,b] * B[i,j,c,d]
  /// C[a,b,c,d] = A[i,j,a,b] * B[c,d,i,j]
  /// \endcode
  /// Notice that in the above contractions, the inner and outer indices of
  /// the arguments for exactly two contiguous groups in each tensor and that
  /// each group is in the same order in all tensors. That is, the indices of
  /// the tensors must fit the one of the following patterns:
  /// \code
  /// C[M...,N...] = A[M...,K...] * B[K...,N...]
  /// C[M...,N...] = A[M...,K...] * B[N...,K...]
  /// C[M...,N...] = A[K...,M...] * B[K...,N...]
  /// C[M...,N...] = A[K...,M...] * B[N...,K...]
  /// \endcode
  /// This allows use of optimized BLAS functions to evaluate tensor
  /// contractions. Tensor contractions that do not fit this pattern require
  /// one or more tensor permutation so that the tensors fit the required
  /// pattern.
  /// \tparam As Template parameter pack of the left-hand tensor type
  /// \tparam Bs Template parameter pack of the right-hand tensor type
  /// \tparam W The type of the scaling factor
  /// \param A The left-hand tensor that will be contracted
  /// \param B The right-hand tensor that will be contracted
  /// \param alpha The contraction result will be scaled by this value, then
  /// accumulated into \c this
  /// \param gemm_helper The *GEMM operation meta data
  /// \return A reference to \c this
  /// \note if this is uninitialized, i.e., if \c this->empty()==true will
  /// this is equivalent to
  /// \code
  ///   return (*this = left.gemm(right, factor, gemm_helper));
  /// \endcode
  template <typename... As, typename... Bs, typename W>
  Tensor& gemm(const Tensor<As...>& A, const Tensor<Bs...>& B, const W alpha,
               const math::GemmHelper& gemm_helper) {
    numeric_type beta = 1;
    if (this->empty()) {
      *this =
          Tensor(gemm_helper.make_result_range<range_type>(A.range_, B.range()),
                 A.nbatch(), default_construct{true});
      beta = 0;
    }
    TA_ASSERT(this->nbatch() == A.nbatch());
    TA_ASSERT(this->nbatch() == B.nbatch());

    // may need to split gemm into multiply + accumulate for tracing purposes
#ifdef TA_ENABLE_TILE_OPS_LOGGING
    {
      const bool twostep =
          TiledArray::TileOpsLogger<T>::get_instance().gemm &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm_print_contributions;
      std::unique_ptr<T[]> data_copy;
      size_t tile_volume;
      if (twostep) {
        tile_volume = range().volume() * nbatch();
        data_copy = std::make_unique<T[]>(tile_volume);
        std::copy(data_.get(), data_.get() + tile_volume, data_copy.get());
      }
      for (size_t i = 0; i < this->nbatch(); ++i) {
        auto Ci = this->batch(i);
        TiledArray::gemm(alpha, A.batch(i), B.batch(i),
                         twostep ? numeric_type(0) : numeric_type(1), Ci,
                         gemm_helper);
      }
      if (TiledArray::TileOpsLogger<T>::get_instance_ptr() != nullptr &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm) {
        auto& logger = TiledArray::TileOpsLogger<T>::get_instance();
        auto apply = [](auto& fnptr, const Range& arg) {
          return fnptr ? fnptr(arg) : arg;
        };
        auto tformed_left_range =
            apply(logger.gemm_left_range_transform, A.range());
        auto tformed_right_range =
            apply(logger.gemm_right_range_transform, B.range());
        auto tformed_result_range =
            apply(logger.gemm_result_range_transform, this->range_);
        if ((!logger.gemm_result_range_filter ||
             logger.gemm_result_range_filter(tformed_result_range)) &&
            (!logger.gemm_left_range_filter ||
             logger.gemm_left_range_filter(tformed_left_range)) &&
            (!logger.gemm_right_range_filter ||
             logger.gemm_right_range_filter(tformed_right_range))) {
          logger << "TA::Tensor::gemm+: left=" << tformed_left_range
                 << " right=" << tformed_right_range
                 << " result=" << tformed_result_range << std::endl;
          if (TiledArray::TileOpsLogger<T>::get_instance()
                  .gemm_print_contributions) {
            if (!TiledArray::TileOpsLogger<T>::get_instance()
                     .gemm_printer) {  // default printer
              // must use custom printer if result's range transformed
              if (!logger.gemm_result_range_transform)
                logger << *this << std::endl;
              else
                logger << make_map(this->data_.get(), tformed_result_range)
                       << std::endl;
            } else {
              TiledArray::TileOpsLogger<T>::get_instance().gemm_printer(
                  *logger.log, tformed_left_range, A.data(),
                  tformed_right_range, B.data(), tformed_right_range,
                  this->data(), this->nbatch());
            }
          }
        }
      }

      if (twostep) {
        for (size_t v = 0; v != tile_volume; ++v) {
          this->data_.get()[v] += data_copy[v];
        }
      }
    }
#else   // TA_ENABLE_TILE_OPS_LOGGING
    for (size_t i = 0; i < this->nbatch(); ++i) {
      auto Ci = this->batch(i);
      TiledArray::detail::gemm(alpha, A.batch(i), B.batch(i), beta, Ci,
                               gemm_helper);
    }
#endif  // TA_ENABLE_TILE_OPS_LOGGING

    return *this;
  }

  template <typename U, typename AU, typename V, typename AV,
            typename ElementMultiplyAddOp,
            typename = std::enable_if_t<std::is_invocable_r_v<
                void, std::remove_reference_t<ElementMultiplyAddOp>,
                value_type&, const U&, const V&>>>
  Tensor& gemm(const Tensor<U, AU>& left, const Tensor<V, AV>& right,
               const math::GemmHelper& gemm_helper,
               ElementMultiplyAddOp&& elem_muladd_op) {
    // Check that the arguments are not empty and have the correct ranks
    TA_ASSERT(!left.empty());
    TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
    TA_ASSERT(!right.empty());
    TA_ASSERT(right.range().rank() == gemm_helper.right_rank());
    TA_ASSERT(left.nbatch() == right.nbatch());
    const auto batch_sz = left.nbatch();

    // Check that the inner dimensions of left and right match
    TA_ASSERT(gemm_helper.left_right_congruent(left.range().extent_data(),
                                               right.range().extent_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(left.range().lobound_data(),
                                               right.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(left.range().upbound_data(),
                                               right.range().upbound_data()));

    // beta-eligibility probe: a fresh (empty) result tile could use beta=0
    // (and skip zero-init); an existing one needs beta=1 to accumulate.
    [[maybe_unused]] const bool _scale_was_empty = this->empty();
    if (this->empty()) {  // initialize, if empty
      auto result_range = gemm_helper.make_result_range<range_type>(
          left.range(), right.range());
      if constexpr (detail::is_numeric_v<value_type>) {
        // dot_inner denest: the result element is a scalar and elem_muladd_op
        // ACCUMULATES (result += ...), so a freshly-allocated tile must be
        // zero-initialized. The nbatches{} ctor default-initializes, which for
        // a trivially-default-constructible scalar leaves the memory
        // uninitialized -- the muladd would then accumulate onto garbage. Zero
        // the whole (possibly batched) data block. batch_sz > 1 occurs when an
        // outer Hadamard mode is batched into this gemm.
        *this = Tensor(result_range, nbatches{batch_sz});
        std::fill_n(this->data(), this->range_.volume() * batch_sz,
                    value_type{0});
      } else {
        // nested (ToT) result: each inner cell is shaped/initialized by the
        // per-cell op; default-init of the outer tile is fine.
        // N.B. use the explicit nbatches{} tag so a bare integral second
        // argument does not bind to the scalar-fill ctor Tensor(range, value).
        *this = Tensor(result_range, nbatches{batch_sz});
      }
    } else {
      // Check that the outer dimensions of left match the corresponding
      // dimensions in result
      TA_ASSERT(gemm_helper.left_result_congruent(left.range().extent_data(),
                                                  this->range_.extent_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.left_result_congruent(left.range().lobound_data(),
                                                  this->range_.lobound_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.left_result_congruent(left.range().upbound_data(),
                                                  this->range_.upbound_data()));

      // Check that the outer dimensions of right match the corresponding
      // dimensions in result
      TA_ASSERT(gemm_helper.right_result_congruent(right.range().extent_data(),
                                                   this->range_.extent_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.right_result_congruent(
                    right.range().lobound_data(), this->range_.lobound_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.right_result_congruent(
                    right.range().upbound_data(), this->range_.upbound_data()));

      // check that batch size of this matches that of left and right
      TA_ASSERT(this->nbatch() == batch_sz);
    }

    // Compute gemm dimensions
    using integer = TiledArray::math::blas::integer;
    integer M, N, K;
    gemm_helper.compute_matrix_sizes(M, N, K, left.range(), right.range());

    // Get the leading dimension for left and right matrices.
    const integer lda =
        (gemm_helper.left_op() == TiledArray::math::blas::NoTranspose ? K : M);
    const integer ldb =
        (gemm_helper.right_op() == TiledArray::math::blas::NoTranspose ? N : K);

    // GEMM-based ToT scale path: for the scale contraction
    // "m,k;a" * "k,n" -> "m,n;a" (left ToT, right plain scalar), recast each
    // row m as one strided GEMM result_m(A_m x N) += left_m(A_m x K) *
    // right(K x N), directly on the arena slab -- amortizing the per-cell AXPY
    // setup over a single BLAS call. Applies for NoTranspose, matching scalar
    // type, and "clean" rows (all cells present, uniform inner size A_m, laid
    // out as one contiguous single-page stride-A_m block); other rows fall back
    // to the per-cell AXPY loop.
    if constexpr (detail::is_numeric_v<V> && is_tensor_view_v<U> &&
                  is_tensor_view_v<value_type>) {
      using Real = std::remove_cv_t<typename value_type::value_type>;
      if constexpr (std::is_same_v<std::remove_cv_t<V>, Real>) {
        if (gemm_helper.left_op() == TiledArray::math::blas::NoTranspose &&
            gemm_helper.right_op() == TiledArray::math::blas::NoTranspose) {
          // kernel-total timer: destroyed at `return *this;` below, so it
          // captures the whole for-b/for-m loop. loop-residual is derived from
          // it minus the sub-phases.
          detail::ScopedScaleTimer _scale_kt(detail::g_scale[0].kernel_ns);
          if (detail::scale_gemm_timing_enabled())
            (_scale_was_empty ? detail::g_scale[0].calls_firstwrite
                              : detail::g_scale[0].calls_accum)
                .fetch_add(1, std::memory_order_relaxed);
          for (integer b = 0; b != nbatch(); ++b) {
            auto this_data = this->batch_data(b);
            auto left_data = left.batch_data(b);
            auto right_data = right.batch_data(b);  // K x N row-major scalars
            for (integer m = 0; m != M; ++m) {
              auto* lc0 = left_data + (m * K);  // left cells (m,0..K-1)
              auto* rc0 = this_data + (m * N);  // result cells (m,0..N-1)
              // 2-D segment walker (replaces the old all-or-nothing per-row
              // clean gate). Holes live on the ToT side -- the left k-cells
              // (lc0[k]) and the result n-cells (rc0[n]); the plain `right`
              // (K x N scalars) is dense. A cell joins a strided GEMM only if
              // it is present and has the uniform inner size A (SHAPE check).
              // The n (output) axis and the k (contraction) axis are each
              // segmented into maximal present/uniform-A/constant-stride runs;
              // one GEMM is emitted per (n-run x k-run) sub-block (beta=1
              // accumulates across k-runs AND across the nbatch `b` loop). Any
              // (n,k) pair that cannot ride a GEMM (either endpoint absent or
              // size != A) is handled once by the per-cell AXPY residue below
              // -- so no contribution is double-counted.
              //
              // Discover A from the first GEMM-eligible (present) cell. The
              // strided segments require A > 0; a row with no present cell on
              // either side contributes nothing.
              const auto _scale_tcp = detail::scale_phase_start();
              long A = -1;
              for (integer k = 0; k != K; ++k)
                if (!lc0[k].empty()) {
                  A = static_cast<long>(lc0[k].size());
                  break;
                }
              if (A < 0)
                for (integer n = 0; n != N; ++n)
                  if (!rc0[n].empty()) {
                    A = static_cast<long>(rc0[n].size());
                    break;
                  }
              detail::scale_phase_stop(detail::g_scale[0].check_pres_ns,
                                       _scale_tcp);
              if (A <= 0) continue;  // row entirely absent: nothing to write
              const integer Ai = static_cast<integer>(A);

              // GEMM-eligibility per cell: present AND inner size == A. (Cells
              // that are present but ragged are excluded from segments and fall
              // to the per-cell residue, which the element op shapes correctly.)
              auto k_elig = [&](integer k) {
                return !lc0[k].empty() && static_cast<long>(lc0[k].size()) == A;
              };
              auto n_elig = [&](integer n) {
                return !rc0[n].empty() && static_cast<long>(rc0[n].size()) == A;
              };

              // For each maximal present-n run [n0,n1) (constant result n-
              // stride sc, single page) and present-k run [k0,k1) (constant
              // left k-stride sb, single page) emit ONE strided GEMM:
              //   C2(Nseg x A) += right_sub^T(Nseg x Kseg) * L2(Kseg x A),
              // right_sub = right_data + k0*N + n0 (ld=N, the full plain row
              // stride), L2 = lc0[k0].data() (ld=sb), C2 = rc0[n0].data()
              // (ld=sc). Strides are recomputed LOCALLY per segment (a segment
              // breaks on a page jump / overlap / non-constant stride), mirror-
              // ing arena_strided_dgemm_ce_ce_right.
              const auto _scale_tcs = detail::scale_phase_start();
              integer n0 = 0;
              while (n0 != N) {
                if (!n_elig(n0)) {
                  ++n0;
                  continue;
                }
                // Grow the maximal constant-stride present-n run [n0,n1).
                Real* cstart = rc0[n0].data();
                integer n1 = n0 + 1;
                long sc = A;  // single-cell run defaults to the inner size
                while (n1 != N && n_elig(n1)) {
                  const long dC =
                      static_cast<long>(rc0[n1].data() - cstart);
                  const long off = static_cast<long>(n1 - n0);
                  if (off == 1) {
                    sc = dC;
                    if (sc < A) break;  // page jump / overlap
                  } else if (dC != off * sc) {
                    break;
                  }
                  ++n1;
                }
                const integer Nseg = n1 - n0;
                const integer ldc = (Nseg > 1) ? static_cast<integer>(sc) : Ai;

                // Within this n-run, segment the k axis the same way.
                integer k0 = 0;
                while (k0 != K) {
                  if (!k_elig(k0)) {
                    ++k0;
                    continue;
                  }
                  const Real* L2 = lc0[k0].data();
                  integer k1 = k0 + 1;
                  long sb = A;
                  while (k1 != K && k_elig(k1)) {
                    const long dB =
                        static_cast<long>(lc0[k1].data() - L2);
                    const long off = static_cast<long>(k1 - k0);
                    if (off == 1) {
                      sb = dB;
                      if (sb < A) break;  // page jump / overlap
                    } else if (dB != off * sb) {
                      break;
                    }
                    ++k1;
                  }
                  const integer Kseg = k1 - k0;
                  const integer ldb =
                      (Kseg > 1) ? static_cast<integer>(sb) : Ai;
                  if (detail::scale_gemm_timing_enabled()) {
                    detail::g_scale[0].gemm_runs.fetch_add(
                        1, std::memory_order_relaxed);
                    detail::g_scale[0].gemm_flop.fetch_add(
                        2ull * static_cast<std::uint64_t>(Kseg) *
                            static_cast<std::uint64_t>(Nseg) *
                            static_cast<std::uint64_t>(A),
                        std::memory_order_relaxed);
                  }
#ifdef TA_STRIDED_DGEMM_COUNT
                  detail::g_scale_strided_calls[0].fetch_add(
                      1, std::memory_order_relaxed);
#endif
                  detail::ScopedScaleTimer _scale_gt(detail::g_scale[0].gemm_ns);
                  TiledArray::math::blas::gemm(
                      TiledArray::math::blas::Transpose,
                      TiledArray::math::blas::NoTranspose,
                      /*M=*/Nseg, /*N=*/Ai, /*K=*/Kseg, Real(1),
                      /*A=*/right_data + (static_cast<std::ptrdiff_t>(k0) * N +
                                          n0),
                      /*lda=*/N,
                      /*B=*/L2, /*ldb=*/ldb, Real(1),
                      /*C=*/cstart, /*ldc=*/ldc);
                  k0 = k1;
                }
                n0 = n1;
              }
              detail::scale_phase_stop(detail::g_scale[0].check_str_ns,
                                       _scale_tcs);

              // Per-cell AXPY residue: handle exactly the (n,k) pairs NOT
              // covered by a GEMM segment. The only pairs that need genuine
              // per-cell math are those touching a PRESENT-but-ragged cell
              // (inner size != A) -- a pair with an absent left k-cell or absent
              // result n-cell contributes nothing (the operand skip / no result
              // storage), so it is not a "revert". We therefore run the residue
              // loop (and count it as a fallback) only when a present-but-ragged
              // cell exists. Pairs where both endpoints are GEMM-eligible already
              // rode a segment above and are skipped here -> no double count.
              bool ragged = false;
              for (integer k = 0; k != K && !ragged; ++k)
                if (!lc0[k].empty() && static_cast<long>(lc0[k].size()) != A)
                  ragged = true;
              for (integer n = 0; n != N && !ragged; ++n)
                if (!rc0[n].empty() && static_cast<long>(rc0[n].size()) != A)
                  ragged = true;
              if (ragged) {
                if (detail::scale_gemm_timing_enabled()) {
                  std::uint64_t fl = 0;
                  for (integer n = 0; n != N; ++n) {
                    if (n_elig(n) || rc0[n].empty()) continue;
                    for (integer k = 0; k != K; ++k)
                      if (!lc0[k].empty())
                        fl +=
                            2ull * static_cast<std::uint64_t>(rc0[n].size());
                  }
                  detail::g_scale[0].fb_runs.fetch_add(
                      1, std::memory_order_relaxed);
                  detail::g_scale[0].fb_flop.fetch_add(
                      fl, std::memory_order_relaxed);
                  detail::g_scale[0].fb_ragged.fetch_add(
                      1, std::memory_order_relaxed);
                }
                detail::ScopedScaleTimer _scale_fb(detail::g_scale[0].fb_ns);
                for (integer n = 0; n != N; ++n) {
                  // No result cell -> no storage to accumulate into; the
                  // contribution targets a non-existent cell, so drop it (the
                  // scale element op contract requires a present result).
                  if (rc0[n].empty()) continue;
                  const bool ne = n_elig(n);
                  auto c_offset = m * N + n;
                  for (integer k = 0; k != K; ++k) {
                    if (ne && k_elig(k)) continue;  // rode a GEMM segment
                    elem_muladd_op(*(this_data + c_offset),
                                   *(left_data + (m * K + k)),
                                   *(right_data + (k * N + n)));
                  }
                }
              }
            }
          }
          return *this;
        }
      }
    }

    // GEMM-based scale path, mirror for T * ToT ("m,k" * "k,n;a" -> "m,n;a",
    // left plain scalar, right ToT). Per column n: one GEMM
    // result_n(M x A_n) += left(M x K) * right_n(K x A_n). The right/result
    // column-n cells are strided over the slab (constant k-/m-stride within a
    // single arena page); verify that, else fall back to per-cell AXPY.
    if constexpr (detail::is_numeric_v<U> && is_tensor_view_v<V> &&
                  is_tensor_view_v<value_type>) {
      using Real = std::remove_cv_t<typename value_type::value_type>;
      if constexpr (std::is_same_v<std::remove_cv_t<U>, Real>) {
        if (gemm_helper.left_op() == TiledArray::math::blas::NoTranspose &&
            gemm_helper.right_op() == TiledArray::math::blas::NoTranspose) {
          // kernel-total timer (see tot_x_t block); destroyed at `return`.
          detail::ScopedScaleTimer _scale_kt(detail::g_scale[1].kernel_ns);
          if (detail::scale_gemm_timing_enabled())
            (_scale_was_empty ? detail::g_scale[1].calls_firstwrite
                              : detail::g_scale[1].calls_accum)
                .fetch_add(1, std::memory_order_relaxed);
          for (integer b = 0; b != nbatch(); ++b) {
            auto this_data = this->batch_data(b);
            auto left_data = left.batch_data(b);    // M x K row-major scalars
            auto right_data = right.batch_data(b);  // K x N ToT
            for (integer n = 0; n != N; ++n) {
              // 2-D segment walker mirror of the tot_x_t arm. Here the ToT
              // (holey) operands are the right k-cells (right_data[k*N+n]) and
              // the result m-cells (this_data[m*N+n]); the plain `left` (M x K
              // scalars) is dense. Per column n, segment the m (result/output)
              // axis and the k (contraction) axis into maximal present/uniform-
              // A/constant-stride runs and emit ONE GEMM per (m-run x k-run)
              // sub-block (beta=1 accumulates across k-runs and the nbatch
              // loop). Residual (m,k) pairs with an ineligible endpoint go to
              // the per-cell AXPY once each (no double count).
              const auto _scale_tcp = detail::scale_phase_start();
              long A = -1;
              for (integer k = 0; k != K; ++k)
                if (!right_data[k * N + n].empty()) {
                  A = static_cast<long>(right_data[k * N + n].size());
                  break;
                }
              if (A < 0)
                for (integer m = 0; m != M; ++m)
                  if (!this_data[m * N + n].empty()) {
                    A = static_cast<long>(this_data[m * N + n].size());
                    break;
                  }
              detail::scale_phase_stop(detail::g_scale[1].check_pres_ns,
                                       _scale_tcp);
              if (A <= 0) continue;  // column entirely absent
              const integer Ai = static_cast<integer>(A);

              auto k_elig = [&](integer k) {
                const auto& c = right_data[k * N + n];
                return !c.empty() && static_cast<long>(c.size()) == A;
              };
              auto m_elig = [&](integer m) {
                const auto& c = this_data[m * N + n];
                return !c.empty() && static_cast<long>(c.size()) == A;
              };

              const auto _scale_tcs = detail::scale_phase_start();
              integer m0 = 0;
              while (m0 != M) {
                if (!m_elig(m0)) {
                  ++m0;
                  continue;
                }
                Real* cstart = this_data[m0 * N + n].data();
                integer m1 = m0 + 1;
                long sc = A;
                while (m1 != M && m_elig(m1)) {
                  const long dC = static_cast<long>(
                      this_data[m1 * N + n].data() - cstart);
                  const long off = static_cast<long>(m1 - m0);
                  if (off == 1) {
                    sc = dC;
                    if (sc < A) break;  // page jump / overlap
                  } else if (dC != off * sc) {
                    break;
                  }
                  ++m1;
                }
                const integer Mseg = m1 - m0;
                const integer ldc = (Mseg > 1) ? static_cast<integer>(sc) : Ai;

                integer k0 = 0;
                while (k0 != K) {
                  if (!k_elig(k0)) {
                    ++k0;
                    continue;
                  }
                  const Real* B = right_data[k0 * N + n].data();
                  integer k1 = k0 + 1;
                  long sb = A;
                  while (k1 != K && k_elig(k1)) {
                    const long dB = static_cast<long>(
                        right_data[k1 * N + n].data() - B);
                    const long off = static_cast<long>(k1 - k0);
                    if (off == 1) {
                      sb = dB;
                      if (sb < A) break;  // page jump / overlap
                    } else if (dB != off * sb) {
                      break;
                    }
                    ++k1;
                  }
                  const integer Kseg = k1 - k0;
                  const integer ldb =
                      (Kseg > 1) ? static_cast<integer>(sb) : Ai;
                  if (detail::scale_gemm_timing_enabled()) {
                    detail::g_scale[1].gemm_runs.fetch_add(
                        1, std::memory_order_relaxed);
                    detail::g_scale[1].gemm_flop.fetch_add(
                        2ull * static_cast<std::uint64_t>(Mseg) *
                            static_cast<std::uint64_t>(Kseg) *
                            static_cast<std::uint64_t>(A),
                        std::memory_order_relaxed);
                  }
#ifdef TA_STRIDED_DGEMM_COUNT
                  detail::g_scale_strided_calls[1].fetch_add(
                      1, std::memory_order_relaxed);
#endif
                  detail::ScopedScaleTimer _scale_gt(detail::g_scale[1].gemm_ns);
                  TiledArray::math::blas::gemm(
                      TiledArray::math::blas::NoTranspose,
                      TiledArray::math::blas::NoTranspose,
                      /*M=*/Mseg, /*N=*/Ai, /*K=*/Kseg, Real(1),
                      /*A=*/left_data + (static_cast<std::ptrdiff_t>(m0) * K +
                                         k0),
                      /*lda=*/K,
                      /*B=*/B, /*ldb=*/ldb, Real(1),
                      /*C=*/cstart, /*ldc=*/ldc);
                  k0 = k1;
                }
                m0 = m1;
              }
              detail::scale_phase_stop(detail::g_scale[1].check_str_ns,
                                       _scale_tcs);

              // Per-cell AXPY residue (each (m,k) once): only pairs touching a
              // PRESENT-but-ragged cell need genuine per-cell math; absent
              // endpoints contribute nothing. Run (and count as fallback) only
              // when a present-but-ragged cell exists. Pairs with both endpoints
              // GEMM-eligible already rode a segment -> no double count.
              bool ragged = false;
              for (integer k = 0; k != K && !ragged; ++k) {
                const auto& c = right_data[k * N + n];
                if (!c.empty() && static_cast<long>(c.size()) != A)
                  ragged = true;
              }
              for (integer m = 0; m != M && !ragged; ++m) {
                const auto& c = this_data[m * N + n];
                if (!c.empty() && static_cast<long>(c.size()) != A)
                  ragged = true;
              }
              if (ragged) {
                if (detail::scale_gemm_timing_enabled()) {
                  std::uint64_t fl = 0;
                  for (integer m = 0; m != M; ++m) {
                    if (m_elig(m)) continue;
                    const auto& cm = this_data[m * N + n];
                    if (cm.empty()) continue;
                    for (integer k = 0; k != K; ++k)
                      if (!right_data[k * N + n].empty())
                        fl += 2ull * static_cast<std::uint64_t>(cm.size());
                  }
                  detail::g_scale[1].fb_runs.fetch_add(
                      1, std::memory_order_relaxed);
                  detail::g_scale[1].fb_flop.fetch_add(
                      fl, std::memory_order_relaxed);
                  detail::g_scale[1].fb_ragged.fetch_add(
                      1, std::memory_order_relaxed);
                }
                detail::ScopedScaleTimer _scale_fb(detail::g_scale[1].fb_ns);
                for (integer m = 0; m != M; ++m) {
                  // Absent result cell -> nothing to accumulate into; drop the
                  // contribution (the scale element op requires a present
                  // result).
                  if (this_data[m * N + n].empty()) continue;
                  const bool me = m_elig(m);
                  auto c_offset = m * N + n;
                  for (integer k = 0; k != K; ++k) {
                    if (me && k_elig(k)) continue;
                    elem_muladd_op(*(this_data + c_offset),
                                   *(left_data + (m * K + k)),
                                   *(right_data + (k * N + n)));
                  }
                }
              }
            }
          }
          return *this;
        }
      }
    }

    for (integer b = 0; b != nbatch(); ++b) {
      auto this_data = this->batch_data(b);
      auto left_data = left.batch_data(b);
      auto right_data = right.batch_data(b);
      for (integer m = 0; m != M; ++m) {
        for (integer n = 0; n != N; ++n) {
          auto c_offset = m * N + n;
          for (integer k = 0; k != K; ++k) {
            auto a_offset =
                gemm_helper.left_op() == TiledArray::math::blas::NoTranspose
                    ? m * lda + k
                    : k * lda + m;
            auto b_offset =
                gemm_helper.right_op() == TiledArray::math::blas::NoTranspose
                    ? k * ldb + n
                    : n * ldb + k;
            elem_muladd_op(*(this_data + c_offset), *(left_data + a_offset),
                           *(right_data + b_offset));
          }
        }
      }
    }

    return *this;
  }

  // Reduction operations

  /// Generalized tensor trace

  /// This function will compute the sum of the hyper diagonal elements of
  /// tensor.
  /// \return The trace of this tensor
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename TileType = Tensor,
            typename = detail::enable_if_trace_is_defined_t<TileType>>
  decltype(auto) trace() const {
    return TiledArray::trace(*this);
  }

  /// Unary reduction operation

  /// Perform an element-wise reduction of the data by
  /// executing <tt>join_op(result, reduce_op(*this[i]))</tt> for each
  /// \c i in the index range of \c this. \c result is initialized to \c
  /// identity. If HAVE_INTEL_TBB is defined, and this is a contiguous tensor,
  /// the reduction will be executed in an undefined order, otherwise will
  /// execute in the order of increasing \c i.
  /// \tparam ReduceOp The reduction operation type
  /// \tparam JoinOp The join operation type
  /// \tparam T a type that can be used as argument to ReduceOp
  /// \param reduce_op The element-wise reduction operation
  /// \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename ReduceOp, typename JoinOp, typename Identity>
  auto reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
              Identity&& identity) const {
    return detail::tensor_reduce(std::forward<ReduceOp>(reduce_op),
                                 std::forward<JoinOp>(join_op),
                                 std::forward<Identity>(identity), *this);
  }

  /// Binary reduction operation

  /// Perform an element-wise binary reduction of the data of \c this and \c
  /// other by executing <tt>join_op(result, reduce_op(*this[i], other[i]))</tt>
  /// for each \c i in the index range of \c this. \c result is initialized to
  /// \c identity. If HAVE_INTEL_TBB is defined, and this is a contiguous
  /// tensor, the reduction will be executed in an undefined order, otherwise
  /// will execute in the order of increasing \c i.
  /// \tparam Right The right-hand argument tensor type
  /// \tparam ReduceOp The reduction operation type
  /// \tparam JoinOp The join operation type
  /// \tparam Identity A type that can be used as argument to ReduceOp
  /// \param other The right-hand argument of the binary reduction
  /// \param reduce_op The element-wise reduction operation
  /// \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename Right, typename ReduceOp, typename JoinOp,
            typename Identity,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  auto reduce(const Right& other, ReduceOp&& reduce_op, JoinOp&& join_op,
              Identity&& identity) const {
    return detail::tensor_reduce(
        std::forward<ReduceOp>(reduce_op), std::forward<JoinOp>(join_op),
        std::forward<Identity>(identity), *this, other);
  }

  /// Sum of elements

  /// \return The sum of all elements of this tensor
  numeric_type sum() const {
    auto sum_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res += arg; };
    return reduce(sum_op, sum_op, numeric_type(0));
  }

  /// Product of elements

  /// \return The product of all elements of this tensor
  numeric_type product() const {
    auto mult_op = [](numeric_type& MADNESS_RESTRICT res,
                      const numeric_type arg) { res *= arg; };
    return reduce(mult_op, mult_op, numeric_type(1));
  }

  /// Square of vector 2-norm

  /// \return The vector norm of this tensor
  scalar_type squared_norm() const {
    auto square_op = [](scalar_type& MADNESS_RESTRICT res,
                        const numeric_type arg) {
      res += TiledArray::detail::squared_norm(arg);
    };
    auto sum_op = [](scalar_type& MADNESS_RESTRICT res, const scalar_type arg) {
      res += arg;
    };
    return reduce(square_op, sum_op, scalar_type(0));
  }

  /// Vector 2-norm

  /// \tparam ResultType return type
  /// \note This evaluates \c std::sqrt(ResultType(this->squared_norm()))
  /// \return The vector norm of this tensor
  template <typename ResultType = scalar_type>
  ResultType norm() const {
    return std::sqrt(static_cast<ResultType>(squared_norm()));
  }

  /// Minimum element

  /// \return The minimum elements of this tensor
  template <typename Numeric = numeric_type>
  numeric_type min(
      typename std::enable_if<
          detail::is_strictly_ordered<Numeric>::value>::type* = nullptr) const {
    auto min_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res = std::min(res, arg); };
    return reduce(min_op, min_op, std::numeric_limits<numeric_type>::max());
  }

  /// Maximum element

  /// \return The maximum elements of this tensor
  template <typename Numeric = numeric_type>
  numeric_type max(
      typename std::enable_if<
          detail::is_strictly_ordered<Numeric>::value>::type* = nullptr) const {
    auto max_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res = std::max(res, arg); };
    return reduce(max_op, max_op, std::numeric_limits<scalar_type>::min());
  }

  /// Absolute minimum element

  /// \return The minimum elements of this tensor
  scalar_type abs_min() const {
    auto abs_min_op = [](scalar_type& MADNESS_RESTRICT res,
                         const numeric_type arg) {
      res = std::min(res, std::abs(arg));
    };
    auto min_op = [](scalar_type& MADNESS_RESTRICT res, const scalar_type arg) {
      res = std::min(res, arg);
    };
    return reduce(abs_min_op, min_op, std::numeric_limits<scalar_type>::max());
  }

  /// Absolute maximum element

  /// \return The maximum elements of this tensor
  scalar_type abs_max() const {
    auto abs_max_op = [](scalar_type& MADNESS_RESTRICT res,
                         const numeric_type arg) {
      res = std::max(res, std::abs(arg));
    };
    auto max_op = [](scalar_type& MADNESS_RESTRICT res, const scalar_type arg) {
      res = std::max(res, arg);
    };
    return reduce(abs_max_op, max_op, scalar_type(0));
  }

  /// Vector dot (not inner!) product

  /// \tparam Right The right-hand tensor type
  /// \param other The right-hand tensor to be reduced
  /// \return The dot product of the this and \c other
  /// If numeric_type is real, this is equivalent to inner product
  /// \sa Tensor::inner_product
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  numeric_type dot(const Right& other) const {
    auto mult_add_op = [](numeric_type& res, const numeric_type l,
                          const numeric_t<Right> r) { res += l * r; };
    auto add_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type value) { res += value; };
    return reduce(other, mult_add_op, add_op, numeric_type(0));
  }

  /// Vector inner product

  /// \tparam Right The right-hand tensor type
  /// \param other The right-hand tensor to be reduced
  /// \return The dot product of the this and \c other
  /// If numeric_type is real, this is equivalent to dot product
  /// \sa Tensor::dot
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  numeric_type inner_product(const Right& other) const {
    auto mult_add_op = [](numeric_type& res, const numeric_type l,
                          const numeric_t<Right> r) {
      res += TiledArray::detail::inner_product(l, r);
    };
    auto add_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type value) { res += value; };
    return reduce(other, mult_add_op, add_op, numeric_type(0));
  }

  /// @return pointer to the PtrRegistry object used for tracing TA::Tensor
  /// lifetime
  /// @warning only nonnull if configured with `TA_TENSOR_MEM_TRACE=ON`
  static PtrRegistry* ptr_registry() {
#ifdef TA_TENSOR_MEM_TRACE
    static PtrRegistry registry;
    return &registry;
#else
    return nullptr;
#endif
  }

#ifdef TA_TENSOR_MEM_TRACE
  /// @param nbytes sets the minimum size of TA::Tensor objects whose lifetime
  /// will be tracked; must be greater or equal to 1
  static void trace_if_larger_than(std::size_t nbytes) {
    TA_ASSERT(nbytes >= 1);
    trace_if_larger_than_ = nbytes;
  }
  /// @return the minimum size of TA::Tensor objects whose lifetime
  /// will be tracked
  static std::size_t trace_if_larger_than() { return trace_if_larger_than_; }
#endif

 private:
#ifdef TA_TENSOR_MEM_TRACE
  static std::size_t trace_if_larger_than_;
#endif

};  // class Tensor

/// \return the number of bytes an `ArenaTensor` view plus its in-arena cell
/// occupy in memory space `S`. `size_of(Tensor<ArenaTensor>)` recurses here
/// once per inner cell; summed over the outer tile this counts the slab.
template <MemorySpace S, typename T, typename R>
std::size_t size_of(const ArenaTensor<T, R>& t) {
  std::size_t result = 0;
  if constexpr (S == MemorySpace::Host) {
    result += sizeof(t);  // the one-pointer view itself
    if (!t.empty()) result += ArenaTensor<T, R>::cell_size(t.size());
  }
  return result;
}

/// \return the number of bytes used by \p t in memory space
/// `S`
template <MemorySpace S, typename T, typename A>
std::size_t size_of(const Tensor<T, A>& t) {
  std::size_t result = 0;
  if constexpr (S == MemorySpace::Host) {
    result += sizeof(t);
  }
  // correct for optional dynamic allocation of Range
  if constexpr (S == MemorySpace::Host) {
    result -= sizeof(Range);
  }
  result += size_of<S>(t.range());

  if (allocates_memory_space<S>(A{})) {
    if (!t.empty()) {
      if constexpr (is_constexpr_size_of_v<S, T>) {
        result += t.size() * sizeof(T);
      } else {
        result += std::accumulate(
            t.begin(), t.end(), std::size_t{0},
            [](const std::size_t s, const T& t) { return s + size_of<S>(t); });
      }
    }
  }
  return result;
}

#ifdef TA_TENSOR_MEM_TRACE
template <typename T, typename A>
std::size_t Tensor<T, A>::trace_if_larger_than_ =
    std::numeric_limits<std::size_t>::max();
#endif

template <typename T, typename A>
Tensor<T, A> operator*(const Permutation& p, const Tensor<T, A>& t) {
  return t.permute(p);
}

// template <typename T, typename A>
// const typename Tensor<T, A>::range_type Tensor<T, A>::empty_range_;

/// equality comparison
/// \param[in] a a Tensor object
/// \param[in] b another Tensor object
/// \return true if ranges and data of \p a and \p b are equal
/// \internal this does not compare nbatch  so any
///           2 empty tensors are equal even if their nbatch
///           differ
template <typename T, typename A>
bool operator==(const Tensor<T, A>& a, const Tensor<T, A>& b) {
  return a.range() == b.range() &&
         std::equal(a.data(), a.data() + a.size(), b.data());
}

/// inequality comparison
/// \param[in] a a Tensor object
/// \param[in] b another Tensor object
/// \return true if ranges and data of \p a and \p b are not equal
template <typename T, typename A>
bool operator!=(const Tensor<T, A>& a, const Tensor<T, A>& b) {
  return !(a == b);
}

namespace detail {

/// Implements taking the trace of a Tensor<T,A>
///
/// \tparam T The type of the elements in the tensor. For this specialization
///           to be considered must satisfy the concept of numeric type.
/// \tparam A The type of the allocator for the tensor
template <typename T, typename A>
struct Trace<Tensor<T, A>, detail::enable_if_numeric_t<T>> {
  decltype(auto) operator()(const Tensor<T, A>& t) const {
    using size_type = typename Tensor<T, A>::size_type;
    using value_type = typename Tensor<T, A>::value_type;
    const auto range = t.range();

    // Get pointers to the range data
    const size_type n = range.rank();
    const auto* MADNESS_RESTRICT const lower = range.lobound_data();
    const auto* MADNESS_RESTRICT const upper = range.upbound_data();
    const auto* MADNESS_RESTRICT const stride = range.stride_data();

    // Search for the largest lower bound and the smallest upper bound
    const size_type lower_max = *std::max_element(lower, lower + n);
    const size_type upper_min = *std::min_element(upper, upper + n);

    value_type result = 0;

    if (lower_max >= upper_min) return result;  // No diagonal element in tile

    // Compute the first and last ordinal index
    size_type first = 0ul, last = 0ul, trace_stride = 0ul;
    for (size_type i = 0ul; i < n; ++i) {
      const size_type lower_i = lower[i];
      const size_type stride_i = stride[i];

      first += (lower_max - lower_i) * stride_i;
      last += (upper_min - lower_i) * stride_i;
      trace_stride += stride_i;
    }

    // Compute the trace
    const value_type* MADNESS_RESTRICT const data = &t[first];
    for (; first < last; first += trace_stride) result += data[first];

    return result;
  }
};

/// specialization of TiledArray::detail::transform for Tensor
template <typename T, typename A>
struct transform<Tensor<T, A>> {
  template <typename Op, typename T1>
  Tensor<T, A> operator()(Op&& op, T1&& t1) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<Op>(op));
  }
  template <typename Op, typename Perm, typename T1,
            typename = std::enable_if_t<
                detail::is_permutation_v<std::remove_reference_t<Perm>>>>
  Tensor<T, A> operator()(Op&& op, Perm&& perm, T1&& t1) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<Op>(op),
                        std::forward<Perm>(perm));
  }
  template <typename Op, typename T1, typename T2>
  Tensor<T, A> operator()(Op&& op, T1&& t1, T2&& t2) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<T2>(t2),
                        std::forward<Op>(op));
  }
  template <typename Op, typename Perm, typename T1, typename T2,
            typename = std::enable_if_t<
                detail::is_permutation_v<std::remove_reference_t<Perm>>>>
  Tensor<T, A> operator()(Op&& op, Perm&& perm, T1&& t1, T2&& t2) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<T2>(t2),
                        std::forward<Op>(op), std::forward<Perm>(perm));
  }
};
}  // namespace detail

namespace detail {

template <typename T, typename A>
struct real_t_impl<Tensor<T, A>> {
  using type = typename Tensor<T, A>::template rebind_numeric_t<
      typename Tensor<T, A>::scalar_type>;
};

template <typename T, typename A>
struct complex_t_impl<Tensor<T, A>> {
  using type = typename Tensor<T, A>::template rebind_numeric_t<
      std::complex<typename Tensor<T, A>::scalar_type>>;
};

}  // namespace detail

#ifndef TILEDARRAY_HEADER_ONLY

extern template class Tensor<double>;
extern template class Tensor<float>;
// extern template class Tensor<int>;
// extern template class Tensor<long>;
extern template class Tensor<std::complex<double>>;
extern template class Tensor<std::complex<float>>;

#endif  // TILEDARRAY_HEADER_ONLY

}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_TENSOR_H__INCLUDED
