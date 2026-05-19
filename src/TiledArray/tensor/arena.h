/// Arena implementation
#ifndef TILEDARRAY_TENSOR_ARENA_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_H__INCLUDED

#include "TiledArray/config.h"
#include "TiledArray/error.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

namespace TiledArray {
namespace detail {

/// Kill switch: when true, hooks fall back to the legacy heap path.
inline bool& arena_disabled() {
  static bool flag = false;
  return flag;
}

/// One-shot bump allocator; slab is co-owned via aliasing shared_ptrs.
class Arena {
 public:
  explicit Arena(
      std::pmr::memory_resource* mr = std::pmr::new_delete_resource()) noexcept
      : resource_(mr) {
    TA_ASSERT(resource_ != nullptr);
  }

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) noexcept = default;
  Arena& operator=(Arena&&) noexcept = default;
  ~Arena() = default;

  /// Allocate the slab once; zero_init clears it for accumulation kernels.
  /// `alignment` (default `alignof(std::max_align_t)`) is the alignment of
  /// the slab base; pass a larger power-of-two when callers need SIMD-aligned
  /// element pointers at known interior offsets.
  void reserve(std::size_t bytes, bool zero_init = false,
               std::size_t alignment = alignof(std::max_align_t)) {
    TA_ASSERT(capacity_ == 0);
    TA_ASSERT(bytes > 0);
    TA_ASSERT(alignment >= alignof(std::max_align_t));
    TA_ASSERT((alignment & (alignment - 1)) == 0);
    void* raw = resource_->allocate(bytes, alignment);
    auto* mr = resource_;
    auto deleter = [mr, bytes, alignment](std::byte* p) noexcept {
      mr->deallocate(p, bytes, alignment);
    };
    slab_ = std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw),
                                         std::move(deleter));
    capacity_ = bytes;
    cursor_ = 0;
    if (zero_init) std::memset(slab_.get(), 0, bytes);
  }

  /// Aliasing view at a caller-aligned offset.
  template <typename T>
  std::shared_ptr<T[]> slice(std::size_t offset, std::size_t /*n_elem*/) const {
    TA_ASSERT(slab_);
    TA_ASSERT(offset % alignof(T) == 0);
    TA_ASSERT(offset <= capacity_);
    auto* p = reinterpret_cast<T*>(slab_.get() + offset);
    return std::shared_ptr<T[]>(slab_, p);
  }

  /// Bump-allocate n elements of T; result is T-aligned.
  template <typename T>
  std::shared_ptr<T[]> claim(std::size_t n) {
    TA_ASSERT(slab_);
    auto base = reinterpret_cast<std::uintptr_t>(slab_.get() + cursor_);
    auto aligned = (base + alignof(T) - 1) & ~(alignof(T) - 1);
    std::size_t pad = static_cast<std::size_t>(aligned - base);
    std::size_t consumed = pad + n * sizeof(T);
    TA_ASSERT(cursor_ + consumed <= capacity_);
    cursor_ += consumed;
    return std::shared_ptr<T[]>(slab_, reinterpret_cast<T*>(aligned));
  }

  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t cursor() const noexcept { return cursor_; }
  std::size_t remaining() const noexcept { return capacity_ - cursor_; }
  bool empty() const noexcept { return cursor_ == 0; }
  std::pmr::memory_resource* resource() const noexcept { return resource_; }

 private:
  std::pmr::memory_resource* resource_;
  std::shared_ptr<std::byte[]> slab_;
  std::size_t capacity_ = 0;
  std::size_t cursor_ = 0;
};

/// Per-cell offsets and total slab size produced by plan().
struct ArenaPlan {
  std::vector<std::size_t> offsets;
  std::size_t total_bytes = 0;
};

/// Cache-line-floor alignment used by production callers.
inline constexpr std::size_t kArenaCachelineAlign = 128;

/// Round bytes up to a power-of-two alignment.
inline std::size_t arena_align_up(std::size_t bytes,
                                  std::size_t alignment) noexcept {
  return (bytes + alignment - 1) & ~(alignment - 1);
}

/// Pre-walk cells once to compute offsets and total bytes.
template <typename ShapeFn>
ArenaPlan plan(std::size_t N_cells, ShapeFn&& shape_fn,
               std::size_t element_size, std::size_t alignment) {
  ArenaPlan out;
  out.offsets.resize(N_cells);
  std::size_t total = 0;
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    out.offsets[ord] = total;
    auto&& r = shape_fn(ord);
    std::size_t bytes = r.volume() * element_size;
    total += arena_align_up(bytes, alignment);
  }
  out.total_bytes = total;
  return out;
}

/// PMR adapter over an Arena; deallocation is a no-op (slab-owned lifetime).
class ArenaResource final : public std::pmr::memory_resource {
 public:
  explicit ArenaResource(Arena* arena) noexcept : arena_(arena) {
    TA_ASSERT(arena != nullptr);
  }

  Arena* arena() const noexcept { return arena_; }

 protected:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    auto h = arena_->claim<std::byte>(arena_align_up(bytes, alignment));
    return h.get();
  }

  void do_deallocate(void* /*p*/, std::size_t /*bytes*/,
                     std::size_t /*alignment*/) override {}

  bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

 private:
  Arena* arena_;
};

}  // namespace detail
}  // namespace TiledArray

#endif
