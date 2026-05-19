/// Arena implementation
#ifndef TILEDARRAY_TENSOR_ARENA_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_H__INCLUDED

#include "TiledArray/config.h"
#include "TiledArray/error.h"

#include <algorithm>
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

/// Default size, in bytes, of a standard arena page. Inner ToT tensors are
/// small (tens to hundreds of elements), so a page packs many of them; the
/// default amortizes the per-page allocation and inter-cell alignment
/// padding. Override at configure time by defining TILEDARRAY_ARENA_PAGE_BYTES.
#ifndef TILEDARRAY_ARENA_PAGE_BYTES
#define TILEDARRAY_ARENA_PAGE_BYTES (64 * 1024)
#endif
inline constexpr std::size_t kArenaDefaultPageBytes =
    TILEDARRAY_ARENA_PAGE_BYTES;

/// Cache-line-floor alignment; also the alignment every standard page base is
/// allocated to, so any per-cell alignment up to this value is satisfied
/// without a dedicated page.
inline constexpr std::size_t kArenaCachelineAlign = 128;

/// Round bytes up to a power-of-two alignment.
inline std::size_t arena_align_up(std::size_t bytes,
                                  std::size_t alignment) noexcept {
  return (bytes + alignment - 1) & ~(alignment - 1);
}

/// Incremental, multi-page bump allocator backing `Tensor<ArenaTensor>` outer
/// tiles. Memory is handed out from a growing list of pages; each page is a
/// stable heap block (the page list may grow, but a page's buffer never
/// moves), which is what keeps the raw `Cell*` of every `ArenaTensor` view
/// valid for the arena's lifetime. Pages are co-owned with the cells via
/// aliasing `shared_ptr`s.
///
/// Two ways to drive it:
///  - up-front: the total size is known, so `reserve_page(total, ...)` lays
///    down a single exact page and subsequent `claim`s pack into it (one
///    contiguous slab, zero tail waste -- what the kernels and einsum use);
///  - incremental: `claim` cells one at a time as their sizes are discovered;
///    a fresh page is appended whenever the current one cannot satisfy a
///    request. A cell larger than a page gets its own dedicated, exactly
///    sized page (which never becomes the bump target).
///
/// Thread-safety: a single `Arena` is built by a single thread. ToT outer
/// tiles are produced one task per tile (`init_tiles` / kernels), each with
/// its own `Arena`, so the bump path is deliberately not synchronized.
class Arena {
 public:
  explicit Arena(
      std::pmr::memory_resource* mr = std::pmr::new_delete_resource(),
      std::size_t page_size = kArenaDefaultPageBytes,
      bool zero_init = false) noexcept
      : resource_(mr), page_size_(page_size), zero_init_(zero_init) {
    TA_ASSERT(resource_ != nullptr);
    TA_ASSERT(page_size_ > 0);
  }

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) noexcept = default;
  Arena& operator=(Arena&&) noexcept = default;
  ~Arena() = default;

  /// Append a page sized exactly `bytes` (the page base is aligned to
  /// `alignment`, bumped to at least `max_align_t`) and make it the current
  /// bump page. Used by the up-front path with `bytes` set to the known
  /// total, and for a single-cell tile so the lone page is exactly sized.
  void reserve_page(std::size_t bytes, std::size_t alignment) {
    TA_ASSERT(bytes > 0);
    std::size_t a = alignment > alignof(std::max_align_t)
                        ? alignment
                        : alignof(std::max_align_t);
    TA_ASSERT((a & (a - 1)) == 0);
    add_page(bytes, a);
    current_ = pages_.size() - 1;
  }

  /// Bump-allocate `bytes` aligned to `alignment`. Tries the current page;
  /// on a miss appends a fresh page -- a dedicated, exactly sized page when
  /// the request exceeds the page size (or needs more alignment than a
  /// standard page base provides), otherwise a standard page that becomes
  /// the new bump target. The returned handle aliases the owning page's
  /// buffer, so it keeps that page alive on its own.
  std::shared_ptr<std::byte[]> claim_bytes(std::size_t bytes,
                                           std::size_t alignment) {
    TA_ASSERT(bytes > 0);
    TA_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0);

    if (current_ != kNoPage) {
      Page& p = pages_[current_];
      const auto base = reinterpret_cast<std::uintptr_t>(p.buffer.get());
      const auto cur = base + p.cursor;
      const auto aligned =
          (cur + alignment - 1) & ~(std::uintptr_t(alignment) - 1);
      const std::size_t pad = static_cast<std::size_t>(aligned - cur);
      if (pad + bytes <= p.capacity - p.cursor) {
        p.cursor += pad + bytes;
        bytes_allocated_ += bytes;
        return std::shared_ptr<std::byte[]>(
            p.buffer, reinterpret_cast<std::byte*>(aligned));
      }
    }

    // Need a fresh page. A page base is at least `kArenaCachelineAlign`-
    // aligned and a fresh cursor is 0, so a standard page needs no padding;
    // an over-large request, or one needing finer alignment than a standard
    // page base, gets a dedicated exactly-sized page.
    if (bytes > page_size_ || alignment > kArenaCachelineAlign) {
      std::size_t a =
          alignment > kArenaCachelineAlign ? alignment : kArenaCachelineAlign;
      Page& d = add_page(bytes, a);
      d.cursor = bytes;  // dedicated: full after this one claim
      bytes_allocated_ += bytes;
      // A dedicated page is never the bump target; `current_` is unchanged.
      return std::shared_ptr<std::byte[]>(d.buffer, d.buffer.get());
    }

    Page& p = add_page(page_size_, kArenaCachelineAlign);
    current_ = pages_.size() - 1;
    p.cursor = bytes;
    bytes_allocated_ += bytes;
    return std::shared_ptr<std::byte[]>(p.buffer, p.buffer.get());
  }

  /// Typed bump-allocate of `n` elements of `T`; result is `T`-aligned.
  template <typename T>
  std::shared_ptr<T[]> claim(std::size_t n) {
    auto h = claim_bytes(n * sizeof(T), alignof(T));
    return std::shared_ptr<T[]>(h, reinterpret_cast<T*>(h.get()));
  }

  std::size_t page_count() const noexcept { return pages_.size(); }
  std::size_t bytes_allocated() const noexcept { return bytes_allocated_; }
  std::size_t bytes_reserved() const noexcept {
    std::size_t s = 0;
    for (const auto& p : pages_) s += p.capacity;
    return s;
  }
  bool empty() const noexcept { return bytes_allocated_ == 0; }
  std::size_t page_size() const noexcept { return page_size_; }
  std::pmr::memory_resource* resource() const noexcept { return resource_; }

 private:
  struct Page {
    std::shared_ptr<std::byte[]> buffer;
    std::size_t capacity = 0;
    std::size_t cursor = 0;
  };

  static constexpr std::size_t kNoPage = static_cast<std::size_t>(-1);

  Page& add_page(std::size_t capacity, std::size_t alignment) {
    void* raw = resource_->allocate(capacity, alignment);
    auto* mr = resource_;
    auto deleter = [mr, capacity, alignment](std::byte* p) noexcept {
      mr->deallocate(p, capacity, alignment);
    };
    Page pg;
    pg.buffer = std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw),
                                             std::move(deleter));
    pg.capacity = capacity;
    if (zero_init_) std::memset(pg.buffer.get(), 0, capacity);
    pages_.push_back(std::move(pg));
    return pages_.back();
  }

  std::pmr::memory_resource* resource_;
  std::size_t page_size_;
  bool zero_init_;
  std::vector<Page> pages_;
  std::size_t current_ = kNoPage;  // index of the current bump page
  std::size_t bytes_allocated_ = 0;
};

/// PMR adapter over an Arena; deallocation is a no-op (page-owned lifetime).
class ArenaResource final : public std::pmr::memory_resource {
 public:
  explicit ArenaResource(Arena* arena) noexcept : arena_(arena) {
    TA_ASSERT(arena != nullptr);
  }

  Arena* arena() const noexcept { return arena_; }

 protected:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    return arena_->claim_bytes(bytes, alignment).get();
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
