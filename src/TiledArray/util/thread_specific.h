
#ifndef TILEDARRAY_UTIL_THREAD_SPECIFIC_H__INCLUDED
#define TILEDARRAY_UTIL_THREAD_SPECIFIC_H__INCLUDED

#include <TiledArray/error.h>

#include <pthread.h>
#include <cerrno>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include <boost/iterator/iterator_facade.hpp>

namespace TiledArray {
namespace detail {

/// Implements a highly-limited subset of tbb::enumerable_thread_specific
/// using Pthread thread-specific storage
template <typename Item>
class thread_specific {
 public:
  template <typename F, typename = std::enable_if_t<!std::is_same<
                            thread_specific, std::decay_t<F>>::value>>
  explicit thread_specific(F &&init)
      : init_(std::forward<F>(init)),
        mtx_(std::make_unique<std::mutex>()),
        key_ptr_(new pthread_key_t, &pthread_key_deleter) {
    auto retcode = pthread_key_create(&key(), NULL);
    // ensure that pthread_key_create sets all thread-specific values to null
    TA_ASSERT(pthread_getspecific(key()) == NULL);
    if (retcode == EAGAIN)
      TA_EXCEPTION(
          "thread_specific could not be constructed due to reaching the "
          "maximum number of thread-specific keys");
    else if (retcode == ENOMEM) {
      TA_EXCEPTION(
          "thread_specific could not be constructed due to a memory allocation "
          "error");
    } else if (retcode != 0)
      TA_EXCEPTION(
          "thread_specific could not be constructed due to an unknown error");
  }

  thread_specific(const thread_specific &) = delete;
  thread_specific(thread_specific &&other) = default;
  thread_specific &operator=(const thread_specific &) = delete;
  thread_specific &operator=(thread_specific &&) = default;
  ~thread_specific() = default;

  const pthread_key_t &key() const { return *key_ptr_; }

  /// @return reference to the thread-local @c Item instance.
  Item &local() { return *(ts_item_ptr().get()); }

 private:
  template <typename>
  friend class TSPool;

  pthread_key_t &key() { return *key_ptr_; }
  static void pthread_key_deleter(pthread_key_t *key_ptr) {
    if (key_ptr) {
      pthread_key_delete(*key_ptr);
      delete key_ptr;
    }
  };

  Item init_;
  std::map<std::thread::id, std::unique_ptr<Item>> items_;
  std::unique_ptr<std::mutex> mtx_;
  std::unique_ptr<pthread_key_t, decltype(&pthread_key_deleter)> key_ptr_;

  const std::unique_ptr<Item> &ts_item_ptr() {
    const std::unique_ptr<Item> *ptr =
        reinterpret_cast<const std::unique_ptr<Item> *>(
            pthread_getspecific(key()));
    if (ptr == nullptr) {
      std::lock_guard<std::mutex> lock{*mtx_};
      const auto thread_id = std::this_thread::get_id();
      auto it = items_.find(thread_id);
      // case 1: there is already an item for this thread id? (e.g. thread id
      // was reused) initialize ts_item_ptr_ so that the next call from this
      // thread is fast
      if (it != items_.end()) {
        // none of the non-TBB MADNESS task backends delete their
        // threads but can't guarantee in general
        ptr = &(it->second);
      }
      // case 2: need to make a new object
      else {
        auto result = items_.emplace(thread_id, std::make_unique<Item>(init_));
        ptr = &(result.first->second);
      }
      pthread_setspecific(key(), (const void *)ptr);
    }
    return *ptr;
  }

 public:
  using iterator = typename decltype(items_)::iterator;
  using const_iterator = typename decltype(items_)::const_iterator;
};

/// A pool of thread-specific objects
template <typename Item>
class TSPool {
 public:
  /// Don't allow copies or default initialization.
  TSPool() = delete;
  TSPool(TSPool const &) = delete;
  TSPool &operator=(TSPool const &) = delete;

  TSPool &operator=(TSPool &&) = default;
  TSPool(TSPool &&a) = default;

  /// Initializes the pool with a single @c Item
  explicit TSPool(Item e) : item_(std::move(e)), items_(item_) {}
  /// @return reference to the thread-local @c Item instance.
  Item &local() { return items_.local(); }

  /// Iterator over thread-local items
  template <typename Item_ = Item>
  class Iterator
      : public boost::iterator_facade<Iterator<Item_>, Item_,
                                      boost::bidirectional_traversal_tag> {
   public:
    Iterator(thread_specific<Item> &items) : item_(items.items_.begin()) {
      static_assert(!std::is_const_v<Item_>);
    }
    Iterator(const thread_specific<Item> &items) : item_(items.items_.begin()) {
      static_assert(std::is_const_v<Item_>);
    }
    Iterator(thread_specific<Item> &items, bool /* end_tag */)
        : item_(items.items_.end()) {
      static_assert(!std::is_const_v<Item_>);
    }
    Iterator(const thread_specific<Item> &items, bool /* end_tag */)
        : item_(items.items_.end()) {
      static_assert(std::is_const_v<Item_>);
    }

   private:
    friend class boost::iterator_core_access;

    using nc_items_type = decltype(thread_specific<Item>::items_);
    using items_type =
        std::conditional_t<std::is_const_v<Item_>,
                           std::add_const_t<nc_items_type>, nc_items_type>;
    std::conditional_t<std::is_const_v<Item_>,
                       typename items_type::const_iterator,
                       typename items_type::iterator>
        item_;  // points to the current item

    void increment() { ++item_; }
    void decrement() { --item_; }
    bool equal(Iterator const &other) const {
      return this->item_ == other.item_;
    }
    Item_ &dereference() const { return *(item_->second); }
  };

  using iterator = Iterator<Item>;
  using const_iterator = Iterator<const Item>;
  auto begin() { return iterator{items_}; }
  auto end() { return iterator{items_, true}; }
  auto begin() const { return const_iterator{items_}; }
  auto end() const { return const_iterator{items_, true}; }

 private:
  Item item_;  //!< used only to initialize the thread-specific items
  thread_specific<Item> items_;
};

template <typename Item>
TSPool<Item> make_tspool(Item e) {
  return TSPool<Item>(std::move(e));
}

template <typename Item>
std::shared_ptr<TSPool<Item>> make_shared_tspool(Item e) {
  return std::make_shared<TSPool<Item>>(std::move(e));
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_UTIL_THREAD_SPECIFIC_H__INCLUDED
