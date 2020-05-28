//
// Created by Eduard Valeyev on 5/12/20.
//

#ifndef TILEDARRAY_SRC_TILEDARRAY_UTIL_FUNCTION_H
#define TILEDARRAY_SRC_TILEDARRAY_UTIL_FUNCTION_H

#include <functional>
#include <memory>
#include <type_traits>

namespace TiledArray {

/// @brief analogous to std::function, but has shallow-copy semantics
template <typename F>
class shared_function {
 public:
  static_assert(std::is_same_v<std::remove_reference_t<F>, F>);
  shared_function() = default;
  shared_function(F &&f) : f_(std::make_shared<F>(std::move(f))) {}
  shared_function(shared_function const &other) { f_ = other.f_; }
  shared_function(shared_function &&) = default;
  shared_function &operator=(shared_function const &) = default;
  shared_function &operator=(shared_function &&) = default;

  operator bool() const { return f_; }

  template <typename... As, typename = std::void_t<decltype(
                                (std::declval<F &>())(std::declval<As>()...))>>
  auto operator()(As &&... as) const {
    return (*f_)(std::forward<As>(as)...);
  }

 private:
  std::shared_ptr<F> f_;
};

template <class F>
shared_function<std::decay_t<F>> make_shared_function(F &&f) {
  return {std::forward<F>(f)};
}

template <class F>
class function_ref;

/// Specialization for function types.
template <class R, class... Args>
class function_ref<R(Args...)> {
 public:
  constexpr function_ref() noexcept = delete;

  /// Creates a `function_ref` which refers to the same callable as `rhs`.
  constexpr function_ref(const function_ref<R(Args...)> &rhs) noexcept =
      default;

  /// Constructs a `function_ref` referring to `f`.
  ///
  /// \synopsis template <typename F> constexpr function_ref(F &&f) noexcept
  template <typename F,
            std::enable_if_t<
                !std::is_same<std::decay_t<F>, function_ref>::value &&
                std::is_invocable_r<R, F &&, Args...>::value> * = nullptr>
  constexpr function_ref(F &&f) noexcept
      : obj_(const_cast<void *>(
            reinterpret_cast<const void *>(std::addressof(f)))) {
    callback_ = [](void *obj, Args... args) -> R {
      return std::invoke(
          *reinterpret_cast<typename std::add_pointer<F>::type>(obj),
          std::forward<Args>(args)...);
    };
  }

  /// Makes `*this` refer to the same callable as `rhs`.
  constexpr function_ref<R(Args...)> &operator=(
      const function_ref<R(Args...)> &rhs) noexcept = default;

  /// Makes `*this` refer to `f`.
  ///
  /// \synopsis template <typename F> constexpr function_ref &operator=(F &&f)
  /// noexcept;
  template <typename F,
            std::enable_if_t<std::is_invocable_r<R, F &&, Args...>::value> * =
                nullptr>
  constexpr function_ref<R(Args...)> &operator=(F &&f) noexcept {
    obj_ = reinterpret_cast<void *>(std::addressof(f));
    callback_ = [](void *obj, Args... args) {
      return std::invoke(
          *reinterpret_cast<typename std::add_pointer<F>::type>(obj),
          std::forward<Args>(args)...);
    };

    return *this;
  }

  /// Swaps the referred callables of `*this` and `rhs`.
  constexpr void swap(function_ref<R(Args...)> &rhs) noexcept {
    std::swap(obj_, rhs.obj_);
    std::swap(callback_, rhs.callback_);
  }

  /// Call the stored callable with the given arguments.
  R operator()(Args... args) const {
    return callback_(obj_, std::forward<Args>(args)...);
  }

 private:
  void *obj_ = nullptr;
  R (*callback_)(void *, Args...) = nullptr;
};

/// Swaps the referred callables of `lhs` and `rhs`.
template <typename R, typename... Args>
constexpr void swap(function_ref<R(Args...)> &lhs,
                    function_ref<R(Args...)> &rhs) noexcept {
  lhs.swap(rhs);
}

/// wraps Op into a shallow-copy copyable handle
/// if Op is an rvalue ref makes an owning handle
/// otherwise makes an non-owning handle
template <typename Op>
auto make_op_shared_handle(Op &&op) {
  constexpr const bool op_has_external_owner = std::is_reference_v<Op>;
  using Op_ = std::remove_reference_t<Op>;
  using result_t =
      std::conditional_t<op_has_external_owner, std::reference_wrapper<Op_>,
                         TiledArray::shared_function<Op_>>;
  if constexpr (op_has_external_owner)
    return result_t(op);
  else
    return make_shared_function<Op_>(std::forward<Op>(op));
}

}  // namespace TiledArray

#endif  // TILEDARRAY_SRC_TILEDARRAY_UTIL_FUNCTION_H
