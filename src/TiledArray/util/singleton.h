/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  util/singleton.h
 *  Mar 11, 2019
 *
 */

#ifndef TILEDARRAY_UTIL_SINGLETON_H__INCLUDED
#define TILEDARRAY_UTIL_SINGLETON_H__INCLUDED

#include <cassert>
#include <memory>

#include <TiledArray/type_traits.h>

namespace TiledArray {

/// @brief Singleton base class
/// To create a singleton class @c A do:
/// \code
/// class A : Singleton<A> {
///   private:
///     friend class Singleton<A>;
///     A(...);  // define (private) constructors
/// };
/// \endcode
/// Here's how to use it:
/// \code
/// A::set_instance(Args...);  // this creates the first instance of A, if A is
/// not default-constructible, otherwise can skip this A& the_instance_ref =
/// A::get_instance();      // throws if the instance of A had not been created
/// A* the_instance_ptr = A::get_instance_ptr();  // returns nullptr if the
/// instance of A had not been created
/// // the instance of A will be destroyed with other static-linkage objects
/// \endcode
template <typename Derived>
class Singleton {
  // can't use std::is_default_constructible since Derived's ctors should be
  // private
  template <typename T, typename Enabler = void>
  struct is_default_constructible_helper : public std::false_type {};
  template <typename T>
  struct is_default_constructible_helper<T, std::void_t<decltype(T{})>>
      : public std::true_type {};
  constexpr static bool derived_is_default_constructible =
      is_default_constructible_helper<Derived>::value;

 public:
  /// @return reference to the instance
  template <typename D = Derived>
  static std::enable_if_t<Singleton<D>::derived_is_default_constructible, D&>
  get_instance() {
    const auto& result_ptr = instance_accessor();
    if (result_ptr != nullptr) return *result_ptr;
    set_instance();
    return *instance_accessor();
  }

  /// @return reference to the instance
  /// @throw std::logic_error if the reference has not been contructed (because
  /// Derived is not default-constructible and set_instance() had not been
  /// called)
  template <typename D = Derived>
  static D& get_instance(
      std::enable_if_t<!Singleton<Derived>::derived_is_default_constructible>* =
          nullptr) {
    const auto& result_ptr = instance_accessor();
    if (result_ptr != nullptr) return *result_ptr;
    throw std::logic_error(
        "TiledArray::Singleton: is not default-constructible and "
        "set_instance() "
        "has not been called");
  }

  /// @return pointer to the instance, or nullptr if it has not yet been
  /// constructed
  template <typename D = Derived>
  static std::enable_if_t<Singleton<D>::derived_is_default_constructible, D*>
  get_instance_ptr() {
    const auto& result_ptr = instance_accessor();
    if (result_ptr != nullptr) return result_ptr.get();
    set_instance();
    return instance_accessor();
  }

  /// @return pointer to the instance, or nullptr if it has not yet been
  /// constructed
  template <typename D = Derived>
  static D* get_instance_ptr(
      std::enable_if_t<!Singleton<Derived>::derived_is_default_constructible>* =
          nullptr) {
    const auto& result_ptr = instance_accessor();
    if (result_ptr != nullptr) return result_ptr.get();
    return nullptr;
  }

  /// Constructs the instance. This must be called if Derived is not
  /// default-constructible.
  /// @tparam Args a parameter pack type
  /// @param args a parameter pack
  template <typename... Args>
  static void set_instance(Args&&... args) {
    TA_ASSERT(instance_accessor() == nullptr);
    instance_accessor() = std::move(
        std::unique_ptr<Derived>(new Derived(std::forward<Args>(args)...)));
  }

 protected:
  template <typename... Args>
  Singleton(Args&&... args) {}  // all constructors are private

  static auto& instance_accessor() {
    static std::unique_ptr<Derived> instance(nullptr);
    return instance;
  }
};

}  // namespace TiledArray

#endif  // TILEDARRAY_UTIL_SINGLETON_H__INCLUDED
