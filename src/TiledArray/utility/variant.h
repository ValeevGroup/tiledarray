/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  utility/variant.h
 *  Oct 26, 2015
 *
 */

#ifndef TILEDARRAY_UTILITY_VARIANT_H__INCLUDED
#define TILEDARRAY_UTILITY_VARIANT_H__INCLUDED

#include <typeindex>
#include <type_traits>

namespace TiledArray {
  namespace detail {

    template <typename... Types>
    struct VariantHelper;

    template <typename T, typename...Types>
    struct VariantHelper<T,Types...> {
        template <typename X>
        static void destroy(X* data, std::type_index type_idx) {
          if (type_idx == std::type_index(typeid(T)))
            reinterpret_cast<T*>(data)->~T();
          else
            VariantHelper<Types...>::destroy(data, type_idx);
        }
        template <typename X1, typename X2>
        static void copy(X1* dest_data, const X2* src_data,
                         std::type_index type_idx) {
          if (type_idx == std::type_index(typeid(T)))
            new (dest_data) T(*(reinterpret_cast<T*>(src_data)));
          else
            VariantHelper<Types...>::copy(dest_data, src_data, type_idx);
        }
        template <typename X1, typename X2>
        static void move(X1* dest_data, const X2* src_data,
                         std::type_index type_idx) {
          if (type_idx == std::type_index(typeid(T)))
            new (dest_data) T(std::move(*(reinterpret_cast<T*>(src_data))));
          else
            VariantHelper<Types...>::copy(dest_data, src_data, type_idx);
        }
    };

    template <>
    struct VariantHelper<> {
        template <typename X>
        static void destroy(X* data, std::type_index type_idx) {
          throw std::bad_cast();
        }
        template <typename X1, typename X2>
        static void copy(X1* dest_data, const X2* src_data,
                         std::type_index type_idx) {
          throw std::bad_cast();
        }
        template <typename X1, typename X2>
        static void move(X1* dest_data, const X2* src_data,
                         std::type_index type_idx) {
          throw std::bad_cast();
        }
    };


    template <typename... Types>
    class Variant {
      public:
        Variant() : type_idx_(typeid(void)) {}

        Variant(const Variant& other) : type_idx_(other.type_idx_) {
          VariantHelper<Types...>::destroy(&data_, type_idx_);
          VariantHelper<Types...>::copy(&data_, &other.data_, type_idx_);
        }
        Variant(Variant&& other) : type_idx_(other.type_idx_) {
          VariantHelper<Types...>::destroy(&data_, type_idx_);
          VariantHelper<Types...>::move(&data_, &other.data_, type_idx_);
        }

        bool valid() const {
          return std::type_index(typeid(void)) != type_idx_;
        }

        template <typename T> T& as() {
          if (std::type_index(typeid(T)) == type_idx_)
            return *reinterpret_cast<T*>(&data_);
          else
            throw std::bad_cast();
        }
        template <typename T> const T& as() const {
          if (std::type_index(typeid(T)) == type_idx_)
            return *reinterpret_cast<const T*>(&data_);
          else
            throw std::bad_cast();
        }

        ~Variant() {
          if (valid())
            VariantHelper<Types...>::destroy(&data_, type_idx_);
        }

      private:
        using data_t = typename std::aligned_union<0, Types...>::type;

        data_t data_;
        std::type_index type_idx_;
    };

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_UTILITY_VARIANT_H__INCLUDED
