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
 */

#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <TiledArray/perm_index.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/range.h>
#include <memory>

namespace TiledArray {

  namespace detail {

    /*
     * TileImpl exists only to enable ADL
     * http://en.wikipedia.org/wiki/Argument-dependent_name_lookup
     *
     * This is necessary because Tile uses TiledArray's intrusive interface, which
     * prevents unqualified function calls on the data member with names that match
     * functions defined in Tile. An example involving add would be: If we tried
     * to call a non-qualified add(tile_) directly from inside Tile name look up
     *would
     * see the member Tile::add() and fail to find the add defined in the data
     * member namespace.
     *
     * TODO clean up interface to automatically deduce return type once we are
     * confident that all our compilers support c++14.
     */
    template <typename T>
    class TileImpl {
      T tile_;

    public:
      using eval_type = T;
      using value_type = T;
      using numeric_type = typename T::numeric_type;
      using size_type = std::size_t;

      TileImpl() = default;
      ~TileImpl() = default;
      TileImpl(TileImpl const &) = default;
      TileImpl(TileImpl &&) = default;
      TileImpl &operator=(TileImpl &&) = default;
      TileImpl &operator=(TileImpl const &) = default;

      TileImpl(T &&t) : tile_{std::move(t)} {}
      TileImpl(T const &t) : tile_{t} {}

      T &tile() { return tile_; }
      T const &tile() const { return tile_; }

      bool empty_() const { return empty(tile_); }
      auto norm_() const -> decltype(norm(tile_)) { return norm(tile_); }
      auto squared_norm_() const -> decltype(squared_norm(tile_)) {
        return squared_norm(tile_);
      }

      T clone_() const { return clone(tile_); }

      T permute_(Permutation const &p) const { return permute(tile_, p); }

      template <typename... Args>
      auto add_(Args &&... args) const
      -> decltype(add(tile_, std::forward<Args>(args)...)) {
        return add(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      T &add_to_(Args &&... args) {
        add_to(tile_, std::forward<Args>(args)...);
        return tile_;
      }

      /*
       * Subtract
       */
      template <typename... Args>
      auto subt_(Args &&... args) const
      -> decltype(subt(tile_, std::forward<Args>(args)...)) {
        return subt(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      T &subt_to_(Args &&... args) {
        subt_to(tile_, std::forward<Args>(args)...);
        return tile_;
      }

      template <typename... Args>
      auto mult_(Args &&... args) const
      -> decltype(mult(tile_, std::forward<Args>(args)...)) {
        return mult(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      T &mult_to_(Args &&... args) {
        mult_to(tile_, std::forward<Args>(args)...);
        return tile_;
      }

      template <typename... Args>
      auto gemm_(Args &&... args) const
      -> decltype(gemm(tile_, std::forward<Args>(args)...)) {
        return gemm(tile_, std::forward<Args>(args)...);
      }

      // Non const version
      template <typename... Args>
      auto gemm_(Args &&... args)
      -> decltype(gemm(tile_, std::forward<Args>(args)...)) {
        return gemm(tile_, std::forward<Args>(args)...);
      }

      /*
       * Other Maths
       */
      template <typename... Args>
      auto neg_(Args &&... args) const
      -> decltype(neg(tile_, std::forward<Args>(args)...)) {
        return neg(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      T &neg_to_(Args &&... args) {
        neg_to(tile_, std::forward<Args>(args)...);
        return tile_;
      }

      template <typename... Args>
      auto scale_(Args &&... args) const
      -> decltype(scale(tile_, std::forward<Args>(args)...)) {
        return scale(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      T &scale_to_(Args &&... args) {
        scale_to(tile_, std::forward<Args>(args)...);
        return tile_;
      }

      template <typename... Args>
      auto sum_(Args... args) const
      -> decltype(sum(tile_, std::forward<Args>(args)...)) {
        return sum(tile_, std::forward<Args>(args)...);
      }

      // TODO no external for Tensor
      // template <typename... Args>
      // auto product_(Args... args) const
      //     -> decltype(product(tile_, std::forward<Args>(args)...)) {
      //     return product(tile_, std::forward<Args>(args)...);
      // }

      template <typename... Args>
      auto min_(Args... args) const
      -> decltype(min(tile_, std::forward<Args>(args)...)) {
        return min(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      auto abs_min_(Args... args) const
      -> decltype(abs_min(tile_, std::forward<Args>(args)...)) {
        return abs_min(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      auto max_(Args... args) const
      -> decltype(max(tile_, std::forward<Args>(args)...)) {
        return max(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      auto abs_max_(Args... args) const
      -> decltype(abs_max(tile_, std::forward<Args>(args)...)) {
        return abs_max(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      auto dot_(Args... args) const
      -> decltype(dot(tile_, std::forward<Args>(args)...)) {
        return dot(tile_, std::forward<Args>(args)...);
      }

      template <typename... Args>
      auto trace_(Args... args) const
      -> decltype(trace(tile_, std::forward<Args>(args)...)) {
        return trace(tile_, std::forward<Args>(args)...);
      }
    };

  } // namespace detail

  /// An N-dimensional shallow copy wrapper

  /// \tparam T the value type of the tensor
  // Clearly there are specific requirements on T, copyable, moveable, ect ...,
  // but I am not yet ready to exactly specify them.
  template <typename T>
  class Tile {
    Range range_;
    std::shared_ptr<detail::TileImpl<T>> tile_;

  public:
    using value_type = T;
    using range_type = Range;
    using numeric_type = typename T::numeric_type;
    using size_type = std::size_t;

    Tile() = default;
    ~Tile() = default;
    Tile(Tile const &) = default;
    Tile(Tile &&) = default;
    Tile &operator=(Tile &&) = default;
    Tile &operator=(Tile const &) = default;

    Tile(Range r) : range_{std::move(r)} {}
    Tile(Range r, T t)
    : range_{std::move(r)},
      tile_{std::make_shared<detail::TileImpl<T>>(
          detail::TileImpl<T>{std::move(t)})} {}

          // Problematic constructor for truly generic code, not obvious that T
          // provides the proper constructor this will ultimately need to be part
          // of the specific requirements on T.

          // template <typename Value>
          // Tile(TA::Range r, Value v)
          //         : range_{std::move(r)},
          //           tile_{std::make_shared<detail::TileImpl<T>>(
          //                 detail::TileImpl<T>{detail::TileImpl<T>{T{r, v}}})} {}

  public:
          T &tile() { return tile_->tile(); }
          T const &tile() const { return tile_->tile(); }

          Tile clone() const { return Tile{range_, tile_->clone_()}; }

          Range const &range() const { return range_; }

          bool empty() const { return (!tile_ || tile_->empty_()); }

          Tile permute(Permutation const &p) const {
            return Tile{p ^ range_, tile_->permute_(p)};
          }

          // TODO this may not be the most robust way to handle serialization.
          template <typename Archive>
          typename madness::enable_if<
          madness::archive::is_output_archive<Archive>>::type
          serialize(Archive &ar) {
            ar &range_;
            bool empty = !static_cast<bool>(tile_);
            ar &empty;
            if (!empty) {
              ar & tile_->tile();
            }
          }

          template <typename Archive>
          typename madness::enable_if<
          madness::archive::is_input_archive<Archive>>::type
          serialize(Archive &ar) {
            ar &range_;
            bool empty = false;
            ar &empty;
            if (!empty) {
              T tile;
              ar &tile;
              tile_ = std::make_shared<detail::TileImpl<T>>(
                  detail::TileImpl<T>{std::move(tile)});
            }
          }

          /*
           * Add functions
           */
          template <typename U>
          auto add(Tile<U> const &other) const
          -> Tile<decltype(tile_->add_(other.tile()))> {
            auto out = tile_->add_(other.tile());
            return Tile<decltype(out)>(range_, std::move(out));
          }

          // Allows for adding of types which are not wrapped by Tile.
          template <typename Other>
          auto add(Other const &other) const -> Tile<decltype(tile_->add_(other))> {
            auto out = tile_->add_(other);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          template <typename U>
          auto add(Tile<U> const &other, const numeric_type factor) const
          -> Tile<decltype(tile_->add_(other.tile(), factor))> {
            auto out = tile_->add_(other.tile(), factor);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          // Allows for adding of types which are not wrapped by Tile.
          template <typename Other>
          auto add(Other const &other, const numeric_type factor) const
          -> Tile<decltype(tile_->add_(other, factor))> {
            auto out = tile_->add_(other, factor);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          template <typename U>
          auto add(Tile<U> const &other, const numeric_type factor,
              Permutation const &perm) const
          -> Tile<decltype(tile_->add_(other.tile(), factor, perm))> {
            auto new_range = perm ^ range_;
            auto out = tile_->add_(other.tile(), factor, perm);
            return Tile<decltype(out)>(std::move(new_range), std::move(out));
          }

          // Allows for adding of types which are not wrapped by Tile.
          template <typename Other>
          auto add(Other const &other, const numeric_type factor,
              Permutation const &perm) const
          -> Tile<decltype(tile_->add_(other, factor, perm))> {
            auto new_range = perm ^ range_;
            auto out = tile_->add_(other, factor, perm);
            return Tile<decltype(out)>(std::move(new_range), std::move(out));
          }

          // TODO No version of this function in Tensor external interface
          // auto add(const numeric_type factor) const
          //     -> Tile<decltype(tile_->add_(factor))> {
          //     auto out = tile_->add_(factor);
          //     return Tile<decltype(out)>(range_, std::move(out));
          // }

          // auto add(const numeric_type factor, Permutation const &perm) const
          //     -> Tile<decltype(tile_->add_(factor, perm))> {
          //     auto new_range = perm ^ range_;
          //     auto out = tile_->add_(factor, perm);
          //     return Tile<decltype(out)>(std::move(new_range), std::move(out));
          // }

          /*
           * Add_to: add_to doesn't allow permutation or range changes so write as
           * variadic.
           */
          template <typename... Args>
          Tile &add_to(Args &&... args) {
            tile_->add_to_(std::forward<Args>(args)...);
            return *this;
          }

          template <typename U, typename... Args>
          Tile &add_to(Tile<U> const &u, Args &&... args) {
            tile_->add_to_(u.tile(), std::forward<Args>(args)...);
            return *this;
          }

          /*
           * Subt functions
           */
          template <typename U>
          auto subt(Tile<U> const &other) const
          -> Tile<decltype(tile_->subt_(other.tile()))> {
            auto out = tile_->subt_(other.tile());
            return Tile<decltype(out)>(range_, std::move(out));
          }

          // Allows for subting of types which are not wrapped by Tile.
          template <typename Other>
          auto subt(Other const &other) const -> Tile<decltype(tile_->subt_(other))> {
            auto out = tile_->subt_(other);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          template <typename U>
          auto subt(Tile<U> const &other, const numeric_type factor) const
          -> Tile<decltype(tile_->subt_(other.tile(), factor))> {
            auto out = tile_->subt_(other.tile(), factor);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          // Allows for subting of types which are not wrapped by Tile.
          template <typename Other>
          auto subt(Other const &other, const numeric_type factor) const
          -> Tile<decltype(tile_->subt_(other, factor))> {
            auto out = tile_->subt_(other, factor);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          template <typename U>
          auto subt(Tile<U> const &other, const numeric_type factor,
              Permutation const &perm) const
          -> Tile<decltype(tile_->subt_(other.tile(), factor, perm))> {
            auto new_range = perm ^ range_;
            auto out = tile_->subt_(other.tile(), factor, perm);
            return Tile<decltype(out)>(std::move(new_range), std::move(out));
          }

          // Allows for subting of types which are not wrapped by Tile.
          template<typename Other>
          auto subt(Other const &other, const numeric_type factor,
              Permutation const &perm) const
          -> Tile<decltype(tile_->subt_(other, factor, perm))> {
            auto new_range = perm ^ range_;
            auto out = tile_->subt_(other, factor, perm);
            return Tile<decltype(out)>(std::move(new_range), std::move(out));
          }

          // TODO No version of this function in Tensor external interface
          // auto subt(const numeric_type factor) const
          //     -> Tile<decltype(tile_->subt_(factor))> {
          //     auto out = tile_->subt_(factor);
          //     return Tile<decltype(out)>(range_, std::move(out));
          // }

          // auto subt(const numeric_type factor, Permutation const &perm) const
          //     -> Tile<decltype(tile_->subt_(factor, perm))> {
          //     auto new_range = perm ^ range_;
          //     auto out = tile_->subt_(factor, perm);
          //     return Tile<decltype(out)>(std::move(new_range), std::move(out));
          // }

          /*
           * subt_to: subt_to doesn't allow permutation or range changes so write as
           * variadic.
           */
          template <typename... Args>
          Tile &subt_to(Args &&... args) {
            tile_->subt_to_(std::forward<Args>(args)...);
            return *this;
          }

          template <typename U, typename... Args>
          Tile &subt_to(Tile<U> const &u, Args &&... args) {
            tile_->subt_to_(u.tile(), std::forward<Args>(args)...);
            return *this;
          }

          /*
           * mult
           */
          template <typename U>
          auto mult(Tile<U> const &other) const
          -> Tile<decltype(tile_->mult_(other.tile()))> {
            auto out = tile_->mult_(other.tile());
            return Tile<decltype(out)>(range_, std::move(out));
          }

          // Allows for multing of types which are not wrapped by Tile.
          template <typename Other>
          auto mult(Other const &other) const -> Tile<decltype(tile_->mult_(other))> {
            auto out = tile_->mult_(other);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          template <typename U>
          auto mult(Tile<U> const &other, const numeric_type factor) const
          -> Tile<decltype(tile_->mult_(other.tile(), factor))> {
            auto out = tile_->mult_(other.tile(), factor);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          // Allows for multing of types which are not wrapped by Tile.
          template <typename Other>
          auto mult(Other const &other, const numeric_type factor) const
          -> Tile<decltype(tile_->mult_(other, factor))> {
            auto out = tile_->mult_(other, factor);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          template <typename U>
          auto mult(Tile<U> const &other, const numeric_type factor,
              Permutation const &perm) const
          -> Tile<decltype(tile_->mult_(other.tile(), factor, perm))> {
            auto new_range = perm ^ range_;
            auto out = tile_->mult_(other.tile(), factor, perm);
            return Tile<decltype(out)>(std::move(new_range), std::move(out));
          }

          // Allows for multing of types which are not wrapped by Tile.
          template<typename Other>
          auto mult(Other const &other, const numeric_type factor,
              Permutation const &perm) const
          -> Tile<decltype(tile_->mult_(other, factor, perm))> {
            auto new_range = perm ^ range_;
            auto out = tile_->mult_(other, factor, perm);
            return Tile<decltype(out)>(std::move(new_range), std::move(out));
          }

          auto mult(const numeric_type factor) const
          -> Tile<decltype(tile_->mult_(factor))> {
            auto out = tile_->mult_(factor);
            return Tile<decltype(out)>(range_, std::move(out));
          }

          auto mult(const numeric_type factor, Permutation const &perm) const
          -> Tile<decltype(tile_->mult_(factor, perm))> {
            auto new_range = perm ^ range_;
            auto out = tile_->mult_(factor, perm);
            return Tile<decltype(out)>(std::move(new_range), std::move(out));
          }

          /*
           * mult_to
           */
          template <typename... Args>
          Tile &mult_to(Args &&... args) {
            tile_->mult_to_(std::forward<Args>(args)...);
            return *this;
          }

          template <typename U, typename... Args>
          Tile &mult_to(Tile<U> const &u, Args &&... args) {
            tile_->mult_to_(u.tile(), std::forward<Args>(args)...);
            return *this;
          }

          /*
           * Neg
           */
          Tile neg() const { return Tile(range_, tile_->neg_()); }

          Tile neg(Permutation const &perm) const {
            return Tile(perm ^ range_, tile_->neg_(perm));
          }

          Tile &neg_to() {
            tile_->neg_to_();
            return *this;
          }

          /*
           * Scale
           */
          Tile scale(const numeric_type factor) const {
            return Tile(range_, tile_->scale_(factor));
          }

          Tile scale(const numeric_type factor, Permutation const &perm) const {
            return Tile(perm ^ range_, tile_->scale_(factor, perm));
          }

          Tile &scale_to(const numeric_type factor) {
            tile_->scale_to_(factor);
            return *this;
          }

          /*
           * Gemm
           */
          template <typename Other>
          auto gemm(Other const &o, const numeric_type factor,
              math::GemmHelper const &gh) const
          -> Tile<decltype(tile_->gemm_(o, factor, gh))> {
            TA_ASSERT(!empty());
            TA_ASSERT(range_.dim() == gh.left_rank());

            // Check that the arguments are not empty and have the correct ranks
            TA_ASSERT(!o.empty());
            TA_ASSERT(o.range().dim() == gh.right_rank());

            // Check that the inner dimensions of left and right match
            TA_ASSERT(gh.left_right_coformal(range_.start(), o.range().start()));
            TA_ASSERT(gh.left_right_coformal(range_.finish(), o.range().finish()));
            TA_ASSERT(gh.left_right_coformal(range_.size(), o.range().size()));

            auto range = gh.make_result_range<Range>(range_, o.range());

            auto out = tile_->gemm_(o, factor, gh);
            return Tile<decltype(out)>(std::move(range), std::move(out));
          }

          template <typename U>
          auto gemm(Tile<U> const &u, const numeric_type factor,
              math::GemmHelper const &gh) const
          -> Tile<decltype(tile_->gemm_(u.tile(), factor, gh))> {
            TA_ASSERT(!empty());
            TA_ASSERT(range_.dim() == gh.left_rank());

            // Check that the arguments are not empty and have the correct ranks
            TA_ASSERT(!u.empty());
            TA_ASSERT(u.range().dim() == gh.right_rank());

            // Check that the inner dimensions of left and right match
            TA_ASSERT(gh.left_right_coformal(range_.start(), u.range().start()));
            TA_ASSERT(gh.left_right_coformal(range_.finish(), u.range().finish()));
            TA_ASSERT(gh.left_right_coformal(range_.size(), u.range().size()));

            auto range = gh.make_result_range<Range>(range_, u.range());
            auto out = tile_->gemm_(u.tile(), factor, gh);
            return Tile<decltype(out)>(std::move(range), std::move(out));
          }

          template <typename Left, typename Right>
          Tile &gemm(Left const &l, Right const &r, const numeric_type factor,
              math::GemmHelper const &gh) {
            TA_ASSERT(!empty());
            TA_ASSERT(range_.dim() == gh.result_rank());

            // Check that the arguments are not empty and have the correct ranks
            TA_ASSERT(!l.empty());
            TA_ASSERT(l.range().dim() == gh.left_rank());
            TA_ASSERT(!r.empty());
            TA_ASSERT(r.range().dim() == gh.right_rank());

            // Check that the outer dimensions of left match the the corresponding
            // dimensions in result
            TA_ASSERT(gh.left_result_coformal(l.range().start(), range_.start()));
            TA_ASSERT(gh.left_result_coformal(l.range().finish(), range_.finish()));
            TA_ASSERT(gh.left_result_coformal(l.range().size(), range_.size()));

            // Check that the outer dimensions of right match the the corresponding
            // dimensions in result
            TA_ASSERT(gh.right_result_coformal(r.range().start(), range_.start()));
            TA_ASSERT(
                gh.right_result_coformal(r.range().finish(), range_.finish()));
            TA_ASSERT(gh.right_result_coformal(r.range().size(), range_.size()));

            // Check that the inner dimensions of left and right match
            TA_ASSERT(gh.left_right_coformal(l.range().start(), r.range().start()));
            TA_ASSERT(
                gh.left_right_coformal(l.range().finish(), r.range().finish()));
            TA_ASSERT(gh.left_right_coformal(l.range().size(), r.range().size()));

            tile_->gemm_(l, r, factor, gh);
            return *this;
          }

          template <typename U, typename V>
          Tile &gemm(Tile<U> const &l, Tile<V> const &r, const numeric_type factor,
              math::GemmHelper const &gh) {
            TA_ASSERT(!empty());
            TA_ASSERT(range_.dim() == gh.result_rank());

            // Check that the arguments are not empty and have the correct ranks
            TA_ASSERT(!l.empty());
            TA_ASSERT(l.range().dim() == gh.left_rank());
            TA_ASSERT(!r.empty());
            TA_ASSERT(r.range().dim() == gh.right_rank());

            // Check that the outer dimensions of left match the the corresponding
            // dimensions in result
            TA_ASSERT(gh.left_result_coformal(l.range().start(), range_.start()));
            TA_ASSERT(gh.left_result_coformal(l.range().finish(), range_.finish()));
            TA_ASSERT(gh.left_result_coformal(l.range().size(), range_.size()));

            // Check that the outer dimensions of right match the the corresponding
            // dimensions in result
            TA_ASSERT(gh.right_result_coformal(r.range().start(), range_.start()));
            TA_ASSERT(
                gh.right_result_coformal(r.range().finish(), range_.finish()));
            TA_ASSERT(gh.right_result_coformal(r.range().size(), range_.size()));

            // Check that the inner dimensions of left and right match
            TA_ASSERT(gh.left_right_coformal(l.range().start(), r.range().start()));
            TA_ASSERT(
                gh.left_right_coformal(l.range().finish(), r.range().finish()));
            TA_ASSERT(gh.left_right_coformal(l.range().size(), r.range().size()));

            tile_->gemm_(l.tile(), r.tile(), factor, gh);
            return *this;
          }

          /*
           * Reduction type functions
           */
          auto trace() const -> decltype(tile_->trace_()) { return tile_->trace_(); }

          auto sum() const -> decltype(tile_->sum_()) { return tile_->sum_(); }

          // TODO no external for Tensor
          // auto product() const -> decltype(tile_->product_()) {
          //     return tile_->product_();
          // }

          auto min() const -> decltype(tile_->min_()) { return tile_->min_(); }

          auto max() const -> decltype(tile_->max_()) { return tile_->max_(); }

          auto abs_max() const -> decltype(tile_->abs_max_()) {
            return tile_->abs_max_();
          }

          auto abs_min() const -> decltype(tile_->abs_min_()) {
            return tile_->abs_min_();
          }

          auto norm() const -> decltype(tile_->norm_()) { return tile_->norm_(); }

          auto squared_norm() const -> decltype(tile_->squared_norm_()) {
            return tile_->squared_norm_();
          }

          template <typename Other>
          auto dot(Other const &o) const -> decltype(tile_->dot_(o)) {
            return tile_->dot_(o);
          }
  };

  template <typename Left, typename Right>
  auto operator+(Tile<Left> const &l, Tile<Right> const &r)
  -> decltype(l.add(r)) {
    return l.add(r);
  }

  template <typename Left, typename Right>
  auto operator-(Tile<Left> const &l, Tile<Right> const &r)
  -> decltype(l.subt(r)) {
    return l.subt(r);
  }

  template <typename Left, typename Right>
  auto operator*(Tile<Left> const &l, Tile<Right> const &r)
  -> decltype(l.mult(r)) {
    return l.mult(r);
  }

  template <typename N, typename U>
  typename std::enable_if<TiledArray::detail::is_numeric<N>::value, Tile<U>>::type
  operator*(Tile<U> const &left, N right) {
    return left.scale(right);
  }

  template <typename N, typename U>
  typename std::enable_if<TiledArray::detail::is_numeric<N>::value, Tile<U>>::type
  operator*(N left, Tile<U> const &right) {
    return right.scale(left);
  }

  template <typename U>
  Tile<U> operator-(Tile<U> const &u) {
    return u.neg();
  }

  template <typename U>
  Tile<U> operator^(Permutation const &perm, Tile<U> const tile) {
    return tile.permute(perm);
  }

  template <typename U>
  inline std::ostream &operator<<(std::ostream &os, Tile<U> const &tile) {
    os << tile.range() << ": " << tile.tile() << "\n";
    return os;
  }

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_H__INCLUDED
