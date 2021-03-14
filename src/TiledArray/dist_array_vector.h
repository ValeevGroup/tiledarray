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

#ifndef TILEDARRAY_DIST_ARRAY_VECTOR_H__INCLUDED
#define TILEDARRAY_DIST_ARRAY_VECTOR_H__INCLUDED

#include <TiledArray/dist_array.h>
#include <TiledArray/error.h>
#include <TiledArray/expressions/expr.h>
#include <TiledArray/expressions/tsr_expr.h>
#include <TiledArray/math/linalg/basic.h>

#include <cmath>
#include <utility>
#include <vector>

namespace TiledArray {

namespace detail {

/// A vector of objects that, if there is only one object, can be implicitly
/// converted to it

/// @tparam T the type of contained objects; must be copy constructable and
/// move constructable, does not have to be copy or move assignable
template <typename T>
class VectorBase : public container::svector<T> {
 public:
  using base_type = container::svector<T>;

  VectorBase() = default;
  /// copy ctor delegates to ctor that uses emplace_back
  VectorBase(const VectorBase &other)
      : VectorBase(static_cast<const base_type &>(other)) {}
  /// move ctor delegates to ctor that uses emplace_back
  VectorBase(VectorBase &&other)
      : VectorBase(static_cast<base_type &&>(other)) {}

  /// constructs a vector of default-constructed objects
  /// @param[in] sz the target size of the vector
  explicit VectorBase(std::size_t sz) : base_type(sz) {}

  /// copies objects from a std::vector
  /// @param v a std::vector from which the objects will be copied
  explicit VectorBase(const std::vector<T> &v) {
    this->reserve(v.size());
    for (auto &&e : v) {
      this->emplace_back(e);
    }
  }

  /// moves objects from a std::vector
  /// @param v a std::vector from which the objects will be moved
  explicit VectorBase(std::vector<T> &&v) {
    this->reserve(v.size());
    for (auto &&e : v) {
      this->emplace_back(std::move(e));
    }
  }

  /// copies objects from a container::svector
  /// @param v a container::svector from which the objects will be copied
  template <std::size_t Size>
  explicit VectorBase(const container::svector<T, Size> &v) {
    this->reserve(v.size());
    for (auto &&e : v) {
      this->emplace_back(e);
    }
  }

  /// moves objects from a container::svector
  /// @param v a container::svector from which the objects will be moved
  template <std::size_t Size>
  explicit VectorBase(container::svector<T, Size> &&v) {
    this->reserve(v.size());
    for (auto &&e : v) {
      this->emplace_back(std::move(e));
    }
  }

  template <typename B, typename E>
  VectorBase(B &&begin, E &&end)
      : container::svector<T>{std::forward<B>(begin), std::forward<E>(end)} {}

  /// copy assignment
  VectorBase &operator=(const VectorBase &other) {
    this->clear();
    if (this->size() < other.size()) this->reserve(other.size());
    for (auto &&e : other) {
      this->emplace_back(e);
    }
    return *this;
  }

  /// move assignment
  VectorBase &operator=(VectorBase &&other) {
    this->clear();
    if (this->size() < other.size()) this->reserve(other.size());
    for (auto &&e : other) {
      this->emplace_back(std::move(e));
    }
    return *this;
  }

  /// converts to a nonconst lvalue ref to the first object is there is only one
  /// @throw TiledArray::Exception if the number of objects is not equal to 1
  operator T &() & {
    if (this->size() == 1)
      return this->operator[](0);
    else
      throw TiledArray::Exception(
          "casting Vector<T> to T only possible if it contains 1 "
          "T");
  }

  /// converts to a const lvalue ref to the first object is there is only one
  /// @throw TiledArray::Exception if the number of objects is not equal to 1
  operator const T &() const & {
    if (this->size() == 1)
      return this->operator[](0);
    else
      throw TiledArray::Exception(
          "casting Vector<T> to T only possible if it contains 1 "
          "T");
  }

  /// converts to a nonconst rvalue ref to the first object is there is only one
  /// @throw TiledArray::Exception if the number of objects is not equal to 1
  operator T &&() && {
    if (this->size() == 1)
      return std::move(this->operator[](0));
    else
      throw TiledArray::Exception(
          "casting Vector<T> to T only possible if it contains 1 "
          "T");
  }

  /// @return true if nonempty
  explicit operator bool() const { return !this->empty(); }
};  // VectorBase<T>

}  // namespace detail

/// a vector of Shape objects
template <typename Shape_>
class ShapeVector : public detail::VectorBase<Shape_> {
 public:
  using base_type = detail::VectorBase<Shape_>;
  using T = typename base_type::value_type;
  using value_type = typename Shape_::value_type;

  ShapeVector() = default;

  ShapeVector(std::size_t i) : base_type(i) {}

  template <typename... Args,
            typename = std::enable_if_t<
                (std::is_same_v<std::remove_reference_t<Args>, T> && ...)>>
  ShapeVector(Args &&..._t) : base_type{std::forward<Args>(_t)...} {}

  ShapeVector(const std::vector<T> &v) : base_type(v) {}

  ShapeVector(std::vector<T> &&v) : base_type(std::move(v)) {}

  template <std::size_t Size>
  ShapeVector(const container::svector<T, Size> &v) : base_type(v) {}

  template <std::size_t Size>
  ShapeVector(container::svector<T, Size> &&v) : base_type(std::move(v)) {}

  template <typename B, typename E>
  ShapeVector(B &&begin, E &&end)
      : base_type(std::forward<B>(begin), std::forward<E>(end)) {}

  template <typename S>
  ShapeVector<Shape_> &operator=(const S &shape) {
    static_cast<Shape_ &>(*this) = shape;
    return *this;
  }

  using base_type::operator T &;
  using base_type::operator const T &;
  using base_type::operator T &&;

  /// @return true if nonempty and each element is nonnull (i.e., is
  /// initialized)
  using base_type::operator bool;

  template <typename Op>
  ShapeVector transform(const Op &op) {
    ShapeVector result;
    result.reserve(this->size());
    for (auto &&v : *this) result.emplace_back(v.transform(op));
    return result;
  }

  /// @return minimum value of threshold of elements
  value_type threshold() {
    value_type thresh = (*this)[0].threshold();
    for (auto &&v : *this) {
      value_type thresh_val = v.threshold();
      if (thresh_val < thresh) thresh = thresh_val;
    }
    return thresh;
  }
};

namespace expressions {

template <typename E>
using enable_if_expression = std::void_t<typename ExprTrait<E>::engine_type>;

template <typename E>
struct is_tsr_expression : public std::false_type {};
template <typename Array, bool Alias>
struct is_tsr_expression<TsrExpr<Array, Alias>> : public std::true_type {};
template <typename E>
constexpr const bool is_tsr_expression_v = is_tsr_expression<E>::value;

/// a vector of Expr objects
template <typename Expr_, typename = enable_if_expression<Expr_>>
class ExprVector : public TiledArray::detail::VectorBase<Expr_> {
 public:
  using base_type = TiledArray::detail::VectorBase<Expr_>;
  using T = typename base_type::value_type;

  ExprVector() = default;
  ExprVector(const ExprVector &) = default;
  ExprVector(ExprVector &&) = default;

  ExprVector(std::size_t i) : base_type(i) {}

  template <typename... Args,
            typename = std::enable_if_t<
                (std::is_same_v<std::remove_reference_t<Args>, T> && ...)>>
  ExprVector(Args &&..._t) : base_type{std::forward<Args>(_t)...} {}

  ExprVector(const std::vector<T> &v) : base_type(v) {}

  ExprVector(std::vector<T> &&v) : base_type(std::move(v)) {}

  template <std::size_t Size>
  ExprVector(const container::svector<T, Size> &v) : base_type(v) {}

  template <std::size_t Size>
  ExprVector(container::svector<T, Size> &&v) : base_type(std::move(v)) {}

  template <typename B, typename E>
  ExprVector(B &&begin, E &&end)
      : base_type(std::forward<B>(begin), std::forward<E>(end)) {}

  // Apply unary operator- to each member of ExprVector
  ExprVector<Expr_> operator-() const {
    auto expr_sz = (*this).size();
    ExprVector<Expr_> result;
    for (auto i = 0; i < expr_sz; ++i) {
      result[i] = -(*this)[i];
    }
    return result;
  }

  template <typename E>
  ExprVector &operator=(const ExprVector<E> &expr) {
    const auto expr_sz = expr.size();
    if (this->size() ==
        0) {  // if null, and this is TsrExpr, resize the underlying
              // DistArrayVector to match the size of expr and rebuild this
      if constexpr (is_tsr_expression_v<Expr_>) {
        TA_ASSERT(arrayvec_ptr_);
        TA_ASSERT(!arrayvec_annotation_.empty());
        arrayvec_ptr_->resize(expr_sz);
        static_cast<base_type &>(*this) =
            static_cast<base_type &&>((*arrayvec_ptr_)(arrayvec_annotation_));
      }
    } else {  // if this is nonull must match the size of expr
      TA_ASSERT(this->size() == expr_sz);
    }
    for (size_t i = 0; i != expr_sz; ++i) {
      (*this)[i] = expr[i];
    }
    return *this;
  }

  template <typename E>
  ExprVector &operator=(ExprVector<E> &&expr) {
    return this->operator=(static_cast<const ExprVector<E> &>(expr));
  }

  ExprVector &operator=(ExprVector &&expr) {
    return this->operator=<Expr_>(std::move(expr));
  }
  ExprVector &operator=(const ExprVector &expr) {
    return this->operator=<Expr_>(expr);
  }

  template <typename E,
            typename = enable_if_expression<std::remove_reference_t<E>>>
  ExprVector &operator=(E &&expr) {
    container::svector<std::remove_reference_t<E>> vec;
    vec.reserve(1);
    vec.emplace_back(std::forward<E>(expr));
    return this->operator=(
        ExprVector<std::remove_reference_t<E>>(std::move(vec)));
  }

  using base_type::operator T &;
  using base_type::operator const T &;
  using base_type::operator T &&;

  /// @return true if nonempty and each element is nonnull (i.e., is
  /// initialized)
  using base_type::operator bool;

#define TA_EXPRVEC_UNARYREDUCTION_DEF(op)                      \
  auto op(World &world) const {                                \
    container::svector<decltype((*this)[0].op(world))> result; \
    result.reserve(this->size());                              \
    for (auto &&v : (*this)) result.emplace_back(v.op(world)); \
    return result;                                             \
  }                                                            \
  auto op() const { return this->op(this->default_world()); }

  TA_EXPRVEC_UNARYREDUCTION_DEF(norm)

#undef TA_EXPRVEC_UNARYREDUCTION_DEF

#define TA_EXPRVEC_BINARYREDUCTION_DEF(op)                                    \
  template <typename E, typename = enable_if_expression<E>>                   \
  auto op(const E &right_expr, World &world) const {                          \
    container::svector<decltype((*this)[0].op(right_expr, world))> result;    \
    result.reserve(this->size());                                             \
    for (auto &&v : (*this)) result.emplace_back(v.op(right_expr, world));    \
    return result;                                                            \
  }                                                                           \
  template <typename E, typename = enable_if_expression<E>>                   \
  auto op(const E &right_expr) const {                                        \
    return this->op(right_expr, this->default_world());                       \
  }                                                                           \
  template <typename E, typename = enable_if_expression<E>>                   \
  auto op(const ExprVector<E> &right_expr, World &world) const {              \
    container::svector<decltype((*this)[0].op(right_expr[0], world))> result; \
    const auto sz = this->size();                                             \
    TA_ASSERT(sz == right_expr.size());                                       \
    result.reserve(sz);                                                       \
    for (size_t i = 0; i != sz; ++i)                                          \
      result.emplace_back((*this)[i].op(right_expr[i], world));               \
    return result;                                                            \
  }                                                                           \
  template <typename E, typename = enable_if_expression<E>>                   \
  auto op(const ExprVector<E> &right_expr) const {                            \
    return this->op(right_expr, this->default_world());                       \
  }

  TA_EXPRVEC_BINARYREDUCTION_DEF(dot)

#undef TA_EXPRVEC_BINARYREDUCTION_DEF

  World &default_world() const {
    if (*this) {
      if constexpr (has_array<Expr_>::value)
        return (*this)[0].array().world();
      else
        return TiledArray::get_default_world();
    } else
      throw TiledArray::Exception(
          "ExprVector::default_world() called on null object");
  }

  /// calls \c set_shape(shape) on each member
  /// \param[in] shape the shape object to use for each member of this
  /// \return reference to this
  ExprVector &set_shape(typename EngineParamOverride<
                        typename Expr_::engine_type>::shape_type const &shape) {
    for (auto &&v : (*this)) {
      v.set_shape(shape);
    }
    return *this;
  }

  /// calls \c set_shape(shape[i]) on member \c i
  /// \param[in] shape the vector of shapes
  /// \return reference to this
  ExprVector &set_shape(
      ShapeVector<typename EngineParamOverride<
          typename Expr_::engine_type>::shape_type> const &shapes) {
    auto sz = this->size();
    TA_ASSERT(sz == shapes.size());
    size_t i = 0;
    for (auto &&v : (*this)) {
      v.set_shape(shapes[i]);
      ++i;
    }
    return *this;
  }

  /// Expression plus-assignment operator

  /// \tparam D The derived expression type
  /// \param other The expression that will be added to this array
  template <typename D>
  ExprVector &operator+=(const ExprVector<D> &other) {
    const auto sz = this->size();
    TA_ASSERT(sz == other.size());
    size_t i = 0;
    for (auto &&v : *this) {
      v += other[i];
      ++i;
    }
    return *this;
  }

  /// Expression conjugate applied to each element
  ExprVector &conj() {
    size_t i = 0;
    for (auto &&v : *this) {
      v.conj();
      ++i;
    }
    return *this;
  }

 private:
  // for TsrExpr type need to keep ptr to the bound DistArrayVector
  template <typename E>
  constexpr static decltype(auto) arrayvec_ptr_init() {
    if constexpr (is_tsr_expression_v<E>) {
      TiledArray::DistArrayVector<
          typename ExprTrait<E>::array_type::value_type,
          typename ExprTrait<E>::array_type::policy_type> *result = nullptr;
      return result;
    } else
      return nullptr;
  }
  decltype(arrayvec_ptr_init<Expr_>()) arrayvec_ptr_ =
      arrayvec_ptr_init<Expr_>();
  std::string arrayvec_annotation_;

  template <typename Tile, typename Policy>
  friend class TiledArray::DistArrayVector;

  void set_arrayvec_ptr(void *ptr, const std::string &annotation) {
    if constexpr (is_tsr_expression_v<Expr_>) {
      arrayvec_ptr_ = static_cast<decltype(arrayvec_ptr_init<Expr_>())>(ptr);
      arrayvec_annotation_ = annotation;
    }
  }
};

template <typename E>
auto operator*(const double &factor, const ExprVector<E> &expr) {
  const auto expr_sz = expr.size();
  ExprVector<decltype(factor * expr[0])> result;
  result.reserve(expr_sz);
  for (size_t i = 0; i != expr_sz; ++i) {
    result.emplace_back(factor * expr[i]);
  }
  return result;
}

template <typename E, typename = enable_if_expression<E>>
const Expr<E> &to_base_expr(const ExprVector<E> &e) {
  return static_cast<const Expr<E> &>(static_cast<const E &>(e));
}

// template <typename Left, typename Right>
// ExprVector<MultExpr<Left,Right>> operator*(const ExprVector<Left>& left,
//                                           const Expr<Right>& right);

#define TA_EXPRVEC_BINARYOP_DEF(op, result_type)                            \
  template <typename Left, typename Right>                                  \
  ExprVector<result_type<Left, Right>> operator op(                         \
      const ExprVector<Left> &left, const Expr<Right> &right) {             \
    const auto sz = left.size();                                            \
    ExprVector<result_type<Left, Right>> result;                            \
    result.reserve(sz);                                                     \
    for (size_t i = 0; i != sz; ++i) result.emplace_back(left[i] op right); \
    return result;                                                          \
  }                                                                         \
  template <typename Left, typename Right>                                  \
  ExprVector<result_type<Left, Right>> operator op(                         \
      const Expr<Left> &left, const ExprVector<Right> &right) {             \
    const auto sz = right.size();                                           \
    ExprVector<result_type<Left, Right>> result;                            \
    result.reserve(sz);                                                     \
    for (size_t i = 0; i != sz; ++i) result.emplace_back(left op right[i]); \
    return result;                                                          \
  }                                                                         \
  template <typename Left, typename Right>                                  \
  ExprVector<result_type<Left, Right>> operator op(                         \
      const ExprVector<Left> &left, const ExprVector<Right> &right) {       \
    if (left.size() == 1 && right.size() > 1) {                             \
      const auto sz = right.size();                                         \
      ExprVector<result_type<Left, Right>> result;                          \
      result.reserve(sz);                                                   \
      for (size_t i = 0; i != sz; ++i)                                      \
        result.emplace_back(left[0] op right[i]);                           \
      return result;                                                        \
    } else if (left.size() > 1 && right.size() == 1) {                      \
      const auto sz = left.size();                                          \
      ExprVector<result_type<Left, Right>> result;                          \
      result.reserve(sz);                                                   \
      for (size_t i = 0; i != sz; ++i)                                      \
        result.emplace_back(left[i] op right[0]);                           \
      return result;                                                        \
    } else {                                                                \
      TA_ASSERT(left.size() == right.size());                               \
      const auto sz = left.size();                                          \
      ExprVector<result_type<Left, Right>> result;                          \
      result.reserve(sz);                                                   \
      for (size_t i = 0; i != sz; ++i)                                      \
        result.emplace_back(left[i] op right[i]);                           \
      return result;                                                        \
    }                                                                       \
  }

// fwd declare expressions
template <typename, typename>
class AddExpr;
template <typename, typename>
class SubtExpr;
template <typename, typename>
class MultExpr;

TA_EXPRVEC_BINARYOP_DEF(+, AddExpr)
TA_EXPRVEC_BINARYOP_DEF(-, SubtExpr)
TA_EXPRVEC_BINARYOP_DEF(*, MultExpr)
#undef TA_EXPRVEC_BINARYOP_DEF

}  // namespace expressions

/// A vector of DistArray objects

/// Each DistArray object is distributed over the same World, i.e. the vector is
/// not distributed. If you need a vector distributed across a world, with each
/// DistArray object belonging to a local world, see
/// TiledArray/vector_of_arrays.h .
/// @internal Rationale: sometimes it is more convenient to represent a sequence
/// (vector) of arrays
///           as a vector, rather than a "fused" array with an extra dimension.
///           The latter makes sense when operations are homogeneous across the
///           extra dimension. If extents of modes will depend on the vector
///           index then fusing the arrays becomes cumbersome (N.B.: if the
///           "inner" arrays do not need to be distributed then you should use a
///           tensor-of-tensor, i.e. DistArray<Tensor<Tensor<T>>>)
/// @tparam T a DistArray type
template <typename Tile, typename Policy>
class DistArrayVector : public detail::VectorBase<DistArray<Tile, Policy>> {
 public:
  using base_type = detail::VectorBase<DistArray<Tile, Policy>>;
  using array_type = TiledArray::DistArray<Tile, Policy>;
  using eval_type = typename array_type::eval_type;
  using policy_type = Policy;
  using value_type = array_type;
  using element_type = typename array_type::element_type;
  using scalar_type = typename array_type::scalar_type;
  using T = array_type;

  DistArrayVector() = default;

  DistArrayVector(std::size_t i) : base_type(i) {}

  template <typename... Args,
            typename = std::enable_if_t<
                (std::is_same_v<std::remove_reference_t<Args>, T> && ...)>>
  DistArrayVector(Args &&..._t) : base_type(std::forward<Args>(_t)...) {}

  DistArrayVector(const std::vector<T> &t_pack) : base_type(t_pack) {}

  DistArrayVector(std::vector<T> &&t_pack) : base_type(std::move(t_pack)) {}

  template <std::size_t Size>
  DistArrayVector(const container::svector<T, Size> &v) : base_type(v) {}

  template <std::size_t Size>
  DistArrayVector(container::svector<T, Size> &&v) : base_type(std::move(v)) {}

  template <typename B, typename E>
  DistArrayVector(B &&begin, E &&end)
      : base_type(std::forward<B>(begin), std::forward<E>(end)) {}

  using base_type::operator T &;
  using base_type::operator const T &;
  using base_type::operator T &&;

  /// @return true is nonempty and each element is nonnull (i.e., is
  /// initialized)
  explicit operator bool() const {
    bool result = false;
    if (this->size()) {
      result = true;
      for (auto &&v : (*this)) {
        result = result && static_cast<bool>(v);
      }
    }
    return result;
  }

  /// @return a World to which all DistArrays belong
  /// @throw TiledArray::Exception if this is null
  World &world() const {
    if (!(*this))
      throw TiledArray::Exception(
          "DistArrayVector::world() called on null object");
    auto &w = (*this)[0].world();
#ifndef NDEBUG
    for (size_t v = 1; v < this->size(); ++v)
      TA_ASSERT(w.id() == (*this)[v].world().id());
#endif
    return w;
  }

  /// @return a TiledRange object describing all objects
  /// @throw TiledArray::Exception if this is null
  const TiledRange &trange() const {
    if (!(*this))
      throw TiledArray::Exception(
          "DistArrayVector::trange() called on null object");
    auto &tr = (*this)[0].trange();
#ifndef NDEBUG
    for (size_t v = 1; v < this->size(); ++v)
      TA_ASSERT(tr == (*this)[v].trange());
#endif
    return tr;
  }

  /// Calls truncate() on each element
  /// @throw TiledArray::Exception if this is null
  void truncate() {
    if (!(*this))
      throw TiledArray::Exception(
          "DistArrayVector::truncate() called on null object");
    for (size_t v = 1; v < this->size(); ++v) (*this)[v].truncate();
  }

  /// Calls is_initalized() on each element and return true if all
  /// @return a bool true if all elemments are initialized
  /// @throw TiledArray::Exception if this is null
  bool is_initialized() {
    if (!(*this))
      throw TiledArray::Exception(
          "DistArrayVector::is_initalized() called on null object");
    for (auto &&v : *this) {
      if (!v.is_initalized()) return false;
    };
    return true;
  }

  /// returns the shapes of the member arrays
  /// @return vector of shapes
  /// @throw TiledArray::Exception if this is null
  ShapeVector<typename array_type::shape_type> shape() const {
    if (!(*this))
      throw TiledArray::Exception(
          "DistArrayVector::shape() called on null object");
    ShapeVector<typename array_type::shape_type> result;
    result.reserve(this->size());
    for (auto &&v : *this) result.emplace_back(v.shape());
    return result;
  }

  /// Create a tensor expression

  /// \param annotation An annotation string
  /// \return A const tensor expression object
  TiledArray::expressions::ExprVector<
      TiledArray::expressions::TsrExpr<const T, true>>
  operator()(const std::string &annotation) const {
    TA_ASSERT(*this);
    TiledArray::expressions::ExprVector<
        TiledArray::expressions::TsrExpr<const T, true>>
        result;
    for (auto &&elem : (*this)) {
      result.emplace_back(elem(annotation));
    }
    return result;
  }

  /// Create a tensor expression

  /// \param annotation An annotation string
  /// \return A non-const tensor expression object
  TiledArray::expressions::ExprVector<TiledArray::expressions::TsrExpr<T, true>>
  operator()(const std::string &annotation) {
    TiledArray::expressions::ExprVector<
        TiledArray::expressions::TsrExpr<T, true>>
        result;
    if (!*this) result.set_arrayvec_ptr(this, annotation);
    for (auto &&elem : (*this)) {
      result.emplace_back(elem(annotation));
    }
    return result;
  }

  /// \return A DistArray that is the sum of elements in DistArrayVector
  DistArray<Tile, Policy> vector_sum() const {
    DistArray<Tile, Policy> result = (*this)[0];
    if ((*this).size() > 1) {
      for (auto i = 1; i < (*this).size(); ++i) {
        result("i,j") += (*this)[i]("i,j");
      }
    }
    return result;
  }

};  // DistArrayVector<T>

template <typename Tile, typename Policy>
auto rank(const DistArrayVector<Tile, Policy> &a) {
  MPQC_ASSERT(a.size() > 0);
  size_t result = rank(a[0]);
  for (size_t v = 1; v < a.size(); ++v) MPQC_ASSERT(result == rank(a[v]));
  return result;
}

template <typename Tile, typename Policy>
size_t volume(const DistArrayVector<Tile, Policy> &a) {
  MPQC_ASSERT(a.size() > 0);
  size_t result = volume(a[0]);
  for (size_t v = 1; v < a.size(); ++v) MPQC_ASSERT(result == volume(a[v]));
  return result;
}

template <typename Tile, typename Policy, typename Op,
          typename = typename std::enable_if<!TiledArray::detail::is_array<
              typename std::decay<Op>::type>::value>::type>
inline auto foreach_inplace(DistArrayVector<Tile, Policy> &arg, Op &&op,
                            bool fence = true) {
  const auto array_sz = arg.size();
  for (size_t i = 0; i != array_sz; ++i) {
    foreach_inplace(arg[i], op);
  }
}

template <typename Tile, typename Policy>
inline auto norm2(const DistArrayVector<Tile, Policy> &a) -> decltype(
    norm2(std::declval<typename DistArrayVector<Tile, Policy>::array_type>())) {
  using scalar_type = typename DistArrayVector<Tile, Policy>::scalar_type;
  scalar_type norm2_squared = 0;
  for (auto &t : a) {
    norm2_squared += squared_norm(t);
  }
  return double(std::sqrt(norm2_squared));
}

template <typename Tile, typename Policy>
inline void zero(DistArrayVector<Tile, Policy> &a) {
  for (auto &t : a) {
    TiledArray::math::linalg::zero(t);
  }
}

template <typename Tile, typename Policy>
inline auto dot(const DistArrayVector<Tile, Policy> &a,
                const DistArrayVector<Tile, Policy> &b) {
  TA_ASSERT(a.size() == b.size());
  typename DistArrayVector<Tile, Policy>::scalar_type result = 0;
  for (decltype(a.size()) i = 0; i != a.size(); ++i) {
    result += dot(a[i], b[i]);
    // it seems to have some random issue (hang queue error) when calling
    // dot_product adding fence() seems to help prevent it
    TiledArray::get_default_world().gop.fence();
  }
  return result;
}

template <typename Tile, typename Policy, typename Scalar>
inline void axpy(DistArrayVector<Tile, Policy> &y, Scalar a,
                 const DistArrayVector<Tile, Policy> &x) {
  TA_ASSERT(x.size() == y.size());
  for (decltype(x.size()) i = 0; i != x.size(); ++i) {
    TiledArray::math::linalg::axpy(y[i], a, x[i]);
  }
}

// template <typename Tile, typename Policy>
// inline DistArrayVector<Tile, Policy> copy(DistArrayVector<Tile, Policy> &a) {
//   return a;
// }

template <typename Tile, typename Policy, typename Scalar>
inline void scale(DistArrayVector<Tile, Policy> &y, Scalar a) {
  for (auto &t : y) {
    TiledArray::math::linalg::scale(t, a);
  }
}

}  // namespace TiledArray

namespace madness {

template <typename T>
auto trace(const std::vector<madness::Future<T>> &vec) {
  T result = 0;
  for (auto &&v : vec) result += v.get();
  return result;
}

template <typename T, std::size_t Size>
auto trace(
    const TiledArray::container::svector<madness::Future<T>, Size> &vec) {
  T result = 0;
  for (auto &&v : vec) result += v.get();
  return result;
}
}  // namespace madness

#endif  // TILEDARRAY_DIST_ARRAY_VECTOR_H__INCLUDED
