#ifndef TILEDARRAY_EINSUM_H__INCLUDED
#define TILEDARRAY_EINSUM_H__INCLUDED

#include "TiledArray/expressions/detail/einsum_traits.h"
#include "TiledArray/expressions/detail/tensor_tot_kernel.h"
#include "TiledArray/expressions/detail/tot_tot_kernel.h"
#include "TiledArray/expressions/fwd.h"
#include "TiledArray/fwd.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/util/index.h"
#include "TiledArray/util/range.h"
//#include "TiledArray/util/string.h"

namespace TiledArray::expressions {

/// einsum function without result indices assumes every index present
/// in both @p A and @p B is contracted, or, if there are no free indices,
/// pure Hadamard product is performed.
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template <typename Array>
auto einsum(TsrExpr<Array> A, TsrExpr<Array> B) {
  printf("einsum(A,B)\n");
  auto a = std::get<0>(idx(A));
  auto b = std::get<0>(idx(B));
  Array R;
  R(a ^ b) = A * B;
  return R;
}

/** @brief einsum function with result indices explicitly specified
 *
 *  @tparam ArrayA The tensor type of @p A. Expected to be an instantiation of
 *                 DistArray.
 *  @tparam ArrayB The tensor type of @p B. Expected to be an instantiation of
 *                 DistArray.
 *
 *  @param[in] A first argument to the product
 *  @param[in] B second argument to the product
 *  @param[in] r result indices
 *
 *  @result The tensor resulting from multiplying @p A with @p B so that the
 *          result has indices consistent with @p r.
 *
 *  @warning just as in the plain expression code, reductions are a special
 *           case; use Expr::reduce()
 *
 */
template <typename ArrayA, typename ArrayB, typename... Indices>
auto einsum(TsrExpr<ArrayA> A, TsrExpr<ArrayB> B, const std::string &cs,
            World &world = get_default_world()) {
  using traits_type = detail::EinsumTraits<ArrayA, ArrayB>;
  using return_type = typename traits_type::return_type;
  return einsum(A, B, idx<return_type>(cs), world);
}

template <typename ArrayA, typename ArrayB, typename... Indices>
auto einsum(TsrExpr<ArrayA> A, TsrExpr<ArrayB> B,
            std::tuple<Index, Indices...> cs, World &world) {
  using traits_type = detail::EinsumTraits<ArrayA, ArrayB>;

  // Dispatch based on whether A and B are, or are not, ToTs
  if constexpr (traits_type::both_are_tots) {
    return detail::tot_tot_kernel(A, B, std::move(cs), world);
  } else if constexpr (traits_type::rhs_is_tot) {
    return detail::tensor_tot_kernel(A, B, std::move(cs), world);
  } else if constexpr (traits_type::lhs_is_tot) {
    // N.B. When indices are attached to the tensors, the tensors commute
    return detail::tensor_tot_kernel(B, A, std::move(cs), world);
  } else {
    // Else is triggered when neither A or B are a ToT
    //
    // N.B. C++ won't let us just put false, but getting here we know
    //     both_are_tots is false so it's just as good...
    static_assert(traits_type::both_are_tots,
                  "Einsum expects at least one ToT");
  }
}

}  // namespace TiledArray::expressions

#endif /* TILEDARRAY_EINSUM_H__INCLUDED */
