/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_EXPRESSIONS_CONTRACTION_HELPERS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_CONTRACTION_HELPERS_H__INCLUDED

#include "TiledArray/conversions/make_array.h"
#include "TiledArray/expressions/index_list.h"
#include "TiledArray/expressions/tsr_expr.h"
#include "TiledArray/tensor/tensor.h"

namespace TiledArray::expressions {

/// Assembles the range for the target annotation
///
/// When dealing with Einstein notation it is common to need to assemble
/// iteration ranges/tensor sizes from the annotations of a tensor. More
/// specifically, one needs to map the annotations to extents and use those
/// extents to form a range. This function wraps that process. This function
/// also ensures that all modes labeled with the same annotation have the same
/// extent (*i.e.*, that they are compatible).
///
/// \tparam LHSType The type of the tensor on the left side of the `*`. Assumed
///                 to satisfy the concept of tile.
/// \tparam RHSType The type of the tensor on the right side of the `*`. Assumed
///                 to satisfy the concept of tile.
/// \param[in] target_idxs The annotation for the range we are creating.
/// \param[in] lhs_idxs The annotation for the tensor on the left of the `*`.
/// \param[in] rhs_idxs The annotation for the tensor on the right of the `*`.
/// \param[in] lhs The tensor on the left of the `*`.
/// \param[in] rhs The tensor on the righ of the `*`.
/// \return A range such that the extent of the i-th mode is the extent in
///         \p lhs and/or \p rhs associated with annotation `target_idxs[i]`.
/// \throw TiledArray::Exception if \p lhs and \p rhs do not agree on the extent
///                              for a particular annotation. Strong throw
///                              guarantee.
template <typename IndexList_, typename LHSType, typename RHSType>
auto range_from_annotation(const IndexList_& target_idxs,
                           const IndexList_& lhs_idxs,
                           const IndexList_& rhs_idxs, LHSType&& lhs,
                           RHSType&& rhs) {
  using range_type = std::decay_t<decltype(lhs.range())>;
  using size_type = typename range_type::size_type;
  using extent_type = std::pair<size_type, size_type>;

  container::svector<extent_type> ranges;  // Will be the ranges for each extent
  const auto& lrange = lhs.range();
  const auto& rrange = rhs.range();

  for (const auto& idx : target_idxs) {
    const auto lmodes = lhs_idxs.positions(idx);
    const auto rmodes = rhs_idxs.positions(idx);
    TA_ASSERT(lmodes.size() || rmodes.size());  // One of them better have it

    auto corr_extent =
        lmodes.size() ? lrange.dim(lmodes[0]) : rrange.dim(rmodes[0]);
    for (auto lmode : lmodes) TA_ASSERT(lrange.dim(lmode) == corr_extent);
    for (auto rmode : rmodes) TA_ASSERT(rrange.dim(rmode) == corr_extent);
    ranges.emplace_back(std::move(corr_extent));
  }
  return range_type(ranges);
}

template <typename IndexList_, typename LHSType, typename RHSType>
auto trange_from_annotation(const IndexList_& target_idxs,
                            const IndexList_& lhs_idxs,
                            const IndexList_& rhs_idxs, LHSType&& lhs,
                            RHSType&& rhs) {
  container::svector<TiledRange1> ranges;  // Will be the ranges for each extent
  const auto& lrange = lhs.trange();
  const auto& rrange = rhs.trange();

  for (const auto& idx : target_idxs) {
    const auto lmodes = lhs_idxs.positions(idx);
    const auto rmodes = rhs_idxs.positions(idx);
    TA_ASSERT(lmodes.size() || rmodes.size());  // One of them better have it

    auto corr_extent =
        lmodes.size() ? lrange.dim(lmodes[0]) : rrange.dim(rmodes[0]);
    for (auto lmode : lmodes) TA_ASSERT(lrange.dim(lmode) == corr_extent);
    for (auto rmode : rmodes) TA_ASSERT(rrange.dim(rmode) == corr_extent);
    ranges.emplace_back(std::move(corr_extent));
  }
  return TiledRange(ranges.begin(), ranges.end());
}

/// Maps a tensor's annotation to an actual index.
///
/// This function assumes that the contraction is being done in a loop form with
/// one loop running over free indices (those present on both sides of the
/// assignment) and one running over the bound indices (those only present on
/// the right side of the assignment). Subject to this assumption, this function
/// will map the coordinate indices of the two loops to the coordinate index of
/// the tensor by using the annotation of the tensor.
///
/// \tparam IndexType The type used to hold ordinal indices. Assumed to satisfy
///         random-access container.
/// \param[in] free_vars An index list containing the free
/// variables
///            of the contraction.
/// \param[in] bound_vars An index list containing the bound
/// variables
///            of the contraction.
/// \param[in] tensor_vars An index list containing the
/// annotation for
///            the tensor we want the index of.
/// \param[in] free_idx A coordinate index such that `free_idx[i]` is the offset
///            along modes annotated `free_vars[i]`.
/// \param[in] bound_idx A coordinate index such that `bound_idx[i]` is the
///            offset along modes annotated `bound_vars[i]`.
/// \return A coordinate index such that the i-th element is the offset
///         associated with annotation `tensor_vars[i]`.
template <typename IndexList_, typename IndexType>
auto make_index(const IndexList_& free_vars, const IndexList_& bound_vars,
                const IndexList_& tensor_vars, IndexType&& free_idx,
                IndexType&& bound_idx) {
  std::decay_t<IndexType> rv(tensor_vars.size());
  for (std::size_t i = 0; i < tensor_vars.size(); ++i) {
    const auto& x = tensor_vars[i];
    const bool is_free = free_vars.count(x);
    const auto modes =
        is_free ? free_vars.positions(x) : bound_vars.positions(x);
    TA_ASSERT(modes.size() == 1);  // Annotation should only appear once
    rv[i] = is_free ? free_idx[modes[0]] : bound_idx[modes[0]];
  }
  return rv;
}

/// Wraps process of getting a list with the bound variables
inline auto make_bound_annotation(const BipartiteIndexList& free_vars,
                                  const BipartiteIndexList& lhs_vars,
                                  const BipartiteIndexList& rhs_vars) {
  const auto bound_temp = bound_annotations(free_vars, lhs_vars, rhs_vars);
  BipartiteIndexList bound_vars(
      container::svector<std::string>(bound_temp.begin(), bound_temp.end()),
      container::svector<std::string>{});
  return bound_vars;
}

/// Wraps process of getting a list with the bound variables
inline auto make_bound_annotation(const IndexList& free_vars,
                                  const IndexList& lhs_vars,
                                  const IndexList& rhs_vars) {
  const auto bound_temp = bound_annotations(free_vars, lhs_vars, rhs_vars);
  IndexList bound_vars(bound_temp.begin(), bound_temp.end());
  return bound_vars;
}

namespace kernels {

// Contract two tensors to a scalar
template <typename IndexList_, typename LHSType, typename RHSType>
auto s_t_t_contract_(const IndexList_& free_vars, const IndexList_& lhs_vars,
                     const IndexList_& rhs_vars, LHSType&& lhs, RHSType&& rhs) {
  using value_type = typename std::decay_t<LHSType>::value_type;

  TA_ASSERT(free_vars.size() == 0);
  TA_ASSERT(lhs_vars.size() > 0);
  TA_ASSERT(rhs_vars.size() > 0);
  TA_ASSERT(inner_size(lhs_vars) == 0);
  TA_ASSERT(inner_size(rhs_vars) == 0);
  TA_ASSERT(lhs_vars.is_permutation(rhs_vars));

  // Get the indices being contracted over
  const auto bound_vars = make_bound_annotation(free_vars, lhs_vars, rhs_vars);

  // Lambdas to bind the annotations, making it easier to get coordinate indices
  auto lhs_idx = [=](const auto& bound_idx) {
    const std::decay_t<decltype(bound_idx)> empty;
    return make_index(free_vars, bound_vars, lhs_vars, empty, bound_idx);
  };

  auto rhs_idx = [=](const auto& bound_idx) {
    const std::decay_t<decltype(bound_idx)> empty;
    return make_index(free_vars, bound_vars, rhs_vars, empty, bound_idx);
  };

  auto bound_range =
      range_from_annotation(bound_vars, lhs_vars, rhs_vars, lhs, rhs);

  value_type rv = 0;
  for (const auto& bound_idx : bound_range) {
    const auto& lhs_elem = lhs(lhs_idx(bound_idx));
    const auto& rhs_elem = rhs(rhs_idx(bound_idx));
    rv += lhs_elem * rhs_elem;
  }
  return rv;
}

// Contract two tensors to a tensor
template <typename IndexList_, typename LHSType, typename RHSType>
auto t_s_t_contract_(const IndexList_& free_vars, const IndexList_& lhs_vars,
                     const IndexList_& rhs_vars, LHSType&& lhs, RHSType&& rhs) {
  auto rhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(free_vars, std::decay_t<IndexList_>{}, rhs_vars, free_idx,
                      bound_idx);
  };

  // We need to avoid passing lhs since it's a double, ranges all come from rhs
  // so it doesn't matter if we pass it twice
  auto orange = range_from_annotation(free_vars, lhs_vars, rhs_vars, rhs, rhs);
  std::decay_t<RHSType> rv(orange, 0.0);
  std::decay_t<decltype(*rhs.range().begin())> empty;
  for (const auto& free_idx : orange) {
    auto& out_elem = rv(free_idx);
    const auto& rhs_elem = rhs(rhs_idx(free_idx, empty));
    out_elem += lhs * rhs_elem;
  }
  return rv;
}

// Contract two tensors to a tensor
template <typename IndexList_, typename LHSType, typename RHSType>
auto t_t_t_contract_(const IndexList_& free_vars, const IndexList_& lhs_vars,
                     const IndexList_& rhs_vars, LHSType&& lhs, RHSType&& rhs) {
  // Get the indices being contracted over
  const auto bound_vars = make_bound_annotation(free_vars, lhs_vars, rhs_vars);

  // Lambdas to bind the annotations, making it easier to get coordinate indices
  auto lhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(free_vars, bound_vars, lhs_vars, free_idx, bound_idx);
  };

  auto rhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(free_vars, bound_vars, rhs_vars, free_idx, bound_idx);
  };

  auto orange = range_from_annotation(free_vars, lhs_vars, rhs_vars, lhs, rhs);
  std::decay_t<LHSType> rv(orange, 0.0);

  if (bound_vars.size() == 0) {
    std::decay_t<decltype(*lhs.range().begin())> empty;
    for (const auto& free_idx : orange) {
      auto& out_elem = rv(free_idx);
      const auto& lhs_elem = lhs(lhs_idx(free_idx, empty));
      const auto& rhs_elem = rhs(rhs_idx(free_idx, empty));
      out_elem += lhs_elem * rhs_elem;
    }
  } else {
    auto brange =
        range_from_annotation(bound_vars, lhs_vars, rhs_vars, lhs, rhs);

    for (const auto& free_idx : orange) {
      auto& out_elem = rv(free_idx);
      for (const auto& bound_idx : brange) {
        const auto& lhs_elem = lhs(lhs_idx(free_idx, bound_idx));
        const auto& rhs_elem = rhs(rhs_idx(free_idx, bound_idx));
        out_elem += lhs_elem * rhs_elem;
      }
    }
  }
  return rv;
}

// Contract two ToTs to a ToT
template <typename IndexList_, typename LHSType, typename RHSType>
auto t_tot_tot_contract_(const IndexList_& free_vars,
                         const IndexList_& lhs_vars, const IndexList_& rhs_vars,
                         LHSType&& lhs, RHSType&& rhs) {
  // Break the annotations up into their inner and outer parts
  const auto free_ovars = outer(free_vars);
  const auto lhs_ovars = outer(lhs_vars);
  const auto lhs_ivars = inner(lhs_vars);
  const auto rhs_ovars = outer(rhs_vars);
  const auto rhs_ivars = inner(rhs_vars);

  // We assume there's no operation going across the outer and inner tensors
  // (i.e., the set of outer annotations must be disjoint from the inner)
  {
    auto all_outer = all_annotations(free_vars, lhs_ovars, rhs_ovars);
    auto all_inner = all_annotations(lhs_ivars, rhs_ivars);
    TA_ASSERT(common_annotations(all_outer, all_inner).size() == 0);
  }

  // Get the outer indices being contracted over
  const auto bound_ovars =
      make_bound_annotation(free_ovars, lhs_ovars, rhs_ovars);

  // lambdas to bind annotations, making it easier to get coordinate indices
  auto lhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(free_ovars, bound_ovars, lhs_ovars, free_idx, bound_idx);
  };

  auto rhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(free_ovars, bound_ovars, rhs_ovars, free_idx, bound_idx);
  };

  auto orange =
      range_from_annotation(free_ovars, lhs_ovars, rhs_ovars, lhs, rhs);
  Tensor<typename std::decay_t<LHSType>::numeric_type> rv(orange, 0.0);

  // If bound_vars is empty we're doing Hadamard on the outside
  if (bound_ovars.size() == 0) {  // Hadamard on the outside
    std::decay_t<decltype(*lhs.range().begin())> empty;
    for (const auto& free_idx : orange) {
      const auto& inner_lhs = lhs(lhs_idx(free_idx, empty));
      const auto& inner_rhs = rhs(rhs_idx(free_idx, empty));
      rv(free_idx) += s_t_t_contract_(IndexList{}, lhs_ivars, rhs_ivars,
                                      inner_lhs, inner_rhs);
    }
  } else {
    auto bound_range =
        range_from_annotation(bound_ovars, lhs_ovars, rhs_ovars, lhs, rhs);

    for (const auto& free_idx : orange) {
      auto& inner_out = rv(free_idx);
      for (const auto& bound_idx : bound_range) {
        const auto& inner_lhs = lhs(lhs_idx(free_idx, bound_idx));
        const auto& inner_rhs = rhs(rhs_idx(free_idx, bound_idx));
        inner_out += s_t_t_contract_(IndexList{}, lhs_ivars, rhs_ivars,
                                     inner_lhs, inner_rhs);
      }
    }
  }
  return rv;
}

// Contract a tensor and a ToTs to a ToT
template <typename IndexList_, typename LHSType, typename RHSType>
auto tot_t_tot_contract_(const IndexList_& out_vars, const IndexList_& lhs_vars,
                         const IndexList_& rhs_vars, LHSType&& lhs,
                         RHSType&& rhs) {
  // Break the annotations up into their inner and outer parts
  const auto out_ovars = outer(out_vars);
  const auto out_ivars = inner(out_vars);
  const auto lhs_ovars = outer(lhs_vars);
  const auto lhs_ivars = inner(lhs_vars);
  const auto rhs_ovars = outer(rhs_vars);
  const auto rhs_ivars = inner(rhs_vars);

  // We assume lhs is either being contracted with the outer indices or the
  // inner indices
  bool with_outer = common_annotations(lhs_ovars, rhs_ovars).size() != 0;
  bool with_inner = common_annotations(lhs_ovars, rhs_ivars).size() != 0;
  TA_ASSERT(!(with_outer && with_inner));

  if (with_inner) {
    auto orange =
        range_from_annotation(out_ovars, lhs_ivars, rhs_ovars, lhs, rhs);
    using tot_type = std::decay_t<RHSType>;
    typename tot_type::value_type default_tile;
    tot_type rv(orange, default_tile);
    const auto bound_vars =
        make_bound_annotation(out_ovars, lhs_ivars, rhs_ovars);
    auto rhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
      return make_index(out_ovars, bound_vars, rhs_ovars, free_idx, bound_idx);
    };
    for (const auto& free_idx : orange) {
      auto& inner_out = rv(free_idx);
      std::decay_t<decltype(free_idx)> empty;
      const auto& inner_rhs = rhs(rhs_idx(free_idx, empty));
      const auto elem =
          t_t_t_contract_(out_ivars, lhs_ovars, rhs_ivars, lhs, inner_rhs);
      if (inner_out != default_tile) {
        inner_out += elem;
      } else {
        rv(free_idx) = elem;
      }
    }
    return rv;
  }

  // We assume there's no operation going across the outer and inner tensors
  // (i.e., the set of outer annotations must be disjoint from the inner)
  {
    auto all_outer = all_annotations(out_ovars, lhs_ovars, rhs_ovars);
    auto all_inner = all_annotations(out_ivars, lhs_ivars, rhs_ivars);
    TA_ASSERT(common_annotations(all_outer, all_inner).size() == 0);
  }

  // Get the outer indices being contracted over
  const auto bound_vars =
      make_bound_annotation(out_ovars, lhs_ovars, rhs_ovars);

  // lambdas to bind annotations, making it easier to get coordinate indices
  auto lhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(out_ovars, bound_vars, lhs_ovars, free_idx, bound_idx);
  };

  auto rhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(out_ovars, bound_vars, rhs_ovars, free_idx, bound_idx);
  };

  auto orange =
      range_from_annotation(out_ovars, lhs_ovars, rhs_ovars, lhs, rhs);
  using tot_type = std::decay_t<RHSType>;
  typename tot_type::value_type default_tile;
  tot_type rv(orange, default_tile);

  // If bound_vars is empty we're doing Hadamard on the outside
  if (bound_vars.size() == 0) {  // Hadamard on the outside
    std::decay_t<decltype(*lhs.range().begin())> empty;
    for (const auto& free_idx : orange) {
      auto& inner_out = rv(free_idx);
      const auto& inner_lhs = lhs(lhs_idx(free_idx, empty));
      const auto& inner_rhs = rhs(rhs_idx(free_idx, empty));
      const auto elem = t_s_t_contract_(out_ivars, lhs_ivars, rhs_ivars,
                                        inner_lhs, inner_rhs);
      if (inner_out != default_tile) {
        inner_out += elem;
      } else {
        rv(free_idx) = elem;
      }
    }
  } else {
    auto bound_range =
        range_from_annotation(bound_vars, lhs_ovars, rhs_ovars, lhs, rhs);
    for (const auto& free_idx : orange) {
      auto& inner_out = rv(free_idx);
      for (const auto& bound_idx : bound_range) {
        const auto& inner_lhs = lhs(lhs_idx(free_idx, bound_idx));
        const auto& inner_rhs = rhs(rhs_idx(free_idx, bound_idx));
        const auto elem = t_s_t_contract_(out_ivars, lhs_ivars, rhs_ivars,
                                          inner_lhs, inner_rhs);
        if (inner_out != default_tile) {
          inner_out += elem;
        } else {
          rv(free_idx) = elem;
        }
      }
    }
  }
  return rv;
}

// Contract two ToTs to a ToT
template <typename IndexList_, typename LHSType, typename RHSType>
auto tot_tot_tot_contract_(const IndexList_& out_vars,
                           const IndexList_& lhs_vars,
                           const IndexList_& rhs_vars, LHSType&& lhs,
                           RHSType&& rhs) {
  // Break the annotations up into their inner and outer parts
  const auto out_ovars = outer(out_vars);
  const auto out_ivars = inner(out_vars);
  const auto lhs_ovars = outer(lhs_vars);
  const auto lhs_ivars = inner(lhs_vars);
  const auto rhs_ovars = outer(rhs_vars);
  const auto rhs_ivars = inner(rhs_vars);

  // We assume there's no operation going across the outer and inner tensors
  // (i.e., the set of outer annotations must be disjoint from the inner)
  {
    auto all_outer = all_annotations(out_ovars, lhs_ovars, rhs_ovars);
    auto all_inner = all_annotations(out_ivars, lhs_ivars, rhs_ivars);
    TA_ASSERT(common_annotations(all_outer, all_inner).size() == 0);
  }

  // Get the outer indices being contracted over
  const auto bound_vars =
      make_bound_annotation(out_ovars, lhs_ovars, rhs_ovars);

  // lambdas to bind annotations, making it easier to get coordinate indices
  auto lhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(out_ovars, bound_vars, lhs_ovars, free_idx, bound_idx);
  };

  auto rhs_idx = [=](const auto& free_idx, const auto& bound_idx) {
    return make_index(out_ovars, bound_vars, rhs_ovars, free_idx, bound_idx);
  };

  auto orange =
      range_from_annotation(out_ovars, lhs_ovars, rhs_ovars, lhs, rhs);
  using tot_type = std::decay_t<LHSType>;
  typename tot_type::value_type default_tile;
  tot_type rv(orange, default_tile);

  // If bound_vars is empty we're doing Hadamard on the outside
  if (bound_vars.size() == 0) {  // Hadamard on the outside
    std::decay_t<decltype(*lhs.range().begin())> empty;
    for (const auto& free_idx : orange) {
      auto& inner_out = rv(free_idx);
      const auto& inner_lhs = lhs(lhs_idx(free_idx, empty));
      const auto& inner_rhs = rhs(rhs_idx(free_idx, empty));
      const auto elem = t_t_t_contract_(out_ivars, lhs_ivars, rhs_ivars,
                                        inner_lhs, inner_rhs);
      if (inner_out != default_tile) {
        inner_out += elem;
      } else {
        rv(free_idx) = elem;
      }
    }
  } else {
    auto bound_range =
        range_from_annotation(bound_vars, lhs_ovars, rhs_ovars, lhs, rhs);
    for (const auto& free_idx : orange) {
      auto& inner_out = rv(free_idx);
      for (const auto& bound_idx : bound_range) {
        const auto& inner_lhs = lhs(lhs_idx(free_idx, bound_idx));
        const auto& inner_rhs = rhs(rhs_idx(free_idx, bound_idx));
        const auto elem = t_t_t_contract_(out_ivars, lhs_ivars, rhs_ivars,
                                          inner_lhs, inner_rhs);
        if (inner_out != default_tile) {
          inner_out += elem;
        } else {
          rv(free_idx) = elem;
        }
      }
    }
  }
  return rv;
}

template <bool out_is_tot, bool lhs_is_tot, bool rhs_is_tot>
struct KernelSelector;

template <>
struct KernelSelector<true, true, true> {
  template <typename IndexList_, typename LTileType, typename RTileType>
  auto operator()(const IndexList_& ovars, const IndexList_& lvars,
                  const IndexList_& rvars, LTileType&& ltile,
                  RTileType&& rtile) const {
    return tot_tot_tot_contract_(ovars, lvars, rvars,
                                 std::forward<LTileType>(ltile),
                                 std::forward<RTileType>(rtile));
  }
};

template <>
struct KernelSelector<false, true, true> {
  template <typename IndexList_, typename LTileType, typename RTileType>
  auto operator()(const IndexList_& ovars, const IndexList_& lvars,
                  const IndexList_& rvars, LTileType&& ltile,
                  RTileType&& rtile) const {
    return t_tot_tot_contract_(ovars, lvars, rvars,
                               std::forward<LTileType>(ltile),
                               std::forward<RTileType>(rtile));
  }
};

template <>
struct KernelSelector<true, false, true> {
  template <typename IndexList_, typename LTileType, typename RTileType>
  auto operator()(const IndexList_& ovars, const IndexList_& lvars,
                  const IndexList_& rvars, LTileType&& ltile,
                  RTileType&& rtile) const {
    return tot_t_tot_contract_(ovars, lvars, rvars,
                               std::forward<LTileType>(ltile),
                               std::forward<RTileType>(rtile));
  }
};

}  // namespace kernels

template <typename ResultType, typename LHSType, typename RHSType>
void einsum(TsrExpr<ResultType, true> out, const TsrExpr<LHSType, true>& lhs,
            const TsrExpr<RHSType, true>& rhs) {
  const BipartiteIndexList ovars(out.annotation());
  const BipartiteIndexList lvars(lhs.annotation());
  const BipartiteIndexList rvars(rhs.annotation());

  using out_tile_type = typename ResultType::value_type;
  using lhs_tile_type = typename LHSType::value_type;
  using rhs_tile_type = typename RHSType::value_type;

  constexpr bool out_is_tot =
      TiledArray::detail::is_tensor_of_tensor_v<out_tile_type>;
  constexpr bool lhs_is_tot =
      TiledArray::detail::is_tensor_of_tensor_v<lhs_tile_type>;
  constexpr bool rhs_is_tot =
      TiledArray::detail::is_tensor_of_tensor_v<rhs_tile_type>;

  const auto out_ovars = outer(ovars);
  const auto lhs_ovars = outer(lvars);
  const auto rhs_ovars = outer(rvars);

  const auto bound_vars =
      make_bound_annotation(out_ovars, lhs_ovars, rhs_ovars);

  const auto& ltensor = lhs.array();
  const auto& rtensor = rhs.array();

  const auto orange =
      trange_from_annotation(out_ovars, lhs_ovars, rhs_ovars, ltensor, rtensor);
  const auto brange = trange_from_annotation(bound_vars, lhs_ovars, rhs_ovars,
                                             ltensor, rtensor);

  kernels::KernelSelector<out_is_tot, lhs_is_tot, rhs_is_tot> selector;

  auto l = [=](auto& tile, const Range& r) {
    const auto oidx =
        orange.tiles_range().idx(orange.element_to_tile(r.lobound()));
    auto bitr = brange.tiles_range().begin();
    const auto eitr = brange.tiles_range().end();
    do {
      const bool have_bound = bitr != eitr;
      decltype(oidx) bidx = have_bound ? *bitr : oidx;
      auto lidx = make_index(out_ovars, bound_vars, lhs_ovars, oidx, bidx);
      auto ridx = make_index(out_ovars, bound_vars, rhs_ovars, oidx, bidx);
      if (!ltensor.shape().is_zero(lidx) && !rtensor.shape().is_zero(ridx)) {
        const auto& ltile = ltensor.find(lidx).get();
        const auto& rtile = rtensor.find(ridx).get();
        if (tile.empty())
          tile = selector(ovars, lvars, rvars, ltile, rtile);
        else
          tile += selector(ovars, lvars, rvars, ltile, rtile);
      }
      if (have_bound) ++bitr;
    } while (bitr != brange.tiles_range().end());
    return !tile.empty() ? tile.norm() : 0.0;
  };

  auto rv = make_array<ResultType>(ltensor.world(), orange, l);
  out.array() = rv;
  ltensor.world().gop.fence();
}

}  // namespace TiledArray::expressions

#endif  // TILEDARRAY_EXPRESSIONS_CONTRACTION_HELPERS_H__INCLUDED
