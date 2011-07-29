#ifndef TILEDARRAY_PERM_ALGORITHM_H__INCLUDED
#define TILEDARRAY_PERM_ALGORITHM_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/permutation.h>
#include <TiledArray/coordinate_system.h>
#include <vector>

namespace TiledArray {

  /// permute a std::array
  template <unsigned int DIM, typename T>
  std::array<T,DIM> operator^(const Permutation<DIM>& perm, const std::array<T, static_cast<std::size_t>(DIM) >& orig) {
    std::array<T,DIM> result;
    detail::permute_array(perm.begin(), perm.end(), orig.begin(), result.begin());
    return result;
  }

  /// permute a std::vector<T>
  template <unsigned int DIM, typename T>
  std::vector<T> operator^(const Permutation<DIM>& perm, const std::vector<T>& orig) {
    TA_ASSERT((orig.size() == DIM));
    std::vector<T> result(DIM);
    detail::permute_array<typename Permutation<DIM>::const_iterator, typename std::vector<T>::const_iterator, typename std::vector<T>::iterator>
      (perm.begin(), perm.end(), orig.begin(), result.begin());
    return result;
  }

  template <unsigned int DIM, typename T>
  std::vector<T> operator^=(std::vector<T>& orig, const Permutation<DIM>& perm) {
    orig = perm ^ orig;

    return orig;
  }

  template<unsigned int DIM>
  Permutation<DIM> operator ^(const Permutation<DIM>& perm, const Permutation<DIM>& p) {
    Permutation<DIM> result(perm ^ p.data());
    return result;
  }

  template <unsigned int DIM, typename T>
  std::array<T,DIM> operator ^=(std::array<T, static_cast<std::size_t>(DIM) >& a, const Permutation<DIM>& perm) {
    return (a = perm ^ a);
  }

  namespace detail {

    template <typename CS, typename Size, typename ArgArray, typename ResArray>
    void permute_tensor(const Permutation<CS::dim>& perm, const Size& size, const ArgArray& arg, ResArray& result) {
      typename CS::size_array p_size;
      permute_array(perm.begin(), perm.end(), size.begin(), p_size.begin());
      typename CS::size_array invp_weight = -perm ^ CS::calc_weight(p_size);

      typename CS::index i(0);
      const typename CS::index start(0);
      typename CS::ordinal_index res_ord = 0;
      typename CS::volume_type v = CS::calc_volume(size);

      for(typename CS::ordinal_index o = 0; o < v; ++o, CS::increment_coordinate(i, start, size)) {
        res_ord = CS::calc_ordinal(i, invp_weight);
        result[res_ord] = arg[o];
      }
    }

    template <unsigned int DIM, typename Size, typename ArgArray, typename ResArray>
    void permute_tensor(DimensionOrderType order, const Permutation<DIM>& perm, const Size& size, const ArgArray& arg, ResArray& result) {
      std::array<typename Size::value_type, DIM> weight;
      if(order == decreasing_dimension_order) {
        permute<CoordinateSystem<DIM, 0ul, decreasing_dimension_order,
          typename Size::value_type> >(perm, size, arg, result);
      } else {
        permute<CoordinateSystem<DIM, 0ul, increasing_dimension_order,
          typename Size::value_type> >(perm, size, arg, result);
      }
    }

  } // namespace detail
} // namespace TiledArray


#endif // TILEDARRAY_PERM_ALGORITHM_H__INCLUDED
