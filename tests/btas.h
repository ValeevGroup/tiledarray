/*
 * btas.h
 *
 *  Created on: Jul 11, 2017
 *      Author: evaleev
 */

#ifndef TESTS_BTAS_H_
#define TESTS_BTAS_H_

#include "TiledArray/config.h"

#include <btas/features.h>
#include <btas/tensor.h>

#include <madness/world/archive.h>

namespace TiledArray {
namespace detail {

template <typename T, typename ... Args>
struct is_tensor_helper<btas::Tensor<T, Args...> > : public std::true_type { };

template <typename T, typename ... Args>
struct is_contiguous_tensor_helper<btas::Tensor<T, Args...> > : public std::true_type { };

}
}

namespace madness {
  namespace archive {

#ifdef BTAS_HAS_BOOST_CONTAINER
  template <class Archive, typename T, std::size_t N, typename A>
  struct ArchiveLoadImpl<Archive, boost::container::small_vector<T, N, A>> {
    static inline void load(const Archive& ar,
                            boost::container::small_vector<T, N, A>& x) {
      std::size_t n{};
      ar& n;
      x.resize(n);
      for (auto& xi : x) ar& xi;
    }
  };

  template <class Archive, typename T, std::size_t N, typename A>
  struct ArchiveStoreImpl<Archive, boost::container::small_vector<T, N, A>> {
    static inline void store(const Archive& ar,
                            const boost::container::small_vector<T, N, A>& x) {
      ar& x.size();
      for (const auto& xi : x) ar & xi;
    }
  };
#endif

  template <class Archive, typename T>
  struct ArchiveLoadImpl<Archive, btas::varray<T>> {
    static inline void load(const Archive& ar, btas::varray<T>& x) {
      typename btas::varray<T>::size_type n{};
      ar& n;
      x.resize(n);
      for (typename btas::varray<T>::value_type& xi : x) ar& xi;
    }
    };

    template<class Archive, typename T>
    struct ArchiveStoreImpl<Archive, btas::varray<T> > {
        static inline void store(const Archive& ar, const btas::varray<T>& x) {
          ar & x.size();
          for (const typename btas::varray<T>::value_type& xi : x)
            ar & xi;
        }
    };

    template <class Archive, CBLAS_ORDER _Order, typename _Index>
    struct ArchiveLoadImpl<Archive, btas::BoxOrdinal<_Order, _Index>> {
      static inline void load(const Archive& ar,
                              btas::BoxOrdinal<_Order, _Index>& o) {
        typename btas::BoxOrdinal<_Order, _Index>::stride_type stride{};
        typename btas::BoxOrdinal<_Order, _Index>::value_type offset{};
        bool cont{};
        ar& stride& offset& cont;
        o = btas::BoxOrdinal<_Order, _Index>(
            std::move(stride), std::move(offset), std::move(cont));
      }
    };

    template <class Archive, CBLAS_ORDER _Order, typename _Index>
    struct ArchiveStoreImpl<Archive, btas::BoxOrdinal<_Order, _Index>> {
      static inline void store(const Archive& ar,
                               const btas::BoxOrdinal<_Order, _Index>& o) {
        ar& o.stride() & o.offset() & o.contiguous();
      }
    };

    template <class Archive, CBLAS_ORDER _Order, typename _Index,
              typename _Ordinal>
    struct ArchiveLoadImpl<Archive, btas::RangeNd<_Order, _Index, _Ordinal>> {
      static inline void load(const Archive& ar,
                              btas::RangeNd<_Order, _Index, _Ordinal>& r) {
        typedef typename btas::BaseRangeNd<
            btas::RangeNd<_Order, _Index, _Ordinal>>::index_type index_type;
        index_type lobound{}, upbound{};
        _Ordinal ordinal{};
        ar& lobound& upbound& ordinal;
        r = btas::RangeNd<_Order, _Index, _Ordinal>(
            std::move(lobound), std::move(upbound), std::move(ordinal));
      }
    };

    template <class Archive, CBLAS_ORDER _Order, typename _Index,
              typename _Ordinal>
    struct ArchiveStoreImpl<Archive, btas::RangeNd<_Order, _Index, _Ordinal>> {
      static inline void store(
          const Archive& ar, const btas::RangeNd<_Order, _Index, _Ordinal>& r) {
        ar & r.lobound() & r.upbound() & r.ordinal();
      }
    };

    template <class Archive, typename _T, class _Range, class _Store>
    struct ArchiveLoadImpl<Archive, btas::Tensor<_T, _Range, _Store>> {
      static inline void load(const Archive& ar,
                              btas::Tensor<_T, _Range, _Store>& t) {
        _Range range{};
        _Store store{};
        ar& range& store;
        t = btas::Tensor<_T, _Range, _Store>(std::move(range),
                                             std::move(store));
      }
    };

    template <class Archive, typename _T, class _Range, class _Store>
    struct ArchiveStoreImpl<Archive, btas::Tensor<_T, _Range, _Store>> {
      static inline void store(const Archive& ar,
                              const btas::Tensor<_T, _Range, _Store>& t) {
        ar & t.range() & t.storage();
      }
    };
  }
}



#endif /* TESTS_BTAS_H_ */
