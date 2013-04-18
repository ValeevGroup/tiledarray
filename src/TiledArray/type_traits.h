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

#ifndef TILEDARRAY_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TYPE_TRAITS_H__INCLUDED

#include <TiledArray/config.h>
#include <iterator>
#include <world/enable_if.h>
#include <world/typestuff.h>
#include <complex>

namespace Eigen {

  template <typename, int, int, int, int, int> class Matrix;
  template <typename, int, int, int, int, int> class Array;
  template <typename, int, typename> class Map;

} // namespace Eigen

namespace TiledArray {

  template <typename, unsigned int, typename> class Array;
  template <typename, typename> class Tensor;

  namespace expressions {

    template <typename> class TensorExpression;

  } // namespace expressions

  namespace detail {

    template <typename T>
    struct is_numeric : public std::false_type { };

    template <>
    struct is_numeric<short int> : public std::true_type { };

    template <>
    struct is_numeric<int> : public std::true_type { };

    template <>
    struct is_numeric<unsigned int> : public std::true_type { };

    template <>
    struct is_numeric<long int> : public std::true_type { };

    template <>
    struct is_numeric<unsigned long int> : public std::true_type { };

    template <>
    struct is_numeric<float> : public std::true_type { };

    template <>
    struct is_numeric<double> : public std::true_type { };

#ifdef TILEDARRAY_HAS_LONG_DOUBLE

    template <>
    struct is_numeric<long double> : public std::true_type { };

#endif //TILEDARRAY_HAS_LONG_DOUBLE

#ifdef TILEDARRAY_HAS_LONG_LONG

    template <>
    struct is_numeric<long long int> : public std::true_type { };

    template <>
    struct is_numeric<unsigned long long int> : public std::true_type { };

#endif // TILEDARRAY_HAS_LONG_LONG

    template <typename T>
    struct is_numeric<std::complex<T> > : public is_numeric<T> { };

    /// Type trait for extracting the scalar type of tensors and arrays.
    template <typename T, typename Enabler = typename madness::enable_if<is_numeric<T> >::type>
    struct scalar_type {
      typedef T type;
    };

    template <typename T, typename A>
    struct scalar_type<Tensor<T, A>, void> :
        public scalar_type<typename Tensor<T, A>::value_type>
    { };

    template <typename T, unsigned int DIM, typename Tile>
    struct scalar_type<Array<T, DIM, Tile>, void> :
        public scalar_type<typename Array<T, DIM, Tile>::value_type>
    { };

    template <typename Tile>
    struct scalar_type<expressions::TensorExpression<Tile>, void> :
        public scalar_type<typename expressions::TensorExpression<Tile>::value_type>
    { };

    template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
    struct scalar_type<Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>, void> :
        public scalar_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>::Scalar>
    { };

    template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
    struct scalar_type<Eigen::Array<T, Rows, Cols, Opts, MaxRows, MaxCols>, void> :
        public scalar_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>::Scalar>
    { };

    template <typename PlainObjectType, int MapOptions, typename StrideType>
    struct scalar_type<Eigen::Map<PlainObjectType, MapOptions, StrideType>, void> :
        public scalar_type<PlainObjectType>
    { };


    /// Remove const, volatile, and reference qualifiers.
    template <typename T>
    struct remove_cvr {
      typedef typename std::remove_cv<typename std::remove_reference<T>::type>::type type;
    };

    template <typename T>
    struct has_iterator_catagory
    {
        // yes and no are guaranteed to have different sizes,
        // specifically sizeof(yes) == 1 and sizeof(no) == 2
        typedef char yes[1];
        typedef char no[2];

        template <typename C>
        static yes& test(typename C::iterator_category*);

        template <typename>
        static no& test(...);

        // if the sizeof the result of calling test<T>(0) is equal to the sizeof(yes),
        // the first overload worked and T has a nested type named type.
        static const bool value = sizeof(test<T>(0)) == sizeof(yes);
    };

    struct non_iterator_tag { };

    template <typename T, typename Enabler = void>
    struct is_iterator : public std::false_type {
      typedef non_iterator_tag iterator_category;
    };

    template <typename T>
    struct is_iterator<T, typename madness::enable_if_c<has_iterator_catagory<T>::value >::type > : public std::true_type {
      typedef typename std::iterator_traits<T>::iterator_category iterator_category;
    };

    template <typename T>
    struct is_iterator<T*, void> : std::true_type {
      typedef typename std::iterator_traits<T*>::iterator_category iterator_category;
    };

    template <typename T>
    struct is_iterator<const T*, void> : std::true_type {
      typedef typename std::iterator_traits<const T*>::iterator_category iterator_category;
    };

    template <typename T>
    struct is_input_iterator :
        public std::is_base_of<std::input_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_output_iterator :
        public std::is_base_of<std::output_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_forward_iterator :
        public std::is_base_of<std::forward_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_bidirectional_iterator :
        public std::is_base_of<std::bidirectional_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_random_iterator :
        public std::is_base_of<std::random_access_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

  } // namespace detail
} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
