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

#ifndef ITERATIONTEST_H__INCLUDED
#define ITERATIONTEST_H__INCLUDED

/// Tests basic iteration functionality and compares the elements in the
/// container with expected values. It returns the number of values that were
/// equal.
template <typename Container, typename InIter>
unsigned int iteration_test(Container&, InIter, InIter);
template <typename Container, typename InIter>
unsigned int const_iteration_test(const Container&, InIter, InIter);

/// Tests basic iteration functionality and compares the elements in the
/// container with expected values. It returns the number of values that were
/// equal.
template <typename Container, typename InIter>
unsigned int iteration_test(Container& c, InIter first, InIter last) {
  unsigned int result = 0;
  for (typename Container::iterator it = c.begin();
       it != c.end() && first != last; ++it, ++first)
    if (*it == *first) ++result;

  return true;
}

/// Tests basic const_iteration functionality and compares the elements in the
/// container with expected values. It returns the number of values that were
/// equal.
template <typename Container, typename InIter>
unsigned int const_iteration_test(const Container& c, InIter first,
                                  InIter last) {
  unsigned int result = 0;
  for (typename Container::const_iterator it = c.begin();
       it != c.end() && first != last; ++it, ++first)
    if (*it == *first) ++result;

  return result;
}

#endif  // ITERATIONTEST_H__INCLUDED
