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

#ifndef TILEDARRAY_COUNTER_PROBE_H__INCLUDED
#define TILEDARRAY_COUNTER_PROBE_H__INCLUDED

#include <madness/world/atomicint.h>

namespace TiledArray {
  namespace detail {

    /// Counter probe used to check for the completion of a set of tasks
    class CounterProbe {
    private:
      const madness::AtomicInt& counter_; ///< Counter incremented by the set of tasks
      const int n_; ///< The total number of tasks

    public:
      /// Constructor

      /// \param counter The task completion counter
      /// \param n The total number of tasks
      CounterProbe(const madness::AtomicInt& counter, const int n) :
        counter_(counter), n_(n)
      { }

      /// Probe function

      /// \return \c true when the counter is equal to the number of tasks
      bool operator()() const { return counter_ == n_; }
    }; // class CounterProbe

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_COUNTER_PROBE_H__INCLUDED
