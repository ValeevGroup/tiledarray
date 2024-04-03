/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  util/time.h
 *  Oct 24, 2019
 *
 */

#ifndef TILEDARRAY_UTIL_TIME_H__INCLUDED
#define TILEDARRAY_UTIL_TIME_H__INCLUDED

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>

namespace TiledArray {

using time_point = std::chrono::high_resolution_clock::time_point;

inline time_point now() { return std::chrono::high_resolution_clock::now(); }

inline std::chrono::system_clock::time_point system_now() {
  return std::chrono::system_clock::now();
}

inline double duration_in_s(time_point const &t0, time_point const &t1) {
  return std::chrono::duration<double>{t1 - t0}.count();
}

inline int64_t duration_in_ns(time_point const &t0, time_point const &t1) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

namespace detail {
inline std::deque<double> &call_durations_accessor() {
  static std::deque<double> call_durations;
  return call_durations;
}
}  // namespace detail

/// Access recorded durations
inline const std::deque<double> &durations() {
  return detail::call_durations_accessor();
}

/// Clear recorded durations
inline void clear_durations() { detail::call_durations_accessor().clear(); }

/// Record duration since the given time point
/// \param tp_start The start time point
inline void record_duration_since(const time_point &tp_start) {
  detail::call_durations_accessor().push_back(duration_in_s(tp_start, now()));
}

/// Record duration of a single function call
template <typename F, typename... Args>
void record_duration(F &&f, Args &&...args) {
  auto tp_start = now();
  std::forward<F>(f)(std::forward<Args>(args)...);
  record_duration_since(tp_start);
}

/// Statistics of recorded durations
struct duration_stats_t {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
  double median = 0.0;
  double mean_reciprocal = 0.0;
};

/// Compute statistics of recorded durations
/// \return Statistics of recorded durations
inline duration_stats_t duration_statistics() {
  duration_stats_t stats;
  auto &durations = detail::call_durations_accessor();
  if (durations.empty()) return stats;

  stats.min = durations.front();
  stats.max = durations.front();
  stats.mean = durations.front();
  stats.mean_reciprocal = 1.0 / durations.front();
  double total = stats.mean;
  double total_reciprocal = stats.mean_reciprocal;
  for (size_t i = 1; i < durations.size(); ++i) {
    total += durations[i];
    total_reciprocal += 1. / durations[i];
    stats.min = std::min(stats.min, durations[i]);
    stats.max = std::max(stats.max, durations[i]);
  }
  stats.mean = total / durations.size();
  stats.mean_reciprocal = total_reciprocal / durations.size();

  double sum_sq = 0.0;
  for (size_t i = 0; i < durations.size(); ++i) {
    sum_sq += (durations[i] - stats.mean) * (durations[i] - stats.mean);
  }
  stats.stddev =
      durations.size() > 1 ? std::sqrt(sum_sq / (durations.size() - 1)) : 0.0;

  std::sort(durations.begin(), durations.end());
  stats.median = durations[durations.size() / 2];

  return stats;
}

}  // namespace TiledArray

#ifndef TA_RECORD_DURATION
/// Record duration of a statement
#define TA_RECORD_DURATION(statement) \
  TiledArray::record_duration([&] { statement; });
#endif  // !defined(TA_RECORD_DURATION)

#endif  // TILEDARRAY_UTIL_TIME_H__INCLUDED
