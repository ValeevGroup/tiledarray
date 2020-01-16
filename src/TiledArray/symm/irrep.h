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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  irrep.h
 *  May 18, 2015
 *
 */

#ifndef TILEDARRAY_IRREP_H__INCLUDED
#define TILEDARRAY_IRREP_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/external/madness.h>

namespace TiledArray {

/// Irrep of an \f$ S_n \f$ symmetric group

/** The data is represented using Yamanouchi symbol, \f$ M \f$ , which is a
 row of \f$ n \f$ numbers \f$ M_i \f$ \f$ (i = 0, \dots , n-1) \f$, where
 \f$ M_i \f$ is the number of the row in the standard Young tableau,
 counting from above, in which the number \f$ i \f$ appears. For example,
 the standard irreps for the \f$ S_3 \f$ group are:
 \f[
   \begin{tabular}{ c c c }
     Young tableaux & partition [$\mu$] & Yamanouchi symbols M \\
     & & \\
     \begin{tabular}{ |c| }
        \hline
        1 \\ \hline
        2 \\ \hline
        3 \\ \hline
     \end{tabular} & [111] & 123 \\
     & & \\
     \begin{tabular}{ |c|c|c| }
        \hline
        1 & 2 & 3 \\ \hline
     \end{tabular} & [3] & 111 \\
     & & \\
     \begin{tabular}{ |c|c }
        \hline
        1 & \multicolumn{1}{|c|}{2} \\ \hline
        3 \\ \cline{1-1}
     \end{tabular} & [21] & 112 \\
     & & \\
     \begin{tabular}{ |c|c }
        \hline
        1 & \multicolumn{1}{|c|}{3} \\ \hline
        2 \\ \cline{1-1}
     \end{tabular} & [21] & 121 \\
   \end{tabular}
 \f]
 To construct an irrep, you must provide the partition for the Young
 tableaux and the Yamanouchi symbols as follows:
 \code
 Irrep S3_irrep({1,1,1}, {1,2,3});
 \endcode
 */
class Irrep {
  /// The Yamanouchi symbols for the irrep
  std::unique_ptr<unsigned int[]>
      data_;             ///< Data of the irrep
                         ///< { mu_0, ... , mu_degree-1, M_0, ..., M_degree-1 }
  unsigned int degree_;  ///< The degree of the symmetry group

 public:
  Irrep() = delete;
  Irrep(Irrep&&) = default;
  ~Irrep() = default;
  Irrep& operator=(Irrep&&) = default;

  Irrep(const Irrep& other)
      : data_(std::make_unique<unsigned int[]>(other.degree_ << 1)),
        degree_(other.degree_) {
    std::copy_n(other.data_.get(), other.degree_ << 1, data_.get());
  }

  /// Irrep constructor
  Irrep(const std::initializer_list<unsigned int>& mu,
        const std::initializer_list<unsigned int>& M)
      : data_(std::make_unique<unsigned int[]>(M.size() << 1u)),
        degree_(M.size()) {
    TA_ASSERT(mu.size() > 0ul);
    TA_ASSERT(M.size() > 0ul);
    TA_ASSERT(mu.size() <= M.size());

    // Fill the data of the irrep
    std::fill(std::copy(mu.begin(), mu.end(), data_.get()),
              data_.get() + degree_, 0u);
    std::copy(M.begin(), M.end(), data_.get() + degree_);

#ifndef NDEBUG
    {
      const unsigned int* MADNESS_RESTRICT const M = data_.get() + degree_;
      const unsigned int* MADNESS_RESTRICT const mu = data_.get();
      unsigned int M_max = 0u;
      unsigned int mu_sum = 0u;
      for (unsigned int i = 0u; i < degree_; ++i) {
        // Validate the partition data
        if (i > 0u) TA_ASSERT(mu[i] <= mu[i - 1u]);
        mu_sum += mu[i];

        // Validate the Yamanouchi symbols data
        TA_ASSERT(M[i] <= (M_max + 1u));
        M_max = std::max(M[i], M_max);
        TA_ASSERT(std::count(M, M + degree_, i + 1u) == mu[i]);
      }

      // Check that the correct number of elements are in the partition data
      TA_ASSERT(mu_sum == degree_);
    }
#endif  // NDEBUG
  }

  /// Copy operator

  /// \param other The irrep to be copied
  /// \return A reference to this irrep
  Irrep& operator=(const Irrep& other) {
    if (degree_ != other.degree_) {
      data_ = std::make_unique<unsigned int[]>(other.degree_ << 1);
      degree_ = other.degree_;
    }
    std::copy_n(other.data_.get(), other.degree_ << 1, data_.get());

    return *this;
  }

  /// Irrep degree accessor

  /// \return The degree of the symmetry group to which this irrep belongs
  unsigned int degree() const { return degree_; }

  /// Data accessor

  /// \return A const pointer to the partition and symbol data.
  /// \note The size of the data array is 2 * degree.
  const unsigned int* data() const { return data_.get(); }

};  // class Irrep

}  // namespace TiledArray

#endif  // TILEDARRAY_IRREP_H__INCLUDED
