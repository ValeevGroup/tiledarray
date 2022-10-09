#ifndef TILEDARRAY_EINSUM_EIGEN_H__INCLUDED
#define TILEDARRAY_EINSUM_EIGEN_H__INCLUDED

#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/range.h"
#include "TiledArray/einsum/string.h"
#include "TiledArray/external/eigen.h"
#include "TiledArray/fwd.h"

namespace Eigen {

template <typename Derived, int Options>
const Derived &derived(const TensorBase<Derived, Options> &t) {
  return static_cast<const Derived &>(t);
}

/// Evaluates a binary tensor product specified by a string that follows
/// the
/// [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
/// format
/// \tparam TA a class derived from Eigen::TensorBase, e.g. Eigen::Tensor
/// \tparam TB a class derived from Eigen::TensorBase, e.g. Eigen::Tensor
/// \tparam TC a class derived from Eigen::TensorBase, e.g. Eigen::Tensor
/// \param expr string specification of the tensor product;
///        follows the _explicit_
///        [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
///        format, i.e. `"<AnnA>,<AnnB>-><AnnC>"`, where `<AnnX>` is an index
///        annotation for tensor `X`; note that slicing is not currently
///        supported, i.e. an index can only annotate a single mode of
///        a given tensor; for example, `"ii,k->ik"` is an invalid
///        value for \p expr .
/// \param A first argument
/// \param B second argument
/// \param C result
///
/// *Examples*:
/// - matrix multiplication: `einsum("ij,jk->ik",A,B,C)`
/// - matrix multiplication, with result transposed: `einsum("ij,jk->ki",A,B,C)`
/// - Hadamard product: `einsum("ij,ij->ij",A,B,C)`
/// - standard tensor contraction: `einsum("ijk,jkl->li",A,B,C)`
/// - standard tensor contraction: `einsum("ijk,jkl->li",A,B,C)`
template <typename TA, typename TB, typename TC>
void einsum(std::string expr,
            const Eigen::TensorBase<TA, Eigen::ReadOnlyAccessors> &A,
            const Eigen::TensorBase<TB, Eigen::ReadOnlyAccessors> &B, TC &C) {
  static_assert((TA::NumDimensions + TB::NumDimensions) >= TC::NumDimensions);

  using Index = TiledArray::Einsum::Index<char>;
  using IndexDims = TiledArray::Einsum::IndexMap<char, size_t>;
  using ::Einsum::string::split2;

  auto permutation = [](auto src, auto dst) {
    return TiledArray::Einsum::permutation(dst, src);
  };

  Index a, b, c;
  std::string ab;
  std::tie(ab, c) = split2(expr, "->");
  std::tie(a, b) = split2(ab, ",");

  // these are "Hadamard" (fused) indices
  auto h = a & b & c;

  auto e = (a ^ b);
  auto he = h + e;

  // contracted indices
  auto i = (a & b) - h;

  eigen_assert(a.size() == A.NumDimensions);
  eigen_assert(b.size() == B.NumDimensions);
  eigen_assert(c.size() == C.NumDimensions);
  eigen_assert(he.size() == C.NumDimensions);

  IndexDims dimensions = (IndexDims(a, derived(A).dimensions()) |
                          IndexDims(b, derived(B).dimensions()));

  auto product = [](auto &&dims) {
    int64_t n = 1;
    for (auto dim : dims) {
      n *= dim;
    }
    return n;
  };

  int64_t nh = product(dimensions[h]);
  int64_t na = product(dimensions[a & e]);
  int64_t nb = product(dimensions[b & e]);
  int64_t ni = product(dimensions[i]);

  auto pA = A.shuffle(permutation(a, h + (e & a) + i))
                .reshape(std::array{nh, na, ni})
                .eval();
  auto pB = B.shuffle(permutation(b, h + (e & b) + i))
                .reshape(std::array{nh, nb, ni})
                .eval();
  Eigen::Tensor<typename TC::Scalar, 3> C3(nh, na, nb);

  for (int64_t h = 0; h < nh; ++h) {
    // Eigen::array<Eigen::IndexPair<int>, 1> axis = { Eigen::IndexPair<int>(0,
    // 0) };
    C3.chip(h, 0) =
        pA.chip(h, 0).contract(pB.chip(h, 0), std::array{std::pair{1, 1}});
  }

  std::array<int, TC::NumDimensions> permuted_shape;
  for (int k = 0; k < permuted_shape.size(); ++k) {
    permuted_shape[k] = dimensions[he[k]];
  }

  C = C3.reshape(permuted_shape).shuffle(permutation(he, c));
}

template <typename T, typename TA, typename TB>
T einsum(std::string expr,
         const Eigen::TensorBase<TA, Eigen::ReadOnlyAccessors> &A,
         const Eigen::TensorBase<TB, Eigen::ReadOnlyAccessors> &B) {
  T AB;
  einsum(expr, A, B, AB);
  return AB;
}

}  // namespace Eigen

#endif /* TILEDARRAY_EINSUM_EIGEN_H__INCLUDED */
