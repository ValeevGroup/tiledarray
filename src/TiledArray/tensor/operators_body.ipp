// Shared body of tensor arithmetic operators.
//
// NO header guard, NO namespace — this file is *intentionally* #included from
// inside two different namespaces (TiledArray and btas) to make ADL find the
// right overload for each tile type. Before including, the enclosing namespace
// must define:
//
//   namespace detail {
//     template <typename T>
//     inline constexpr bool ta_ops_match_tensor_v = …;
//   }
//
// returning true for the tensor types this namespace's operators should accept
// and false otherwise — that's how we keep the two copies non-overlapping in
// overload resolution (TA's copy accepts TiledArray::Tensor &c.; btas's copy
// accepts btas::Tensor). The two predicates must be *disjoint*; in particular
// TA's copy must not accept btas::Tensor and vice versa. References to
// `::TA::detail::remove_cvr_t` etc. work from either namespace via the
// `namespace TA = TiledArray` alias at global scope.
//
// Only operators whose body delegates to a free CPO available for both tile
// types belong here. Operators that delegate to member functions (e.g.
// `tensor + number` → `tensor.add(number)`) or to TA-only free CPOs (e.g.
// `add_to(Tensor, scalar)`) live directly in <TiledArray/tensor/operators.h>
// inside namespace TiledArray.

/// element-wise tensor + tensor
template <typename T1, typename T2,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator+(T1&& left, T2&& right) {
  return add(std::forward<T1>(left), std::forward<T2>(right));
}

/// element-wise tensor - tensor
template <typename T1, typename T2,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator-(T1&& left, T2&& right) {
  return subt(std::forward<T1>(left), std::forward<T2>(right));
}

/// element-wise tensor * tensor
template <typename T1, typename T2,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator*(T1&& left, T2&& right) {
  return mult(std::forward<T1>(left), std::forward<T2>(right));
}

/// unary negation
template <typename T,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T>>>>
inline decltype(auto) operator-(T&& arg) {
  return neg(std::forward<T>(arg));
}

/// tensor * scalar (right-hand scalar)
template <typename T, typename N,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T>> &&
              TA::detail::is_numeric_v<N>>>
inline decltype(auto) operator*(T&& tensor, N number) {
  return scale(std::forward<T>(tensor), number);
}

/// scalar * tensor (left-hand scalar)
template <typename N, typename T,
          typename = std::enable_if_t<
              TA::detail::is_numeric_v<N> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T>>>>
inline decltype(auto) operator*(N number, T&& tensor) {
  return scale(std::forward<T>(tensor), number);
}

/// tensor += tensor
template <typename T1, typename T2,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator+=(T1&& left, const T2& right) {
  return add_to(std::forward<T1>(left), right);
}

/// tensor -= tensor
template <typename T1, typename T2,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator-=(T1&& left, const T2& right) {
  return subt_to(std::forward<T1>(left), right);
}

/// tensor *= tensor (element-wise)
template <typename T1, typename T2,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator*=(T1&& left, const T2& right) {
  return mult_to(std::forward<T1>(left), right);
}

/// tensor *= scalar
template <typename T, typename N,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T>> &&
              TA::detail::is_numeric_v<N>>>
inline decltype(auto) operator*=(T&& left, N right) {
  return scale_to(std::forward<T>(left), right);
}

/// permutation * tensor (templated on the permutation type so this overload
/// is usable with any class that satisfies @c TA::detail::is_permutation_v ,
/// not only @c TiledArray::Permutation )
template <typename P, typename T,
          typename = std::enable_if_t<
              TA::detail::is_permutation_v<TA::detail::remove_cvr_t<P>> &&
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T>>>>
inline decltype(auto) operator*(const P& perm, T&& arg) {
  return permute(std::forward<T>(arg), perm);
}

/// Tensor output operator — NumPy-style printing for any contiguous tensor
/// type whose range exposes the standard accessors (@c rank , @c extent_data ,
/// @c stride_data , @c ordinal ). Element-of-tensor (ToT) decoration is
/// emitted when the element type is itself a tensor; the optional @c nbatch
/// member is queried via @c if constexpr so non-batched tensors compile too.
template <typename Char, typename CharTraits, typename T,
          typename = std::enable_if_t<
              detail::ta_ops_match_tensor_v<TA::detail::remove_cvr_t<T>> &&
              TA::detail::is_contiguous_tensor_v<TA::detail::remove_cvr_t<T>>>>
inline std::basic_ostream<Char, CharTraits>& operator<<(
    std::basic_ostream<Char, CharTraits>& os, const T& t) {
  os << t.range() << " {\n";
  const auto n = t.range().volume();
  std::size_t offset = 0ul;
  std::size_t nbatch = 1;
  if constexpr (TA::detail::has_member_function_nbatch_anyreturn_v<T>)
    nbatch = t.nbatch();
  const auto more_than_1_batch = nbatch > 1;
  for (auto b = 0ul; b != nbatch; ++b) {
    if (more_than_1_batch) {
      os << "  [batch " << b << "]{\n";
    }
    if constexpr (TA::detail::is_tensor_v<T>) {  // tensor of scalars
      TA::detail::NDArrayPrinter{}.print(
          t.data() + offset, t.range().rank(), t.range().extent_data(),
          t.range().stride_data(), os, more_than_1_batch ? 4 : 2);
    } else {  // tensor of tensors — annotate each element by its index
      for (auto&& idx : t.range()) {
        const auto& inner_t = *(t.data() + offset + t.range().ordinal(idx));
        os << "  " << idx << ":";
        using inner_range_t =
            std::remove_cv_t<std::remove_reference_t<decltype(inner_t.range())>>;
        if constexpr (TA::detail::has_member_function_stride_data_anyreturn_v<
                          inner_range_t>) {
          TA::detail::NDArrayPrinter{}.print(
              inner_t.data(), inner_t.range().rank(),
              inner_t.range().extent_data(), inner_t.range().stride_data(), os,
              more_than_1_batch ? 6 : 4);
        } else {
          // Inner range doesn't expose stride_data (e.g. btas::zb::RangeNd,
          // which intentionally synthesizes row-major strides on demand and
          // stores none). Skip the strided pretty-printer for this element.
          os << " <inner tile elided: range type has no stride_data()>";
        }
        os << "\n";
      }
    }
    if (more_than_1_batch) {
      os << "\n  }";
      if (b + 1 != nbatch) os << "\n";
    }
    offset += n;
  }
  os << "\n}\n";
  return os;
}
