/*
 * This file is a part of TiledArray.
 * Copyright (C) 2018  Virginia Tech
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

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <madness/config.h>
#include "cpu_cuda_vector.h"
#include "cuda_um_vector.h"

// assign 1 cuBLAS handle / thread, use thread-local storage to manage
class cuBLASHandlePool {
 public:
  static const cublasHandle_t &handle() {
    if (handle_ == 0) {
      handle_ = new cublasHandle_t;
      auto error = cublasCreate(handle_);
      assert(error == CUBLAS_STATUS_SUCCESS);
      error = cublasSetPointerMode(*handle_, CUBLAS_POINTER_MODE_HOST);
      assert(error == CUBLAS_STATUS_SUCCESS);
    }
    return *handle_;
  }

 private:
  static thread_local cublasHandle_t *handle_;
};
thread_local cublasHandle_t *cuBLASHandlePool::handle_;

// use multiple streams
int num_cuda_streams;  // user param; use a a nice prime for round-robins
std::vector<cudaStream_t> cuda_streams;  // pool of streams

// clang-format off
#include <btas/varray/varray.h>
#include <btas/tensor.h>
#include <btas/generic/permute.h>
#include <btas/generic/axpy_impl.h>

#include <tiledarray.h>
#include <TiledArray/permutation.h>
#include <TiledArray/range.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/external/btas.h>
// clang-format on

cublasOperation_t to_cublas_op(madness::cblas::CBLAS_TRANSPOSE cblas_op) {
  cublasOperation_t result;
  switch (cblas_op) {
    case madness::cblas::NoTrans:
      result = CUBLAS_OP_N;
      break;
    case madness::cblas::Trans:
      result = CUBLAS_OP_T;
      break;
    case madness::cblas::ConjTrans:
      result = CUBLAS_OP_C;
      break;
  }
  return result;
}

template <typename T>
cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const T *alpha, const T *A, int lda, const T *B,
                          int ldb, const T *beta, T *C, int ldc);
template <>
cublasStatus_t cublasGemm<float>(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int m, int n, int k,
                                 const float *alpha, const float *A, int lda,
                                 const float *B, int ldb, const float *beta,
                                 float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}
template <>
cublasStatus_t cublasGemm<double>(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const double *alpha, const double *A, int lda,
                                  const double *B, int ldb, const double *beta,
                                  double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

template <typename T>
cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, const T *alpha,
                          const T *x, int incx, T *y, int incy);
template <>
cublasStatus_t cublasAxpy<float>(cublasHandle_t handle, int n,
                                 const float *alpha, const float *x, int incx,
                                 float *y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
template <>
cublasStatus_t cublasAxpy<double>(cublasHandle_t handle, int n,
                                  const double *alpha, const double *x,
                                  int incx, double *y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <typename T>
cublasStatus_t cublasDot(cublasHandle_t handle, int n, const T *x, int incx,
                         const T *y, int incy, T *result);
template <>
cublasStatus_t cublasDot<float>(cublasHandle_t handle, int n, const float *x,
                                int incx, const float *y, int incy,
                                float *result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
template <>
cublasStatus_t cublasDot<double>(cublasHandle_t handle, int n, const double *x,
                                 int incx, const double *y, int incy,
                                 double *result) {
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

namespace TiledArray {

template <typename T>
void make_device_storage(cuda_um_vector<T> &storage, std::size_t n) {
  storage = cuda_um_vector<T>(n);
}

template <typename T>
void make_device_storage(cpu_cuda_vector<T> &storage, std::size_t n) {
  storage = cpu_cuda_vector<T>(n, cpu_cuda_vector<T>::state::device);
}

template <typename T>
T *device_data(cuda_um_vector<T> &storage) {
  return storage.data();
}

template <typename T>
T *device_data(cpu_cuda_vector<T> &storage) {
  return storage.device_data();
}

template <typename T>
const T *device_data(const cuda_um_vector<T> &storage) {
  return storage.data();
}

template <typename T>
const T *device_data(const cpu_cuda_vector<T> &storage) {
  return storage.device_data();
}

template <typename T, typename Range, typename Storage>
btas::Tensor<T, Range, Storage> gemm_cuda_impl(
    const btas::Tensor<T, Range, Storage> &left,
    const btas::Tensor<T, Range, Storage> &right, T factor,
    const TiledArray::math::GemmHelper &gemm_helper) {
  // either both arguments are on host or both on device ... mixed case TBI
  //  TA_ASSERT(left.storage().on_host() == right.storage().on_host() &&
  //            left.storage().on_device() == right.storage().on_device());

  TA_ASSERT((in_memory_space<MemorySpace::CPU>(left.storage()) &&
             in_memory_space<MemorySpace::CPU>(right.storage())) ||
            (in_memory_space<MemorySpace::CUDA>(left.storage()) &&
             in_memory_space<MemorySpace::CUDA>(right.storage())));

  // Check that the arguments are not empty and have the correct ranks
  TA_ASSERT(!left.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(!right.empty());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

  // Construct the result Tensor
  typedef btas::Tensor<T, Range, Storage> Tensor;
  typedef typename Tensor::storage_type storage_type;
  auto result_range =
      gemm_helper.make_result_range<Range>(left.range(), right.range());
  Storage result_storage;
  make_device_storage(result_storage, result_range.area());
  Tensor result(std::move(result_range), std::move(result_storage));

  // Check that the inner dimensions of left and right match
  TA_ASSERT(
      gemm_helper.left_right_congruent(std::cbegin(left.range().lobound()),
                                       std::cbegin(right.range().lobound())));
  TA_ASSERT(
      gemm_helper.left_right_congruent(std::cbegin(left.range().upbound()),
                                       std::cbegin(right.range().upbound())));
  TA_ASSERT(gemm_helper.left_right_congruent(
      std::cbegin(left.range().extent()), std::cbegin(right.range().extent())));

  // Compute gemm dimensions
  integer m = 1, n = 1, k = 1;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  // Get the leading dimension for left and right matrices.
  const integer lda =
      (gemm_helper.left_op() == madness::cblas::NoTrans ? k : m);
  const integer ldb =
      (gemm_helper.right_op() == madness::cblas::NoTrans ? n : k);

  if (in_memory_space<MemorySpace::CUDA>(result.storage())) {
    const auto &handle = cuBLASHandlePool::handle();
    auto zero = T(0);
    auto stream = result.range().ordinal().offset() % num_cuda_streams;
    auto status = cublasSetStream(handle, cuda_streams[stream]);
    assert(status == CUBLAS_STATUS_SUCCESS);
    status = cublasGemm(handle, to_cublas_op(gemm_helper.left_op()),
                        to_cublas_op(gemm_helper.right_op()), m, n, k, &factor,
                        device_data(left.storage()), lda,
                        device_data(right.storage()), ldb, &zero,
                        device_data(result.storage()), n);
    assert(status == CUBLAS_STATUS_SUCCESS);
  } else {
    assert(false);
    TiledArray::math::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n,
                           k, factor, left.data(), lda, right.data(), ldb, T(0),
                           result.data(), n);
  }

  return result;
}

template <typename T, typename Range, typename Storage>
void gemm_cuda_impl(btas::Tensor<T, Range, Storage> &result,
                    const btas::Tensor<T, Range, Storage> &left,
                    const btas::Tensor<T, Range, Storage> &right, T factor,
                    const TiledArray::math::GemmHelper &gemm_helper) {
  // the result determines were to do gemm
  if (in_memory_space<MemorySpace::CUDA>(result.storage())) {
    TA_ASSERT(in_memory_space<MemorySpace::CUDA>(left.storage()) &&
              in_memory_space<MemorySpace::CUDA>(right.storage()));
  } else {
    TA_ASSERT(in_memory_space<MemorySpace::CPU>(result.storage()) &&
              in_memory_space<MemorySpace::CPU>(left.storage()) &&
              in_memory_space<MemorySpace::CPU>(right.storage()));
  }

  // Check that the result is not empty and has the correct rank
  TA_ASSERT(!result.empty());
  TA_ASSERT(result.range().rank() == gemm_helper.result_rank());

  // Check that the arguments are not empty and have the correct ranks
  TA_ASSERT(!left.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(!right.empty());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

  // Check that the outer dimensions of left match the the corresponding
  // dimensions in result
  TA_ASSERT(
      gemm_helper.left_result_congruent(std::cbegin(left.range().lobound()),
                                        std::cbegin(result.range().lobound())));
  TA_ASSERT(
      gemm_helper.left_result_congruent(std::cbegin(left.range().upbound()),
                                        std::cbegin(result.range().upbound())));
  TA_ASSERT(
      gemm_helper.left_result_congruent(std::cbegin(left.range().extent()),
                                        std::cbegin(result.range().extent())));

  // Check that the outer dimensions of right match the the corresponding
  // dimensions in result
  TA_ASSERT(gemm_helper.right_result_congruent(
      std::cbegin(right.range().lobound()),
      std::cbegin(result.range().lobound())));
  TA_ASSERT(gemm_helper.right_result_congruent(
      std::cbegin(right.range().upbound()),
      std::cbegin(result.range().upbound())));
  TA_ASSERT(
      gemm_helper.right_result_congruent(std::cbegin(right.range().extent()),
                                         std::cbegin(result.range().extent())));

  // Check that the inner dimensions of left and right match
  TA_ASSERT(
      gemm_helper.left_right_congruent(std::cbegin(left.range().lobound()),
                                       std::cbegin(right.range().lobound())));
  TA_ASSERT(
      gemm_helper.left_right_congruent(std::cbegin(left.range().upbound()),
                                       std::cbegin(right.range().upbound())));
  TA_ASSERT(gemm_helper.left_right_congruent(
      std::cbegin(left.range().extent()), std::cbegin(right.range().extent())));

  // Compute gemm dimensions
  integer m, n, k;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  // Get the leading dimension for left and right matrices.
  const integer lda =
      (gemm_helper.left_op() == madness::cblas::NoTrans ? k : m);
  const integer ldb =
      (gemm_helper.right_op() == madness::cblas::NoTrans ? n : k);

  if (in_memory_space<MemorySpace::CUDA>(result.storage())) {
    const auto &handle = cuBLASHandlePool::handle();
    auto one = T(1);
    auto stream = result.range().ordinal().offset() % num_cuda_streams;
    auto status = cublasSetStream(handle, cuda_streams[stream]);
    assert(status == CUBLAS_STATUS_SUCCESS);
    status = cublasGemm(handle, to_cublas_op(gemm_helper.left_op()),
                        to_cublas_op(gemm_helper.right_op()), m, n, k, &factor,
                        device_data(left.storage()), lda,
                        device_data(right.storage()), ldb, &one,
                        device_data(result.storage()), n);
    assert(status == CUBLAS_STATUS_SUCCESS);
  } else {
    assert(false);
    TiledArray::math::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n,
                           k, factor, left.data(), lda, right.data(), ldb, T(1),
                           result.data(), n);
  }
}

template <typename T, typename Range>
btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> gemm(
    const btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> &left,
    const btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> &right,
    T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Range, typename AllocHost, typename AllocDevice>
btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>>
gemm(const btas::Tensor<T, Range,
                        TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>>
         &left,
     const btas::Tensor<T, Range,
                        TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>>
         &right,
     T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Range>
void gemm(btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> &result,
          const btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> &left,
          const btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> &right,
          T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

template <typename T, typename Range, typename AllocHost, typename AllocDevice>
void gemm(
    btas::Tensor<T, Range,
                 TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>>
        &result,
    const btas::Tensor<
        T, Range, TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>> &left,
    const btas::Tensor<T, Range,
                       TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>>
        &right,
    T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

/// result[i] += arg[i]
template <typename T, typename Range, typename Storage>
void add_to_cuda_impl(btas::Tensor<T, Range, Storage> &result,
                      const btas::Tensor<T, Range, Storage> &arg) {
  // the result determines were to do gemm
  if (in_memory_space<MemorySpace::CUDA>(result.storage())) {
    TA_ASSERT(in_memory_space<MemorySpace::CUDA>(arg.storage()));
  } else {
    TA_ASSERT(in_memory_space<MemorySpace::CPU>(result.storage()) &&
              in_memory_space<MemorySpace::CPU>(arg.storage()));
  }
  if (in_memory_space<MemorySpace::CUDA>(result.storage())) {
    const auto &handle = cuBLASHandlePool::handle();
    auto one = T(1);
    auto stream = result.range().ordinal().offset() % num_cuda_streams;
    auto status = cublasSetStream(handle, cuda_streams[stream]);
    assert(status == CUBLAS_STATUS_SUCCESS);
    status = cublasAxpy(handle, result.size(), &one, device_data(arg.storage()),
                        1, device_data(result.storage()), 1);
    assert(status == CUBLAS_STATUS_SUCCESS);
  } else {
    assert(false);
    btas::axpy(1.0, arg, result);
  }
}

template <typename T, typename Range, typename AllocHost, typename AllocDevice>
void add_to(
    btas::Tensor<T, Range,
                 TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>>
        &result,
    const btas::Tensor<T, Range,
                       TiledArray::cpu_cuda_vector<T, AllocHost, AllocDevice>>
        &arg) {
  add_to_cuda_impl(result, arg);
}

template <typename T, typename Range>
void add_to(btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> &result,
            const btas::Tensor<T, Range, TiledArray::cuda_um_vector<T>> &arg) {
  add_to_cuda_impl(result, arg);
}

// foreach(i) result += arg[i] * arg[i]
template <typename T, typename Storage>
typename btas::Tensor<T, btas::RangeNd<CblasRowMajor, std::array<short, 2>>,
                      Storage>::value_type
squared_norm_cuda_impl(
    const btas::Tensor<T, btas::RangeNd<CblasRowMajor, std::array<short, 2>>,
                       Storage> &arg) {
  auto storage = arg.storage();
  integer size = storage.size();
  T result = 0;
  if (in_memory_space<MemorySpace::CUDA>(storage)) {
    const auto &handle = cuBLASHandlePool::handle();
    auto status = cublasDot(handle, size, device_data(storage), 1,
                            device_data(storage), 1, &result);
    assert(status == CUBLAS_STATUS_SUCCESS);
  } else {
    result = TiledArray::math::dot(size, storage.data(), storage.data());
  }
  return result;
}

template <typename T>
typename btas::Tensor<T, btas::RangeNd<CblasRowMajor, std::array<short, 2>>,
                      TiledArray::cpu_cuda_vector<T>>::value_type
squared_norm(
    const btas::Tensor<T, btas::RangeNd<CblasRowMajor, std::array<short, 2>>,
                       TiledArray::cpu_cuda_vector<T>> &arg) {
  return squared_norm_cuda_impl(arg);
}

template <typename T>
typename btas::Tensor<T, btas::RangeNd<CblasRowMajor, std::array<short, 2>>,
                      TiledArray::cuda_um_vector<T>>::value_type
squared_norm(
    const btas::Tensor<T, btas::RangeNd<CblasRowMajor, std::array<short, 2>>,
                       TiledArray::cuda_um_vector<T>> &arg) {
  return squared_norm_cuda_impl(arg);
}

}  // namespace TiledArray

template <typename Storage>
void do_main_body(TiledArray::World &world, const long Nm, const long Bm,
                  const long Nn, const long Bn, const long Nk, const long Bk,
                  const long nrepeat) {
  using Real = typename Storage::value_type;

  const std::size_t Tm = Nm / Bm;
  const std::size_t Tn = Nn / Bn;
  const std::size_t Tk = Nk / Bk;

  if (world.rank() == 0)
    std::cout << "TiledArray: dense matrix multiply test...\n"
              << "Number of nodes     = " << world.size()
              << "\nSize of A         = " << Nm << "x" << Nk << " ("
              << double(Nm * Nk * sizeof(double)) / 1.0e9 << " GB)"
              << "\nSize of A block   = " << Bm << "x" << Bk
              << "\nSize of B         = " << Nk << "x" << Nn << " ("
              << double(Nk * Nn * sizeof(double)) / 1.0e9 << " GB)"
              << "\nSize of B block   = " << Bk << "x" << Bn
              << "\nSize of C         = " << Nm << "x" << Nn << " ("
              << double(Nm * Nn * sizeof(double)) / 1.0e9 << " GB)"
              << "\nSize of C block   = " << Bm << "x" << Bn
              << "\n# of blocks of C  = " << Tm * Tn
              << "\nAverage # of blocks of C/node = "
              << double(Tm * Tn) / double(world.size()) << "\n";

  // Construct TiledRange
  std::vector<unsigned int> blocking_m;
  blocking_m.reserve(Tm + 1);
  for (long i = 0l; i <= Nm; i += Bm) blocking_m.push_back(i);

  std::vector<unsigned int> blocking_n;
  blocking_n.reserve(Tn + 1);
  for (long i = 0l; i <= Nn; i += Bn) blocking_n.push_back(i);

  std::vector<unsigned int> blocking_k;
  blocking_k.reserve(Tk + 1);
  for (long i = 0l; i <= Nk; i += Bk) blocking_k.push_back(i);

  // Structure of c
  std::vector<TiledArray::TiledRange1> blocking_C;
  blocking_C.reserve(2);
  blocking_C.push_back(
      TiledArray::TiledRange1(blocking_m.begin(), blocking_m.end()));
  blocking_C.push_back(
      TiledArray::TiledRange1(blocking_n.begin(), blocking_n.end()));

  // Structure of a
  std::vector<TiledArray::TiledRange1> blocking_A;
  blocking_A.reserve(2);
  blocking_A.push_back(
      TiledArray::TiledRange1(blocking_m.begin(), blocking_m.end()));
  blocking_A.push_back(
      TiledArray::TiledRange1(blocking_k.begin(), blocking_k.end()));

  // Structure of b
  std::vector<TiledArray::TiledRange1> blocking_B;
  blocking_B.reserve(2);
  blocking_B.push_back(
      TiledArray::TiledRange1(blocking_k.begin(), blocking_k.end()));
  blocking_B.push_back(
      TiledArray::TiledRange1(blocking_n.begin(), blocking_n.end()));

  TiledArray::TiledRange  // TRange for c
      trange_c(blocking_C.begin(), blocking_C.end());

  TiledArray::TiledRange  // TRange for a
      trange_a(blocking_A.begin(), blocking_A.end());

  TiledArray::TiledRange  // TRange for b
      trange_b(blocking_B.begin(), blocking_B.end());

  using CUDATile =
      btas::Tensor<Real, btas::RangeNd<CblasRowMajor, std::array<short, 2>>,
                   Storage>;
  using CUDAMatrix = TA::DistArray<TA::Tile<CUDATile>>;

  // Construct and initialize arrays
  CUDAMatrix a(world, trange_a);
  CUDAMatrix b(world, trange_b);
  CUDAMatrix c(world, trange_c);
  a.fill(1.0);
  b.fill(1.0);
  cudaDeviceSynchronize();

  auto to_device = [](TA::Tile<CUDATile> &tile) -> void {
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        tile.tensor().storage());
  };
  auto to_host = [](TA::Tile<CUDATile> &tile) -> void {
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        tile.tensor().storage());
  };
  foreach_inplace(a, to_device);
  foreach_inplace(b, to_device);
  cudaDeviceSynchronize();

  // Start clock
  world.gop.fence();
  const double wall_time_start = madness::wall_time();

  // Do matrix multiplication
  for (int i = 0; i < nrepeat; ++i) {
    c("m,n") = a("m,k") * b("k,n");
    world.gop.fence();
    cudaDeviceSynchronize();
    if (world.rank() == 0) std::cout << "Iteration " << i + 1 << "\n";
  }

  // Stop clock
  const double wall_time_stop = madness::wall_time();

  if (world.rank() == 0)
    std::cout << "Average wall time   = "
              << (wall_time_stop - wall_time_start) / double(nrepeat)
              << " sec\nAverage GFLOPS      = "
              << double(nrepeat) * 2.0 * double(Nn * Nm * Nm) /
                     (wall_time_stop - wall_time_start) / 1.0e9
              << "\n";
}

int try_main(int argc, char **argv) {
  // Initialize runtime
  TiledArray::World &world = TiledArray::initialize(argc, argv);

  // Get command line arguments
  if (argc < 6) {
    std::cout << "multiplies A(Nm,Nk) * B(Nk,Nn), with dimensions m, n, and k "
                 "blocked by Bm, Bn, and Bk, respectively"
              << std::endl
              << "Usage: " << argv[0]
              << " Nm Bm Nn Bn Nk Bk [# of repetitions = 5] [# of CUDA streams "
                 "= 17] [real = double] [use CUDA UM? = false]\n";
    return 0;
  }
  const long Nm = atol(argv[1]);
  const long Bm = atol(argv[2]);
  const long Nn = atol(argv[3]);
  const long Bn = atol(argv[4]);
  const long Nk = atol(argv[5]);
  const long Bk = atol(argv[6]);
  if (Nm <= 0 || Nn <= 0 || Nk <= 0) {
    std::cerr << "Error: dimensions must be greater than zero.\n";
    return 1;
  }
  if (Bm <= 0 || Bn <= 0 || Bk <= 0) {
    std::cerr << "Error: block sizes must be greater than zero.\n";
    return 1;
  }
  if ((Nm % Bm) != 0ul || Nn % Bn != 0ul || Nk % Bk != 0ul) {
    std::cerr
        << "Error: diminsion size must be evenly divisible by block size.\n";
    return 1;
  }
  const long nrepeat = (argc >= 8 ? atol(argv[7]) : 5);
  if (nrepeat <= 0) {
    std::cerr << "Error: number of repetitions must be greater than zero.\n";
    return 1;
  }

  num_cuda_streams = (argc >= 9) ? atoi(argv[8]) : 17;

  const auto real_type_str =
      (argc >= 10) ? std::string(argv[9]) : std::string("double");

  const auto arg10_str = (argc >= 11) ? std::string(argv[10]) : std::string{};
  auto to_bool = [](const std::string &str) {
    return (str == "true" || str == "True" || str == "TRUE" || str == "1" ||
            str == "yes" || str == "Yes" || str == "YES");
  };
  const bool use_cuda_um = (argc >= 11) ? to_bool(arg10_str) : false;

  int driverVersion, runtimeVersion;
  auto error = cudaDriverGetVersion(&driverVersion);
  if (error != cudaSuccess) {
    std::cout << "error(cudaDriverGetVersion) = " << error << std::endl;
  }
  error = cudaRuntimeGetVersion(&runtimeVersion);
  if (error != cudaSuccess) {
    std::cout << "error(cudaRuntimeGetVersion) = " << error << std::endl;
  }
  std::cout << "CUDA {driver,runtime} versions = " << driverVersion << ","
            << runtimeVersion << std::endl;
  {
    size_t free_mem, total_mem;
    auto result = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "CUDA memory stats: {total,free} = {" << total_mem << ","
              << free_mem << "}" << std::endl;
  }
  {
    cuda_streams.resize(num_cuda_streams);
    for (auto &stream : cuda_streams) {
      auto error = cudaStreamCreate(&stream);
      assert(error == cudaSuccess);
    }
  }

  if (use_cuda_um) {
    if (real_type_str == "double")
      do_main_body<TiledArray::cuda_um_vector<double>>(world, Nm, Bm, Nn, Bn,
                                                       Nk, Bk, nrepeat);
    else
      do_main_body<TiledArray::cuda_um_vector<float>>(world, Nm, Bm, Nn, Bn, Nk,
                                                      Bk, nrepeat);
    //    throw std::runtime_error("support for CUDA UM is not yet
    //    implemented");
  } else {
    if (real_type_str == "double")
      do_main_body<TiledArray::cpu_cuda_vector<double>>(world, Nm, Bm, Nn, Bn,
                                                        Nk, Bk, nrepeat);
    else
      do_main_body<TiledArray::cpu_cuda_vector<float>>(world, Nm, Bm, Nn, Bn,
                                                       Nk, Bk, nrepeat);
  }

  TiledArray::finalize();

  {
    for (int s = 0; s != num_cuda_streams; ++s) {
      auto error = cudaStreamDestroy(cuda_streams[s]);
      assert(error == cudaSuccess);
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  try {
    try_main(argc, argv);
  } catch (thrust::system::detail::bad_alloc &ex) {
    std::cout << ex.what() << std::endl;

    size_t free_mem, total_mem;
    auto result = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "CUDA memory stats: {total,free} = {" << total_mem << ","
              << free_mem << "}" << std::endl;
  } catch (...) {
    std::cerr << "unknown exception" << std::endl;
  }

  return 0;
}
