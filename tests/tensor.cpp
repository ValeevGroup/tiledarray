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

#include <boost/range/combine.hpp>
#include <range/v3/view/zip.hpp>

#include <iterator>
#include "TiledArray/math/gemm_helper.h"
#include "TiledArray/tensor.h"
#include "tensor_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

BOOST_FIXTURE_TEST_SUITE(tensor_suite, TensorFixture, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(anatomy) {
  // Tensor = Range + nbatch + shared_ptr to data
  BOOST_CHECK(sizeof(TensorD) == sizeof(Range) + sizeof(size_t) +
                                     sizeof(std::shared_ptr<double[]>));
  // std::wcout << "sizeof(TensorD) = " << sizeof(TensorD) << " sizeof(TensorI)
  // = " << sizeof(TensorN) << std::endl;
}

BOOST_AUTO_TEST_CASE(default_constructor) {
  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x);
  TensorN x;

  BOOST_CHECK(x.empty());

  // Check that range data is correct
  BOOST_CHECK_EQUAL(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), 0ul);
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(x).range().volume(), 0ul);

  // Check the element data
  BOOST_CHECK_EQUAL(x.begin(), x.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(x).begin(),
                    const_cast<const TensorN&>(x).end());

  // check for element access error
  BOOST_CHECK_THROW(x[0], Exception);
}

BOOST_AUTO_TEST_CASE(range_constructor) {
  BOOST_REQUIRE_NO_THROW(TensorN x(r));
  TensorN x(r);

  BOOST_CHECK(!x.empty());

  // Check that range data is correct
  BOOST_CHECK_NE(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  // Do not check values of x because it maybe uninitialized
}

BOOST_AUTO_TEST_CASE(value_constructor) {
  BOOST_REQUIRE_NO_THROW(TensorN x(r, 8));
  TensorN x(r, 8);

  BOOST_CHECK(!x.empty());

  // Check that range data is correct
  BOOST_CHECK_NE(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (TensorN::const_iterator it = x.begin(); it != x.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 8);
}

BOOST_AUTO_TEST_CASE(iterator_copy_constructor) {
  BOOST_REQUIRE_NO_THROW(TensorN x(r, t.begin()));
  TensorN x(r, t.begin());

  BOOST_CHECK(!x.empty());

  // Check range data is correct
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i) BOOST_CHECK_EQUAL(x[i], t[i]);
}

BOOST_AUTO_TEST_CASE(copy_constructor) {
  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN tc(t));
  TensorN tc(t);

  BOOST_CHECK_EQUAL(tc.empty(), t.empty());

  BOOST_CHECK_EQUAL(tc.data(), t.data());  // N.B. shallow copy!
  BOOST_CHECK_EQUAL(tc.size(), t.size());
  BOOST_CHECK_EQUAL(tc.range(), t.range());
  BOOST_CHECK_EQUAL(tc.begin(), t.begin());
  BOOST_CHECK_EQUAL(tc.end(), t.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).begin(),
                    const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).end(),
                    const_cast<const TensorN&>(t).end());
  BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_CASE(move_constructor) {
  TensorN tc(t);
  TensorN td(std::move(tc));

  BOOST_CHECK_EQUAL(td.empty(), t.empty());

  // Check that range data is correct
  BOOST_CHECK_EQUAL(td.data(), t.data());
  BOOST_CHECK_EQUAL(td.size(), t.size());
  BOOST_CHECK_EQUAL(td.range(), t.range());
  BOOST_CHECK_EQUAL(td.begin(), t.begin());
  BOOST_CHECK_EQUAL(td.end(), t.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(td).begin(),
                    const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(td).end(),
                    const_cast<const TensorN&>(t).end());
  BOOST_CHECK_EQUAL_COLLECTIONS(td.begin(), td.end(), t.begin(), t.end());

  // check that moved-from object is empty
  BOOST_CHECK(tc.empty());
}

BOOST_AUTO_TEST_CASE(permute_constructor) {
  Permutation perm = make_perm();

  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(t, perm));
  TensorN x(t, perm);

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm * r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ordinal(perm * t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], t[i]);
  }
}

BOOST_AUTO_TEST_CASE(permute_constructor_tensor) {
  const std::array<std::size_t, 4> start = {{0ul, 0ul, 0ul, 0ul}};
  const std::array<std::size_t, 4> finish = {{2ul, 5ul, 7ul, 3ul}};
  TensorN x(range_type(start, finish));
  rand_fill(1693, x.size(), x.data());

  std::array<unsigned int, 4> p = {{0, 1, 2, 3}};

  while (std::next_permutation(p.begin(), p.end())) {
    Permutation perm(p.begin(), p.end());

    TensorN px;
    // check constructor
    BOOST_REQUIRE_NO_THROW(px = TensorN(x, perm));
    BOOST_CHECK(!px.empty());

    for (std::size_t i = 0ul; i < x.size(); ++i) {
      std::size_t pi = px.range().ordinal(perm * x.range().idx(i));
      BOOST_CHECK_EQUAL(px[pi], x[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(unary_constructor) {
  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(t, [](const int arg) { return arg * 83; }));
  TensorN x(t, [](const int arg) { return arg * 83; });

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], 83 * t[i]);
}

BOOST_AUTO_TEST_CASE(unary_permute_constructor) {
  Permutation perm = make_perm();

  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(
      t, [](const int arg) { return arg * 47; }, perm));
  TensorN x(
      t, [](const int arg) { return arg * 47; }, perm);

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm * r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ordinal(perm * t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], 47 * t[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_constructor) {
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check default constructor
  BOOST_REQUIRE_NO_THROW(
      TensorN x(t, s, [](const int l, const int r) { return l - r; }));
  TensorN x(t, s, [](const int l, const int r) { return l - r; });

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], t[i] - s[i]);
}

BOOST_AUTO_TEST_CASE(binary_perm_constructor) {
  Permutation perm = make_perm();
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check default constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(
      t, s, [](const int l, const int r) { return l - r; }, perm));
  TensorN x(
      t, s, [](const int l, const int r) { return l - r; }, perm);

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm * r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ordinal(perm * t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], t[i] - s[i]);
  }
}

BOOST_AUTO_TEST_CASE(clone) {
  // clone non-default-constructed
  TensorN tc;
  BOOST_CHECK(tc.empty());
  BOOST_REQUIRE_NO_THROW(tc = t.clone());
  BOOST_CHECK_EQUAL(tc.empty(), t.empty());
  BOOST_CHECK_NE(tc.data(), t.data());
  BOOST_CHECK_EQUAL(tc.size(), t.size());
  BOOST_CHECK_EQUAL(tc.range(), t.range());
  BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());

  // clone default-constructed tensor
  {
    TensorN tnull;
    BOOST_REQUIRE_NO_THROW(tc = tnull.clone());
    BOOST_CHECK_EQUAL(tc.empty(), tnull.empty());
  }

  // clone rvalue (e.g. temporary) tensor = move
  {
    TensorN t2 = t.clone();
    const auto t2_data = t2.data();
    BOOST_REQUIRE_NO_THROW(tc = std::move(t2).clone());
    BOOST_CHECK(t2.empty());  // t2 is moved-from state
    BOOST_CHECK(!tc.empty());
    BOOST_CHECK_NE(tc.data(), t.data());
    BOOST_CHECK_EQUAL(tc.data(), t2_data);
    BOOST_CHECK_EQUAL(tc.size(), t.size());
    BOOST_CHECK_EQUAL(tc.range(), t.range());
    BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
  }
}

BOOST_AUTO_TEST_CASE(copy_assignment_operator) {
  TensorN tc;
  tc = t;

  BOOST_CHECK_EQUAL(tc.empty(), t.empty());

  BOOST_CHECK_EQUAL(tc.data(), t.data());  // N.B. shallow copy!
  BOOST_CHECK_EQUAL(tc.size(), t.size());
  BOOST_CHECK_EQUAL(tc.range(), t.range());
  BOOST_CHECK_EQUAL(tc.begin(), t.begin());
  BOOST_CHECK_EQUAL(tc.end(), t.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).begin(),
                    const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).end(),
                    const_cast<const TensorN&>(t).end());
  BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_CASE(move_assignment_operator) {
  TensorN td(t);
  TensorN tc;
  tc = std::move(td);

  BOOST_CHECK_EQUAL(tc.empty(), t.empty());

  // Check that range data is correct
  BOOST_CHECK_EQUAL(tc.data(), t.data());
  BOOST_CHECK_EQUAL(tc.size(), t.size());
  BOOST_CHECK_EQUAL(tc.range(), t.range());
  BOOST_CHECK_EQUAL(tc.begin(), t.begin());
  BOOST_CHECK_EQUAL(tc.end(), t.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).begin(),
                    const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).end(),
                    const_cast<const TensorN&>(t).end());
  BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
  // moved-from object is empty
  BOOST_CHECK(td.empty());
}

BOOST_AUTO_TEST_CASE(range_accessor) {
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().lobound_data(), t.range().lobound_data() + t.range().rank(),
      r.lobound_data(), r.lobound_data() + r.rank());  // check start accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().upbound_data(), t.range().upbound_data() + t.range().rank(),
      r.upbound_data(), r.upbound_data() + r.rank());  // check finish accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().extent_data(), t.range().extent_data() + t.range().rank(),
      r.extent_data(), r.extent_data() + r.rank());  // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().stride_data(), t.range().stride_data() + t.range().rank(),
      r.stride_data(), r.stride_data() + r.rank());   // check weight accessor
  BOOST_CHECK_EQUAL(t.range().volume(), r.volume());  // check volume accessor
  BOOST_CHECK_EQUAL(t.range(), r);                    // check range accessof
}

BOOST_AUTO_TEST_CASE(element_access) {
  // check operator[] with array coordinate index and ordinal index
  for (std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_LT(t[i], 42);
    BOOST_CHECK_EQUAL(t[r.idx(i)], t[i]);
  }

  // check access via call operator
  for (std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_LT(t(i), 42);
    BOOST_CHECK_EQUAL(t(r.idx(i)), t[i]);
    BOOST_CHECK_EQUAL(t(i), t[i]);
  }
#if TEST_DIM == 3u
  BOOST_CHECK_EQUAL(t(r.lobound(0), r.lobound(1), r.lobound(2)), t[0]);
  BOOST_CHECK_EQUAL(t({r.lobound(0), r.lobound(1), r.lobound(2)}), t[0]);
#endif

  // check out of range error
  BOOST_CHECK_THROW(t[r.upbound()], Exception);
  BOOST_CHECK_THROW(t[r.volume()], Exception);
}

BOOST_AUTO_TEST_CASE(iteration) {
  BOOST_CHECK_EQUAL(t.begin(), const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(t.end(), const_cast<const TensorN&>(t).end());

  for (TensorN::iterator it = t.begin(); it != t.end(); ++it) {
    BOOST_CHECK_LT(*it, 42);
    BOOST_CHECK_EQUAL(*it, t[std::distance(t.begin(), it)]);
  }

  // check iterator assignment
  TensorN::iterator it = t.begin();
  BOOST_CHECK_NE(t[0], 88);
  *it = 88;
  BOOST_CHECK_EQUAL(t[0], 88);

  // Check that the iterators of an empty tensor are equal
  TensorN t2;
  BOOST_CHECK_EQUAL(t2.begin(), t2.end());
}

BOOST_AUTO_TEST_CASE(element_assignment) {
  // verify preassignment conditions
  BOOST_CHECK_NE(t[1], 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t[1] = 2, 2);
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t[1], 2);
}

BOOST_AUTO_TEST_CASE(serialization) {
  std::size_t buf_size = (t.range().volume() * sizeof(int) +
                          sizeof(size_type) * (r.rank() * 4 + 2)) *
                         2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  BOOST_REQUIRE_NO_THROW(oar & t);
  std::size_t nbyte = oar.size();
  oar.close();

  TensorN ts;
  madness::archive::BufferInputArchive iar(buf, nbyte);
  BOOST_REQUIRE_NO_THROW(iar & ts);
  iar.close();

  delete[] buf;

  BOOST_CHECK_EQUAL(t.range(), ts.range());
  BOOST_CHECK_EQUAL_COLLECTIONS(t.begin(), t.end(), ts.begin(), ts.end());
}

BOOST_AUTO_TEST_CASE(swap) {
  TensorN s = make_tensor(79, 1559);
  rand_fill(431, s.size(), s.data());

  // Store a copy of the current state.
  range_type t_range = t.range();
  const int* const t_data = t.data();
  range_type s_range = s.range();
  const int* const s_data = s.data();

  BOOST_REQUIRE_NO_THROW(t.swap(s));

  // Check that the data has been moved correctly
  BOOST_CHECK_EQUAL(t.range(), s_range);
  BOOST_CHECK_EQUAL(t.data(), s_data);
  BOOST_CHECK_EQUAL(s.range(), t_range);
  BOOST_CHECK_EQUAL(s.data(), t_data);
}

BOOST_AUTO_TEST_CASE(unary_op) {
  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(x = t.unary([](const int arg) { return arg * 83; }));

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_EQUAL(x.range(), r);

  // Check that the data pointers are correct
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  // Check that the element values are correct
  for (std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], 83 * t[i]);
}

BOOST_AUTO_TEST_CASE(unary_permute_op) {
  Permutation perm = make_perm();

  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(
      x = t.unary([](const int arg) { return arg * 47; }, perm));

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm * r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ordinal(perm * t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], 47 * t[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_op) {
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(
      x = t.binary(s, [](const int l, const int r) { return l - r; }));

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], t[i] - s[i]);
}

BOOST_AUTO_TEST_CASE(binary_perm_op) {
  Permutation perm = make_perm();
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check default constructor
  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(
      x = t.binary(
          s, [](const int l, const int r) { return l - r; }, perm));

  BOOST_CHECK(!x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm * r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ordinal(perm * t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], t[i] - s[i]);
  }
}

BOOST_AUTO_TEST_CASE(gemm) {
  using integer = TiledArray::math::blas::integer;
  TensorD x(r);
  rand_fill(431, x.size(), x.data());
  TensorD y(r);
  rand_fill(413, y.size(), y.data());

  const auto ndim_contr =
      r.rank() % 2;  // this many trailing modes will be contracted
  const auto ndim_free =
      r.rank() - ndim_contr;  // this many leading modes will be free
  const auto alpha = 1.5;
  const auto gemm_helper_nt = math::GemmHelper(
      TiledArray::math::blas::Op::NoTrans, TiledArray::math::blas::Op::Trans,
      2 * ndim_free, x.range().rank(), y.range().rank());

  // check result-returning gemm
  TensorD z0;
  {
    BOOST_REQUIRE_NO_THROW(z0 = x.gemm(y, alpha, gemm_helper_nt));

    BOOST_CHECK(!z0.empty());

    // Check that range data is correct.
    auto z0_range_ref =
        gemm_helper_nt.make_result_range<Range>(x.range(), y.range());
    BOOST_CHECK_EQUAL(z0.range(), z0_range_ref);

    // verify data
    std::vector<double> z0_ref(z0.range().volume());
    {
      integer m = 1, n = 1, k = 1;
      gemm_helper_nt.compute_matrix_sizes(m, n, k, x.range(), y.range());
      math::blas::gemm(TiledArray::math::blas::Op::NoTrans,
                       TiledArray::math::blas::Op::Trans, m, n, k, alpha,
                       x.data(), k, y.data(), k, 0, z0_ref.data(), n);
    }
    for (std::size_t i = 0ul; i < z0.size(); ++i)
      BOOST_CHECK_EQUAL(z0[i], z0_ref[i]);
  }

  // check in-place gemm
  {
    // can use uninitialized ...
    TensorD z1;
    BOOST_REQUIRE_NO_THROW(z1.gemm(x, y, alpha, gemm_helper_nt));
    for (std::size_t i = 0ul; i < z0.size(); ++i)
      BOOST_CHECK_EQUAL(z0[i], z1[i]);

    // .. and uninitialized tensor ..
    const double z2_init = -1.3;
    TensorD z2(z0.range(), z2_init);
    BOOST_REQUIRE_NO_THROW(z2.gemm(x, y, alpha, gemm_helper_nt));
    for (std::size_t i = 0ul; i < z0.size(); ++i)
      BOOST_CHECK_EQUAL(z0[i], z2[i] - z2_init);

    // .. and even drop in custom element multiply-add op
    const double z3_init = 0.00;
    TensorD z3(z0.range(), z3_init);
    BOOST_REQUIRE_NO_THROW(
        z3.gemm(x, y, gemm_helper_nt,
                [alpha](auto& result, const auto& left, const auto& right) {
                  result += alpha * (left * right);
                }));
    for (std::size_t i = 0ul; i < z0.size(); ++i)
      BOOST_CHECK_EQUAL(z0[i], z3[i] - z3_init);
  }
}

BOOST_AUTO_TEST_CASE(conj_op) {
  Permutation perm = make_perm();
  TensorZ s(r);
  rand_fill(431, s.size(), s.data());

  TensorZ t;
  BOOST_REQUIRE_NO_THROW(t = s.conj());

  BOOST_CHECK_EQUAL(t.range(), s.range());

  for (std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_EQUAL(t[i].real(), s[i].real());
    BOOST_CHECK_EQUAL(t[i].imag(), -s[i].imag());
  }
}

BOOST_AUTO_TEST_CASE(conj_scal_op) {
  Permutation perm = make_perm();
  TensorZ s(r);
  rand_fill(431, s.size(), s.data());

  TensorZ t;
  BOOST_REQUIRE_NO_THROW(t = s.conj(3.0));

  BOOST_CHECK_EQUAL(t.range(), s.range());

  for (std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_EQUAL(t[i].real(), 3.0 * s[i].real());
    BOOST_CHECK_EQUAL(t[i].imag(), -3.0 * s[i].imag());
  }
}

BOOST_AUTO_TEST_CASE(inplace_conj_op) {
  Permutation perm = make_perm();
  TensorZ s(r);
  rand_fill(431, s.size(), s.data());

  TensorZ t = s.clone();
  BOOST_REQUIRE_NO_THROW(t.conj_to());

  BOOST_CHECK_EQUAL(t.range(), s.range());

  for (std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_EQUAL(t[i].real(), s[i].real());
    BOOST_CHECK_EQUAL(t[i].imag(), -s[i].imag());
  }
}

BOOST_AUTO_TEST_CASE(inplace_conj_scal_op) {
  Permutation perm = make_perm();
  TensorZ s(r);
  rand_fill(431, s.size(), s.data());

  TensorZ t = s.clone();
  BOOST_REQUIRE_NO_THROW(t.conj_to(3.0));

  BOOST_CHECK_EQUAL(t.range(), s.range());

  for (std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_EQUAL(t[i].real(), 3.0 * s[i].real());
    BOOST_CHECK_EQUAL(t[i].imag(), -3.0 * s[i].imag());
  }
}

BOOST_AUTO_TEST_CASE(block) {
  TensorZ s(r);
  auto lobound = r.lobound();
  auto upbound = r.upbound();
  BOOST_REQUIRE_NO_THROW(s.block(lobound, upbound));
#if TEST_DIM == 3u
  BOOST_REQUIRE_NO_THROW(s.block({{lobound[0], upbound[0]},
                                  {lobound[1], upbound[1]},
                                  {lobound[2], upbound[2]}}));
  BOOST_REQUIRE_NO_THROW(s.block({lobound[0], lobound[1], lobound[2]},
                                 {upbound[0], upbound[1], upbound[2]}));
#endif

  // using zipped ranges of bounds (using Boost.Range)
  // need to #include <boost/range/combine.hpp>
  BOOST_CHECK_NO_THROW(s.block(boost::combine(lobound, upbound)));

  BOOST_CHECK_NO_THROW(s.block(ranges::views::zip(lobound, upbound)));

  auto sview0 = s.block(lobound, upbound);
  BOOST_CHECK(sview0.range().includes(lobound));
  BOOST_CHECK(sview0(lobound) == s(lobound));
#if TEST_DIM == 3u
  auto sview1 = s.block({lobound[0], lobound[1], lobound[2]},
                        {upbound[0], upbound[1], upbound[2]});
  BOOST_CHECK(sview1.range().includes(lobound));
  BOOST_CHECK(sview1(lobound) == s(lobound));
#endif
}

BOOST_AUTO_TEST_CASE(allocator) {
  TensorD x(r, 1.0);
  Tensor<double, std::allocator<double>> y(r, 1.0);
  static_assert(std::is_same_v<decltype(x.add(y)), TensorD>);
  static_assert(std::is_same_v<decltype(y.add(x)), decltype(y)>);
  static_assert(std::is_same_v<decltype(x.subt(y)), TensorD>);
  static_assert(std::is_same_v<decltype(y.subt(x)), decltype(y)>);
  static_assert(std::is_same_v<decltype(x.mult(y)), TensorD>);
  static_assert(std::is_same_v<decltype(y.mult(x)), decltype(y)>);
  BOOST_REQUIRE_NO_THROW(x.add_to(y));
  BOOST_REQUIRE_NO_THROW(x.subt_to(y));
  BOOST_REQUIRE_NO_THROW(x.mult_to(y));
}

BOOST_AUTO_TEST_CASE(rebind) {
  static_assert(
      std::is_same_v<TensorD::rebind_t<std::complex<double>>, TensorZ>);
  static_assert(
      std::is_same_v<TensorD::rebind_numeric_t<std::complex<double>>, TensorZ>);
  static_assert(
      std::is_same_v<TiledArray::detail::complex_t<TensorD>, TensorZ>);
  static_assert(std::is_same_v<TiledArray::detail::real_t<TensorZ>, TensorD>);
}

BOOST_AUTO_TEST_CASE(print) {
  std::ostringstream oss;
  std::wostringstream woss;
  BOOST_REQUIRE_NO_THROW(oss << t);
  BOOST_REQUIRE_NO_THROW(woss << t);
  // std::cout << t;
  decltype(t) tb(t.range(), decltype(t)::nbatches{2});
  rand_fill(1, tb.total_size(), tb.data());
  BOOST_REQUIRE_NO_THROW(oss << tb);
  BOOST_REQUIRE_NO_THROW(woss << tb);
  // std::cout << tb;
}

BOOST_AUTO_TEST_CASE(size_of) {
  auto sz0h = TiledArray::size_of<TiledArray::MemorySpace::Host>(TensorN{});
  BOOST_REQUIRE(sz0h == sizeof(TensorN));

  auto sz0d = TiledArray::size_of<TiledArray::MemorySpace::Device>(TensorN{});
  BOOST_REQUIRE(sz0d == 0);

  auto sz0um =
      TiledArray::size_of<TiledArray::MemorySpace::Device_UM>(TensorN{});
  BOOST_REQUIRE(sz0um == 0);

  auto sz1 = TiledArray::size_of<TiledArray::MemorySpace::Host>(
      TensorZ(Range(2, 3, 4)));
  BOOST_REQUIRE(sz1 ==
                sizeof(TensorZ) + 2 * 3 * 4 * sizeof(TensorZ::value_type));

  using TTD = Tensor<Tensor<double>>;
  auto sz2 =
      TiledArray::size_of<TiledArray::MemorySpace::Host>(TTD(Range(2, 3, 4)));
  BOOST_REQUIRE(sz2 == sizeof(TTD) + 2 * 3 * 4 * sizeof(TTD::value_type));

  TTD ttd(Range(2, 3, 4));
  ttd(0, 0, 0) = TensorD(Range(5, 6));
  auto sz3 = TiledArray::size_of<TiledArray::MemorySpace::Host>(ttd);
  BOOST_REQUIRE(sz3 == sizeof(TTD) + 2 * 3 * 4 * sizeof(TTD::value_type) +
                           5 * 6 * sizeof(TTD::value_type::value_type));
}

BOOST_AUTO_TEST_SUITE_END()
