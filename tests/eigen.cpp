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

#include "TiledArray/conversions/eigen.h"
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct EigenFixture : public TiledRangeFixture {
  EigenFixture()
      : trange(dims.begin(), dims.begin() + 2),
        trange1(dims.begin(), dims.begin() + 1),
        trangeN(dims.begin(), dims.begin() + GlobalFixture::dim),
        array(*GlobalFixture::world, trange),
        array1(*GlobalFixture::world, trange1),
        arrayN(*GlobalFixture::world, trangeN),
        matrix(dims[0].elements_range().second,
               dims[1].elements_range().second),
        rmatrix(dims[0].elements_range().second,
                dims[1].elements_range().second),
        vector(dims[0].elements_range().second),
        tensor(extents),
        rtensor(extents) {}

  TiledRange trange;
  TiledRange trange1;
  TiledRange trangeN;
  TArrayI array;
  TArrayI array1;
  TArrayI arrayN;
  Eigen::MatrixXi matrix;
  EigenMatrixXi rmatrix;
  Eigen::VectorXi vector;
  Eigen::Tensor<int, GlobalFixture::dim> tensor;
  Eigen::Tensor<int, GlobalFixture::dim, Eigen::RowMajor> rtensor;
};

BOOST_FIXTURE_TEST_SUITE(eigen_suite, EigenFixture)

BOOST_AUTO_TEST_CASE(tile_map) {
  // Make a tile with random data
  Tensor<int> tensor(trange.make_tile_range(0));
  const Tensor<int>& ctensor = tensor;
  GlobalFixture::world->srand(27);
  for (Tensor<int>::iterator it = tensor.begin(); it != tensor.end(); ++it)
    *it = GlobalFixture::world->rand();

  Eigen::Map<EigenMatrixXi> map =
      eigen_map(tensor, tensor.range().extent(0), tensor.range().extent(1));

  // Check the map dimensions
  BOOST_CHECK_EQUAL(map.rows(), tensor.range().extent(0));
  BOOST_CHECK_EQUAL(map.cols(), tensor.range().extent(1));

  for (Range::const_iterator it = tensor.range().begin();
       it != tensor.range().end(); ++it) {
    BOOST_CHECK_EQUAL(map((*it)[0], (*it)[1]), tensor[*it]);
  }

  Eigen::Map<const EigenMatrixXi> cmap =
      eigen_map(ctensor, ctensor.range().extent(0), ctensor.range().extent(1));

  // Check the map dimensions
  BOOST_CHECK_EQUAL(cmap.rows(), ctensor.range().extent(0));
  BOOST_CHECK_EQUAL(cmap.cols(), ctensor.range().extent(1));

  for (Range::const_iterator it = tensor.range().begin();
       it != tensor.range().end(); ++it) {
    BOOST_CHECK_EQUAL(cmap((*it)[0], (*it)[1]), ctensor[*it]);
  }
}

BOOST_AUTO_TEST_CASE(auto_tile_map) {
  // Make a tile with random data
  Tensor<int> tensor(trange.make_tile_range(0));
  const Tensor<int>& ctensor = tensor;
  GlobalFixture::world->srand(27);
  for (Tensor<int>::iterator it = tensor.begin(); it != tensor.end(); ++it)
    *it = GlobalFixture::world->rand();

  Eigen::Map<EigenMatrixXi> map = eigen_map(tensor);

  // Check the map dimensions
  BOOST_CHECK_EQUAL(map.rows(), tensor.range().extent(0));
  BOOST_CHECK_EQUAL(map.cols(), tensor.range().extent(1));

  for (Range::const_iterator it = tensor.range().begin();
       it != tensor.range().end(); ++it) {
    BOOST_CHECK_EQUAL(map((*it)[0], (*it)[1]), tensor[*it]);
  }

  Eigen::Map<const EigenMatrixXi> cmap = eigen_map(ctensor);

  // Check the map dimensions
  BOOST_CHECK_EQUAL(cmap.rows(), ctensor.range().extent(0));
  BOOST_CHECK_EQUAL(cmap.cols(), ctensor.range().extent(1));

  for (Range::const_iterator it = tensor.range().begin();
       it != tensor.range().end(); ++it) {
    BOOST_CHECK_EQUAL(cmap((*it)[0], (*it)[1]), ctensor[*it]);
  }
}

BOOST_AUTO_TEST_CASE(submatrix_to_tensor) {
  // Fill the matrix with random data
  matrix = decltype(matrix)::Random(matrix.rows(), matrix.cols());
  // Make a target tensor
  Tensor<int> tensor(trange.make_tile_range(0));

  // Copy the sub matrix to the tensor objects
  BOOST_CHECK_NO_THROW(eigen_submatrix_to_tensor(matrix, tensor));

  // Get the target submatrix block
  auto block =
      matrix.block(tensor.range().lobound(0), tensor.range().lobound(1),
                   tensor.range().extent(0), tensor.range().extent(1));

  // Check that the block contains the same values as the tensor
  for (Range::const_iterator it = tensor.range().begin();
       it != tensor.range().end(); ++it) {
    BOOST_CHECK_EQUAL(tensor[*it], block((*it)[0], (*it)[1]));
  }
}

BOOST_AUTO_TEST_CASE(tensor_to_submatrix) {
  // Fill a tensor with data
  Tensor<int> tensor(trange.make_tile_range(0));
  GlobalFixture::world->srand(27);
  for (Tensor<int>::iterator it = tensor.begin(); it != tensor.end(); ++it)
    *it = GlobalFixture::world->rand();

  // Copy the tensor to the submatrix block
  BOOST_CHECK_NO_THROW(tensor_to_eigen_submatrix(tensor, matrix));

  // Get the source submatrix block
  auto block =
      matrix.block(tensor.range().lobound(0), tensor.range().lobound(1),
                   tensor.range().extent(0), tensor.range().extent(1));

  // Check that the block contains the same values as the tensor
  for (Range::const_iterator it = tensor.range().begin();
       it != tensor.range().end(); ++it) {
    BOOST_CHECK_EQUAL(block((*it)[0], (*it)[1]), tensor[*it]);
  }
}

BOOST_AUTO_TEST_CASE(matrix_to_array) {
  // Fill the matrix with random data and replicate across the world
  matrix = decltype(matrix)::Random(matrix.rows(), matrix.cols());
  GlobalFixture::world->gop.broadcast_serializable(matrix, 0);

  // Copy matrix to array
  BOOST_CHECK_NO_THROW(
      (array = eigen_to_array<TArrayI>(*GlobalFixture::world, trange, matrix)));

  // Check that the data in array is equal to that in matrix
  for (Range::const_iterator it = array.tiles_range().begin();
       it != array.tiles_range().end(); ++it) {
    Future<TArrayI::value_type> tile = array.find(*it);
    for (Range::const_iterator tile_it = tile.get().range().begin();
         tile_it != tile.get().range().end(); ++tile_it) {
      BOOST_CHECK_EQUAL(tile.get()[*tile_it],
                        matrix((*tile_it)[0], (*tile_it)[1]));
    }
  }
}

BOOST_AUTO_TEST_CASE(vector_to_array) {
  // Fill the vector with random data and replicate across the world
  vector = Eigen::VectorXi::Random(vector.size());
  GlobalFixture::world->gop.broadcast_serializable(vector, 0);

  // Convert the vector to an array
  BOOST_CHECK_NO_THROW((array1 = eigen_to_array<TArrayI>(*GlobalFixture::world,
                                                         trange1, vector)));

  // Check that the data in array matches the data in vector
  for (Range::const_iterator it = array1.tiles_range().begin();
       it != array1.tiles_range().end(); ++it) {
    Future<TArrayI::value_type> tile = array1.find(*it);
    for (Range::const_iterator tile_it = tile.get().range().begin();
         tile_it != tile.get().range().end(); ++tile_it) {
      BOOST_CHECK_EQUAL(tile.get()[*tile_it], vector((*tile_it)[0]));
    }
  }
}

BOOST_AUTO_TEST_CASE(array_to_matrix) {
  auto a_to_e_rowmajor = [](const TArrayI& array) -> EigenMatrixXi {
    return array_to_eigen<Tensor<int>, DensePolicy, Eigen::RowMajor>(array);
  };

  if (GlobalFixture::world->size() == 1) {
    // Fill the array with random data
    GlobalFixture::world->srand(27);
    for (Range::const_iterator it = array.tiles_range().begin();
         it != array.tiles_range().end(); ++it) {
      TArrayI::value_type tile(array.trange().make_tile_range(*it));
      for (TArrayI::value_type::iterator tile_it = tile.begin();
           tile_it != tile.end(); ++tile_it) {
        *tile_it = GlobalFixture::world->rand();
      }
      array.set(*it, tile);
    }

    // Convert the array to an Eigen matrices: column-major (matrix) and
    // row-major (rmatrix)
    BOOST_CHECK_NO_THROW(matrix = array_to_eigen(array));
    BOOST_CHECK_NO_THROW(rmatrix = a_to_e_rowmajor(array));

    // Check that the matrix dimensions are the same as the array
    BOOST_CHECK_EQUAL(matrix.rows(), array.trange().elements_range().extent(0));
    BOOST_CHECK_EQUAL(matrix.cols(), array.trange().elements_range().extent(1));
    BOOST_CHECK_EQUAL(rmatrix.rows(),
                      array.trange().elements_range().extent(0));
    BOOST_CHECK_EQUAL(rmatrix.cols(),
                      array.trange().elements_range().extent(1));

    // Check that the data in matrix matches the data in array
    for (Range::const_iterator it = array.tiles_range().begin();
         it != array.tiles_range().end(); ++it) {
      Future<TArrayI::value_type> tile = array.find(*it);
      for (Range::const_iterator tile_it = tile.get().range().begin();
           tile_it != tile.get().range().end(); ++tile_it) {
        BOOST_CHECK_EQUAL(matrix((*tile_it)[0], (*tile_it)[1]),
                          tile.get()[*tile_it]);
        BOOST_CHECK_EQUAL(rmatrix((*tile_it)[0], (*tile_it)[1]),
                          tile.get()[*tile_it]);
      }
    }
  } else {
    // Check that eigen_to_array throws when there is more than one node
    BOOST_CHECK_THROW(array_to_eigen(array), TiledArray::Exception);

    // Fill local tiles with data
    GlobalFixture::world->srand(27);
    TArrayI::pmap_interface::const_iterator it = array.pmap()->begin();
    TArrayI::pmap_interface::const_iterator end = array.pmap()->end();
    for (; it != end; ++it) {
      TArrayI::value_type tile(array.trange().make_tile_range(*it));
      for (TArrayI::value_type::iterator tile_it = tile.begin();
           tile_it != tile.end(); ++tile_it) {
        *tile_it = GlobalFixture::world->rand();
      }
      array.set(*it, tile);
    }

    // Distribute the data of array1 to all nodes
    array.make_replicated();

    BOOST_CHECK(array.pmap()->is_replicated());

    // Convert the array to an Eigen matrix
    BOOST_CHECK_NO_THROW(matrix = array_to_eigen(array));
    BOOST_CHECK_NO_THROW(rmatrix = a_to_e_rowmajor(array));

    // Check that the matrix dimensions are the same as the array
    BOOST_CHECK_EQUAL(matrix.rows(), array.trange().elements_range().extent(0));
    BOOST_CHECK_EQUAL(matrix.cols(), array.trange().elements_range().extent(1));
    BOOST_CHECK_EQUAL(rmatrix.rows(),
                      array.trange().elements_range().extent(0));
    BOOST_CHECK_EQUAL(rmatrix.cols(),
                      array.trange().elements_range().extent(1));

    // Check that the data in vector matches the data in array
    for (Range::const_iterator it = array.tiles_range().begin();
         it != array.tiles_range().end(); ++it) {
      BOOST_CHECK(array.is_local(*it));

      Future<TArrayI::value_type> tile = array.find(*it);
      for (Range::const_iterator tile_it = tile.get().range().begin();
           tile_it != tile.get().range().end(); ++tile_it) {
        BOOST_CHECK_EQUAL(matrix((*tile_it)[0], (*tile_it)[1]),
                          tile.get()[*tile_it]);
        BOOST_CHECK_EQUAL(rmatrix((*tile_it)[0], (*tile_it)[1]),
                          tile.get()[*tile_it]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(array_to_vector) {
  if (GlobalFixture::world->size() == 1) {
    // Fill the array with random data
    GlobalFixture::world->srand(27);
    for (Range::const_iterator it = array1.tiles_range().begin();
         it != array1.tiles_range().end(); ++it) {
      TArrayI::value_type tile(array1.trange().make_tile_range(*it));
      for (TArrayI::value_type::iterator tile_it = tile.begin();
           tile_it != tile.end(); ++tile_it) {
        *tile_it = GlobalFixture::world->rand();
      }
      array1.set(*it, tile);
    }

    // Convert the array to an Eigen vector
    BOOST_CHECK_NO_THROW(vector = array_to_eigen(array1));

    // Check that the matrix dimensions are the same as the array
    BOOST_CHECK_EQUAL(vector.rows(),
                      array1.trange().elements_range().extent(0));
    BOOST_CHECK_EQUAL(vector.cols(), 1);

    // Check that the data in vector matches the data in array
    for (Range::const_iterator it = array1.tiles_range().begin();
         it != array1.tiles_range().end(); ++it) {
      Future<TArrayI::value_type> tile = array1.find(*it);
      for (Range::const_iterator tile_it = tile.get().range().begin();
           tile_it != tile.get().range().end(); ++tile_it) {
        BOOST_CHECK_EQUAL(vector((*tile_it)[0]), tile.get()[*tile_it]);
      }
    }
  } else {
    // Check that eigen_to_array throws when there is more than one node
    BOOST_CHECK_THROW(array_to_eigen(array1), TiledArray::Exception);

    // Fill local tiles with data
    GlobalFixture::world->srand(27);
    TArrayI::pmap_interface::const_iterator it = array1.pmap()->begin();
    TArrayI::pmap_interface::const_iterator end = array1.pmap()->end();
    for (; it != end; ++it) {
      TArrayI::value_type tile(array1.trange().make_tile_range(*it));
      for (TArrayI::value_type::iterator tile_it = tile.begin();
           tile_it != tile.end(); ++tile_it) {
        *tile_it = GlobalFixture::world->rand();
      }
      array1.set(*it, tile);
    }

    // Distribute the data of array1 to all nodes
    array1.make_replicated();

    BOOST_CHECK(array1.pmap()->is_replicated());

    // Convert the array to an Eigen vector
    BOOST_CHECK_NO_THROW(vector = array_to_eigen(array1));

    // Check that the matrix dimensions are the same as the array
    BOOST_CHECK_EQUAL(vector.rows(),
                      array1.trange().elements_range().extent(0));
    BOOST_CHECK_EQUAL(vector.cols(), 1);

    // Check that the data in vector matches the data in array
    for (Range::const_iterator it = array1.tiles_range().begin();
         it != array1.tiles_range().end(); ++it) {
      BOOST_CHECK(array1.is_local(*it));

      Future<TArrayI::value_type> tile = array1.find(*it);
      for (Range::const_iterator tile_it = tile.get().range().begin();
           tile_it != tile.get().range().end(); ++tile_it) {
        BOOST_CHECK_EQUAL(vector((*tile_it)[0]), tile.get()[*tile_it]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(subtensor_to_tensor) {
  // Fill the tensor with random data
  tensor.setRandom();
  // Make a target tensor
  Tensor<int> ta_tensor(trangeN.make_tile_range(0));

  // Copy the sub matrix to the tensor objects
  BOOST_CHECK_NO_THROW(eigen_subtensor_to_tensor(tensor, ta_tensor));

  // Check that the block contains the same values as the tensor
  for (Range::const_iterator it = ta_tensor.range().begin();
       it != ta_tensor.range().end(); ++it) {
    const auto& ta_idx = (*it);
    std::array<long, GlobalFixture::dim> idx;
    std::copy(ta_idx.begin(), ta_idx.end(), idx.begin());
    BOOST_CHECK_EQUAL(ta_tensor[*it], tensor(idx));
  }
}

BOOST_AUTO_TEST_CASE(tensor_to_subtensor) {
  // Fill a tensor with data
  Tensor<int> ta_tensor(trangeN.make_tile_range(0));
  GlobalFixture::world->srand(27);
  for (auto it = ta_tensor.begin(); it != ta_tensor.end(); ++it)
    *it = GlobalFixture::world->rand();

  // Copy the tensor to the submatrix block
  BOOST_CHECK_NO_THROW(tensor_to_eigen_subtensor(ta_tensor, tensor));

  // Check that the block contains the same values as the tensor
  for (auto it = ta_tensor.range().begin(); it != ta_tensor.range().end();
       ++it) {
    const auto& ta_idx = (*it);
    std::array<long, GlobalFixture::dim> idx;
    std::copy(ta_idx.begin(), ta_idx.end(), idx.begin());
    BOOST_CHECK_EQUAL(tensor(idx), ta_tensor[*it]);
  }
}

BOOST_AUTO_TEST_CASE(tensor_to_array) {
  // Fill tensor with random data and replicate across the world
  tensor.setRandom();
  GlobalFixture::world->gop.broadcast_serializable(tensor, 0);

  // test serialization if have more than 1 rank
  if (GlobalFixture::world->size() > 1) {
    decltype(tensor) tensor_copy;
    if (GlobalFixture::world->rank() == 1) tensor_copy = tensor;
    GlobalFixture::world->gop.broadcast_serializable(tensor_copy, 1);
    Eigen::Tensor<bool, 0> eq = (tensor == tensor_copy).all();
    BOOST_CHECK(eq() == true);
  }

  // Copy matrix to array
  BOOST_CHECK_NO_THROW((array = eigen_tensor_to_array<TArrayI>(
                            *GlobalFixture::world, trangeN, tensor)));

  // Check that the data in array is equal to that in matrix
  for (Range::const_iterator it = array.tiles_range().begin();
       it != array.tiles_range().end(); ++it) {
    Future<TArrayI::value_type> tile = array.find(*it);
    for (Range::const_iterator tile_it = tile.get().range().begin();
         tile_it != tile.get().range().end(); ++tile_it) {
      std::array<long, GlobalFixture::dim> idx;
      auto& t_idx = *tile_it;
      std::copy(t_idx.begin(), t_idx.end(), idx.begin());
      BOOST_CHECK_EQUAL(tile.get()[*tile_it], tensor(idx));
    }
  }
}

BOOST_AUTO_TEST_CASE(array_to_tensor) {
  using Tensor = Eigen::Tensor<int, GlobalFixture::dim>;
  using RowTensor = Eigen::Tensor<int, GlobalFixture::dim, Eigen::RowMajor>;
  auto a_to_e_rowmajor = [](const TArrayI& array) -> RowTensor {
    return array_to_eigen_tensor<RowTensor>(array);
  };

  auto to_array = [](const auto& seq) {
    TA_ASSERT(seq.size() == GlobalFixture::dim);
    std::array<long, GlobalFixture::dim> result;
    std::copy(seq.begin(), seq.end(), result.begin());
    return result;
  };

  // Fill local tiles with data
  GlobalFixture::world->srand(27);
  TArrayI::pmap_interface::const_iterator it = arrayN.pmap()->begin();
  TArrayI::pmap_interface::const_iterator end = arrayN.pmap()->end();
  for (; it != end; ++it) {
    TArrayI::value_type tile(arrayN.trange().make_tile_range(*it));
    for (TArrayI::value_type::iterator tile_it = tile.begin();
         tile_it != tile.end(); ++tile_it) {
      *tile_it = GlobalFixture::world->rand();
    }
    arrayN.set(*it, tile);
  }

  if (GlobalFixture::world->size() > 1) {
    // Check that array_to_eigen_tensor throws when there is more than one node
    BOOST_CHECK_THROW(array_to_eigen_tensor<Tensor>(arrayN),
                      TiledArray::Exception);
  }

  // Distribute the data of arrayN to all nodes
  if (GlobalFixture::world->size() > 1) {
    arrayN.make_replicated();
    BOOST_CHECK(arrayN.pmap()->is_replicated());
  }

  // Convert the array to an Eigen matrix
  BOOST_CHECK_NO_THROW(tensor = array_to_eigen_tensor<Tensor>(arrayN));
  BOOST_CHECK_NO_THROW(rtensor = a_to_e_rowmajor(arrayN));

  // Check that the matrix dimensions are the same as the array
  BOOST_CHECK_EQUAL_COLLECTIONS(
      tensor.dimensions().begin(), tensor.dimensions().end(),
      arrayN.trange().elements_range().extent().begin(),
      arrayN.trange().elements_range().extent().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      rtensor.dimensions().begin(), rtensor.dimensions().end(),
      arrayN.trange().elements_range().extent().begin(),
      arrayN.trange().elements_range().extent().end());

  // Check that the data in vector matches the data in array
  for (Range::const_iterator it = arrayN.tiles_range().begin();
       it != arrayN.tiles_range().end(); ++it) {
    BOOST_CHECK(arrayN.is_local(*it));

    Future<TArrayI::value_type> tile = arrayN.find(*it);
    for (Range::const_iterator tile_it = tile.get().range().begin();
         tile_it != tile.get().range().end(); ++tile_it) {
      BOOST_CHECK_EQUAL(tensor(to_array(*tile_it)), tile.get()[*tile_it]);
      BOOST_CHECK_EQUAL(rtensor(to_array(*tile_it)), tile.get()[*tile_it]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
