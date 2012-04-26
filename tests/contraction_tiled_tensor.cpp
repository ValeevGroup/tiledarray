#include "TiledArray/contraction_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct ContractionTiledTensorFixture : public AnnotatedArrayFixture {
  typedef ContractionTiledTensor<array_annotation, array_annotation> CTT;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;

  ContractionTiledTensorFixture() : aa_left(a(left_var)), aa_right(a(right_var)), ctt(aa_left, aa_right) { }

  matrix_type array_to_matrix(const array_annotation& array, const std::size_t x) {
    const std::size_t dim = array.trange().elements().dim();
    assert(x <= dim);

    // Get the result matrix dimensions
    std::multiplies<std::size_t> mult_op;
    const std::size_t M = std::accumulate(array.trange().elements().size().begin(),
        array.trange().elements().size().begin() + x, 1ul, mult_op);
    const std::size_t N = std::accumulate(array.trange().elements().size().begin() + x,
        array.trange().elements().size().end(), 1ul, mult_op);

    // Construct the result matrix
    matrix_type result(M,N);

    // Copy each tile of array into result
    std::size_t row = 0;
    std::size_t col = 0;
    for(std::size_t t = 0; t < array.size(); ++t) {
      // Get tile i. The future must be held in memory or remote tiles will be
      // destroyed with the future. Wait for the future to be evaluated and store
      // a reference for convenience.
      madness::Future<array_annotation::value_type> f = array[t];
      const array_annotation::value_type& tile = f.get();

      // Get the number of rows and tiles in the block
      const std::size_t tile_rows = std::accumulate(tile.range().size().begin(), tile.range().size().begin() + x, 1ul, mult_op);
      const std::size_t tile_cols = std::accumulate(tile.range().size().begin() + x, tile.range().size().end(), 1ul, mult_op);

      // copy the tile into a matrix block
      for(std::size_t i = 0ul; i < tile_rows; ++i) {
        for(std::size_t j = 0ul; j < tile_cols; ++j) {
          // Calculate the ordinal index of the tile element
          const std::size_t o = i * tile_cols + j;
          result(row + i, col + j) = tile[o];
        }
      }

      // Increment row and col for next tile
      col += tile_cols;
      if(col >= N) {
        col = 0;
        row += tile_rows;
      }
    }
    return result;
  }

  static const VariableList left_var;
  static const VariableList right_var;
  static const std::shared_ptr<math::Contraction> cont;

  array_annotation aa_left;
  array_annotation aa_right;
  CTT ctt;
};

const VariableList ContractionTiledTensorFixture::left_var(
    AnnotatedArrayFixture::make_var_list(0, GlobalFixture::coordinate_system::dim));
const VariableList ContractionTiledTensorFixture::right_var(
    AnnotatedArrayFixture::make_var_list(1, GlobalFixture::coordinate_system::dim + 1));

const std::shared_ptr<math::Contraction> ContractionTiledTensorFixture::cont(new math::Contraction(
    VariableList(AnnotatedArrayFixture::make_var_list(0, GlobalFixture::coordinate_system::dim)),
    VariableList(AnnotatedArrayFixture::make_var_list(1, GlobalFixture::coordinate_system::dim + 1))));


BOOST_FIXTURE_TEST_SUITE( contraction_tiled_tensor_suite, ContractionTiledTensorFixture )

BOOST_AUTO_TEST_CASE( range )
{
  DynamicTiledRange dtr = cont->contract_trange(a.trange(), a.trange());
  BOOST_CHECK_EQUAL(ctt.range(), dtr.tiles());
  BOOST_CHECK_EQUAL(ctt.size(), dtr.tiles().volume());
  BOOST_CHECK_EQUAL(ctt.trange(), dtr);
}

BOOST_AUTO_TEST_CASE( vars )
{
  VariableList var2(left_var.data().front() + "," + right_var.data().back());
  BOOST_CHECK_EQUAL(ctt.vars(), var2);
}

BOOST_AUTO_TEST_CASE( shape )
{
  BOOST_CHECK_EQUAL(ctt.is_dense(), a.is_dense());
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(ctt.get_shape(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( result )
{
  // Store a copy of the trange so it can be checked later.
  StaticTiledRange<CoordinateSystem<2> > r0 = ctt.trange();

  const std::size_t size = a.trange().tiles().size().front() * a.trange().tiles().size().back();

  // Evaluate and wait for it to finish.
  ctt.eval(ctt.vars(), std::shared_ptr<CTT::pmap_interface>(
      new TiledArray::detail::BlockedPmap(* GlobalFixture::world, size))).get();

  // Construct equivalent matrix.
  matrix_type left = array_to_matrix(aa_left, 1);
  matrix_type right = array_to_matrix(aa_right, 1);

  // Get the result matrix
  matrix_type result = left * right.transpose();

  // Check that the range is unchanged
  BOOST_CHECK_EQUAL(ctt.trange(), r0);
  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[0], std::size_t(result.rows()));
  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[1], std::size_t(result.cols()));

  // Check that all the tiles have been evaluated.
  std::size_t n = std::distance(ctt.begin(), ctt.end());
  world.gop.sum(n);
  BOOST_CHECK_EQUAL(ctt.size(), n);

  for(CTT::const_iterator it = ctt.begin(); it != ctt.end(); ++it) {
    const CTT::value_type& tile = it->get();

    matrix_type block = result.block(
        tile.range().start()[0], tile.range().start()[1],
        tile.range().size()[0], tile.range().size()[1]);

    for(std::size_t i = 0; i < tile.size(); ++i)
      BOOST_CHECK_EQUAL(tile[i], block(i));
  }
}

BOOST_AUTO_TEST_CASE( permute_result )
{
  Permutation p(1,0);

  StaticTiledRange<CoordinateSystem<2> > r0 = p ^ ctt.trange();

  const std::size_t size = a.trange().tiles().size().front() * a.trange().tiles().size().back();

  // Evaluate and wait for it to finish.
  ctt.eval(p ^ ctt.vars(), std::shared_ptr<CTT::pmap_interface>(
      new TiledArray::detail::BlockedPmap(* GlobalFixture::world, size))).get();

  // Construct equivalent matrix.
  matrix_type left = array_to_matrix(aa_left, 1);
  matrix_type right = array_to_matrix(aa_right, 1);

  // Get the result matrix
  matrix_type result = left * right.transpose();
  result.transposeInPlace();

  // Check that the range has been permuted correctly.
  BOOST_CHECK_EQUAL(ctt.trange(), r0);
  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[0], std::size_t(result.rows()));
  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[1], std::size_t(result.cols()));

  // Check that all the tiles have been evaluated.
  std::size_t n = std::distance(ctt.begin(), ctt.end());
  world.gop.sum(n);
  BOOST_CHECK_EQUAL(ctt.size(), n);

  for(CTT::const_iterator it = ctt.begin(); it != ctt.end(); ++it) {
    const CTT::value_type& tile = it->get();

    matrix_type block = result.block(
        tile.range().start()[0], tile.range().start()[1],
        tile.range().size()[0], tile.range().size()[1]);

    for(std::size_t i = 0; i < tile.size(); ++i)
      BOOST_CHECK_EQUAL(tile[i], block(i));
  }
}


BOOST_AUTO_TEST_SUITE_END()
