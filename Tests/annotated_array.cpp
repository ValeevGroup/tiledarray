#include "TiledArray/annotated_array.h"
#include "TiledArray/annotated_tile.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::expressions::VariableList;
using TiledArray::expressions::array::ArrayHolder;

// Define the number of dimensions used.
namespace {
  const unsigned int dim = 3;
}

template <typename A>
struct TiledRangeFixture {
  typedef typename A::tiled_range_type TRange3;
  typedef typename TRange3::tiled_range1_type TRange1;

  TiledRangeFixture() {
    const std::size_t d0[] = {0, 10, 20, 30};
    const std::size_t d1[] = {0, 5, 10, 15, 20};
    const std::size_t d2[] = {0, 3, 6, 9, 12, 15};
    const TRange1 dim0(d0, d0 + 4);
    const TRange1 dim1(d1, d1 + 5);
    const TRange1 dim2(d2, d2 + 6);
    const TRange1 dims[3] = {dim0, dim1, dim2};
    trng.resize(dims, dims + 3);
  }

  ~TiledRangeFixture() { }

  TRange3 trng;
}; // struct TiledRangeFixture

struct ArrayHolderFixture : public TiledRangeFixture<Array<int, dim> > {
  typedef Array<int, dim> Array3T;
  typedef Array<int, dim, CoordinateSystem<dim>, expressions::tile::AnnotatedTile<int> > Array3A;
  typedef Array<int, dim, CoordinateSystem<dim>, madness::Future<Tile<int, dim> > > Array3FT;
  typedef Array<int, dim, CoordinateSystem<dim>, madness::Future<expressions::tile::AnnotatedTile<int> > > Array3FA;

  typedef ArrayHolder<Array3T> AHolder3T;
  typedef ArrayHolder<Array3A> AHolder3A;
  typedef ArrayHolder<Array3FT> AHolder3FT;
  typedef ArrayHolder<Array3FA> AHolder3FA;

  typedef Array3T::index_type index_type;
  typedef Array3T::tile_index_type tile_index_type;
  typedef Array3T::tile_type tile_type;

  ArrayHolderFixture() : world(GlobalFixture::world),
      a(*world, trng), at(*world, trng), aa(*world, trng), aft(*world, trng),
      afa(*world, trng), pat(&at, &AHolder3T::no_delete),
      paa(&aa, &AHolder3A::no_delete), paft(&aft, &AHolder3FT::no_delete),
      pafa(&afa, &AHolder3FA::no_delete), ht(pat), ha(paa), hft(paft), hfa(pafa)
  {
    int v = 1;
    int tv = 1;
    for(TRange3::range_type::const_iterator it = a.tiles().begin(); it != a.tiles().end(); ++it) {
      tile_type t(a.tile(*it), v);
      tv = v++;
      for(tile_type::iterator t_it = t.begin(); t_it != t.end(); ++t_it)
        *t_it = tv++;
      a.insert(*it, t);
    }
    world->gop.fence();
  }

  ~ArrayHolderFixture() { }

  // Sum the first elements of each tile on all nodes.
  double sum_first(const Array3T& a) {
    double sum = 0.0;
    for(Array3T::const_iterator it = a.begin(); it != a.end(); ++it)
      sum += it->second.at(0);

    world->mpi.comm().Allreduce(MPI_IN_PLACE, &sum, 1, MPI::DOUBLE, MPI::SUM);

    return sum;
  }

  // Count the number of tiles in the array on all nodes.
  std::size_t tile_count(const Array3T& a) {
    int n = static_cast<int>(a.volume(true));
    world->mpi.comm().Allreduce(MPI_IN_PLACE, &n, 1, MPI::INT, MPI::SUM);

    return n;
  }

  madness::World* world;
  Array3T a;
  Array3T at;
  Array3A aa;
  Array3FT aft;
  Array3FA afa;
  boost::shared_ptr<Array3T> pat;
  boost::shared_ptr<Array3A> paa;
  boost::shared_ptr<Array3FT> paft;
  boost::shared_ptr<Array3FA> pafa;
  AHolder3T ht;
  AHolder3A ha;
  AHolder3FT hft;
  AHolder3FA hfa;
}; // struct ArrayHolderFixture

BOOST_FIXTURE_TEST_SUITE( array_holder_suite , ArrayHolderFixture )

BOOST_AUTO_TEST_CASE( array_dim )
{
  BOOST_CHECK_EQUAL(ht.dim(), a.dim);
  BOOST_CHECK_EQUAL(ht.order(), a.order);
  BOOST_CHECK_EQUAL(ht.size(), a.size());
  BOOST_CHECK_EQUAL(ht.weight(), a.weight());
  BOOST_CHECK_EQUAL(ht.volume(), a.volume());
  BOOST_CHECK_EQUAL(ht.volume(true), a.volume(true));
  BOOST_CHECK( ht.range() == a.range() );
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(AHolder3T h0(pat));
}

BOOST_AUTO_TEST_SUITE_END()

struct AnnotatedArrayFixture {

}; // struct AnnotatedArrayFixture

BOOST_FIXTURE_TEST_SUITE( annotated_array_suite , AnnotatedArrayFixture )

BOOST_AUTO_TEST_SUITE_END()
