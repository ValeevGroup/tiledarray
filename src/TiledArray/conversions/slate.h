#ifndef TILEDARRAY_CONVERSIONS_SLATE_H
#define TILEDARRAY_CONVERSIONS_SLATE_H

#include <TiledArray/config.h> // TILEDARRAY_HAS_SLATE
#if TILEDARRAY_HAS_SLATE

#include <slate/slate.hh>                // slate::Matrix
#include <TiledArray/type_traits.h>      // TA::numeric_type
#include <TiledArray/shape.h>            // is_dense
#include <TiledArray/pmap/user_pmap.h>   // {,User}Pmap
#include <Eigen/Core>                    // Eigen::{Matrix,Map}
#include <TiledArray/external/madness.h> // MADNESS

namespace TiledArray {
namespace detail {

/// C++14-esque typename wrapper for `numeric_type
template <typename Array>
using numeric_type_t = typename numeric_type<Array>::type;

/// Deduce SLATE Matrix type from Array type
template <typename Array>
using slate_type_from_array_t =
    typename slate::Matrix<numeric_type_t<Array>>;


template <typename Tile, typename T>
auto slate_to_ta_tile(slate::Tile<T> slate_tile, TA::Range const& range) {

  using col_major_mat_t = Eigen::Matrix<T,-1,-1,Eigen::ColMajor>;
  using row_major_mat_t = Eigen::Matrix<T,-1,-1,Eigen::RowMajor>;

  using col_major_map_t = Eigen::Map<const col_major_mat_t>;
  using row_major_map_t = Eigen::Map<row_major_mat_t>;

  Tile tile(range, 0.0);

  // Create Maps                
  auto local_m = slate_tile.mb();
  auto local_n = slate_tile.nb();
  col_major_map_t slate_map(slate_tile.data(), local_m, local_n);

  auto  local_m_ta = range.dim(0).extent();
  auto  local_n_ta = range.dim(1).extent();
  row_major_map_t ta_map(tile.data(), local_m_ta, local_n_ta);

  // Copy data
  ta_map = slate_map;

  return tile;
}

template <typename Array, typename Matrix>
Array make_array_from_slate_dense(Matrix& matrix, TA::TiledRange const& trange,
   std::shared_ptr<TA::Pmap> pmap, TA::World& world) {

  using value_type = typename Array::value_type; // Tile type
  using element_type   = typename std::remove_cv_t<Array>::element_type;


  using col_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::ColMajor>;
  using row_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::RowMajor>;

  using col_major_map_t = Eigen::Map<const col_major_mat_t>;
  using row_major_map_t = Eigen::Map<row_major_mat_t>;

  Array array(world, trange, pmap);
  for (int64_t it = 0; it < matrix.mt(); ++it)
  for (int64_t jt = 0; jt < matrix.nt(); ++jt) 
  if( matrix.tileIsLocal(it,jt) ) {
    auto local_ordinal = trange.tiles_range().ordinal(it,jt);

    auto tile = world.taskq.add(slate_to_ta_tile<value_type,element_type>,
        matrix(it,jt), trange.make_tile_range(local_ordinal));
    
    array.set(local_ordinal, tile);
  }

  return array;
}

template <typename Array, typename Matrix>
Array make_array_from_slate_sparse(Matrix& matrix, TA::TiledRange const& trange,
   std::shared_ptr<TA::Pmap> pmap, TA::World& world) {

  typedef typename Array::value_type value_type;
  typedef typename Array::ordinal_type ordinal_type;
  typedef std::pair<ordinal_type, Future<value_type> > datum_type;

  // Create a vector to hold local tiles
  std::vector<datum_type> tiles;
  tiles.reserve(pmap->size());

  // Construct a tensor to hold updated tile norms for the result shape.
  TA::Tensor<typename detail::shape_t<Array>::value_type> tile_norms(
      trange.tiles_range(), 0);

  // Construct the task function used to construct the result tiles.
  madness::AtomicInt counter;
  counter = 0;
  int task_count = 0;
  auto task = [&](const ordinal_type index) -> value_type {

    value_type tile;
    auto coords = trange.tiles_range().idx(index);
    auto it = coords[0];
    auto jt = coords[1];
    if(!matrix.tileIsLocal(it,jt)) {
      tile_norms[index] = 0.0;
    } else {
      tile = slate_to_ta_tile<value_type>(matrix(it,jt), trange.make_tile_range(index));
      tile_norms[index] = norm(tile);
    }
    ++counter;
    return tile;

  };

  for (const auto index : *pmap) {
    auto result_tile = world.taskq.add(task, index);
    ++task_count;
    tiles.emplace_back(index, std::move(result_tile));
  }

  // Wait for tile norm data to be collected.
  if (task_count > 0)
    world.await(
        [&counter, task_count]() -> bool { return counter == task_count; });

  // Construct the new array
  Array result(world, trange,
               typename Array::shape_type(world, tile_norms, trange), pmap);
  for (auto& it : tiles) {
    const auto index = it.first;
    if (!result.is_zero(index)) result.set(it.first, it.second);
  }

  return result;

}
      
} // namespace TiledArray::detail

class SlateFunctors {

public:

    using slate_int = int64_t;
    using slate_process_idx = std::tuple<slate_int, slate_int>;
    using dim_functor_t  = std::function<slate_int(slate_int)>;
    using tile_functor_t = std::function<int(slate_process_idx)>;

    SlateFunctors( const dim_functor_t& Mb, const dim_functor_t& Nb,
      const tile_functor_t& Rank, const tile_functor_t& Dev ) :
      tileMb_(Mb), tileNb_(Nb), tileRank_(Rank), tileDevice_(Dev) { }

    template <typename PMapInterfacePointer>
    SlateFunctors( TiledRange trange, PMapInterfacePointer pmap_ptr ) {
    if( trange.rank() != 2 )
        throw std::runtime_error("Cannot Convert General Tensor to SLATE (RANK != 2)");
      // Tile row dimension (MB)
      tileMb_ = [trange](slate_int i) { return trange.dim(0).tile(i).extent(); };

      // Tile col dimension (NB)
      tileNb_ = [trange](slate_int i) { return trange.dim(1).tile(i).extent(); };

      // Tile rank assignment
      tileRank_ = [pmap_ptr, trange] (slate_process_idx ij) {
        auto [i,j] = ij;
        return pmap_ptr->owner(trange.tiles_range().ordinal(i,j));
      };

      // Tile device assignment
      // TODO: Needs to be more robust
      tileDevice_ = [](slate_process_idx) { return 0; };
     
    }

  auto& tileMb() { return tileMb_; }
  auto& tileNb() { return tileNb_; }
  auto& tileRank() { return tileRank_; }
  auto& tileDevice() { return tileDevice_; }


  template <typename SlateMatrixType>
  SlateMatrixType make_matrix(int64_t M, int64_t N, MPI_Comm comm) {
    return SlateMatrixType(M, N, tileMb_, tileNb_, tileRank_, tileDevice_, comm);
  }

private:

    dim_functor_t tileMb_, tileNb_;
    tile_functor_t tileRank_, tileDevice_;
};





template <typename SlateMatrixType>
std::shared_ptr<TA::Pmap> make_pmap_from_slate( SlateMatrixType&& matrix, World& world ) {

    // Compute SLATE Tile Statistics
    size_t total_tiles = matrix.nt() * matrix.mt();
    size_t local_tiles = 0;

    // Create a map from tile ordinal to rank
    // to avoid lifetime issues in the internal
    // TA Pmap
    std::vector<size_t> tile2rank(total_tiles);
    for (int64_t it = 0; it < matrix.mt(); ++it)
    for (int64_t jt = 0; jt < matrix.nt(); ++jt) {
        size_t ordinal = it*matrix.nt() + jt; // TODO: Use Range
        tile2rank[ordinal] = matrix.tileRank( it, jt );
        if(matrix.tileIsLocal(it,jt)) local_tiles++;
    }
    

    // Create TA PMap
    std::function<size_t(size_t)> ta_tile_functor = 
        [t2r = std::move(tile2rank)](size_t ordinal) {
            return t2r[ordinal];
        };

    return std::make_shared<TA::detail::UserPmap>(world, total_tiles, local_tiles, 
            ta_tile_functor);
}





/**
 * @brief Convert Array to SLATE matrix
 *
 * @tparam Array Type of input Array
 *
 * @param[in] array Array to convert to SLATE. Must be rank-2.
 * @returns SLATE representation `array`
 */
template <typename Array>
detail::slate_type_from_array_t<Array>
array_to_slate( const Array& array ) {

    using element_type   = typename std::remove_cv_t<Array>::element_type;
    using slate_matrix_t = typename slate::Matrix<element_type>;

    using col_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::ColMajor>;
    using row_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::RowMajor>;

    using col_major_map_t = Eigen::Map<col_major_mat_t>;
    using row_major_map_t = Eigen::Map<const row_major_mat_t>;

    /*******************************/
    /*** Generate SLATE Functors ***/
    /*******************************/
    auto&       world  = array.world();
    const auto& trange = array.trange();
    auto        pmap   = array.pmap();

    SlateFunctors slate_functors( trange, pmap );


    /*********************************/
    /*** Create empty slate matrix ***/
    /*********************************/
    const auto M = trange.dim(0).extent();
    const auto N = trange.dim(1).extent();
    auto matrix = slate_functors.make_matrix<slate_matrix_t>(M, N,
          world.mpi.comm().Get_mpi_comm());
    
    /************************/
    /*** Copy TA -> SLATE ***/
    /************************/
    matrix.insertLocalTiles();

    // Loop over local tiles via ordinal
    // TODO: Make async
    for( auto local_ordinal : *pmap ) {
        // Compute coordinate of tile ordinal
        auto local_coordinate = trange.tiles_range().idx(local_ordinal);
        const auto it = local_coordinate[0];
        const auto jt = local_coordinate[1];

        // Sanity Check
        if(!matrix.tileIsLocal(it,jt))
            throw std::runtime_error("SLATE PMAP is not valid");

        // Extract shallow copy of local SLATE tile and create
        // data map
        auto local_tile_slate = matrix(it,jt);
        auto local_m = local_tile_slate.mb();
        auto local_n = local_tile_slate.nb();
        col_major_map_t slate_map(local_tile_slate.data(), local_m, local_n);

        // Create data map for TA tile
        // TODO: This should be async in a MADNESS task
        auto& local_tile = array.find_local(local_ordinal).get();
        auto  local_m_ta = local_tile.range().dim(0).extent();
        auto  local_n_ta = local_tile.range().dim(1).extent();
        row_major_map_t ta_map(local_tile.data(), local_m_ta, local_n_ta);

        // Copy TA tile to SLATE tile
        slate_map = ta_map;
    } // Loop over local tiles

    return matrix;
} // array_to_slate

/**
 *  @brief Convert a SLATE matrix to an Array
 */
template <typename Array>
auto slate_to_array( /*const*/ detail::slate_type_from_array_t<Array>& matrix, World& world ) {
    // TODO: SLATE Tile accessor is not const-accessible 
    // https://github.com/icl-utk-edu/slate/issues/59

    using value_type = typename Array::value_type; // Tile type
    using element_type   = typename std::remove_cv_t<Array>::element_type;
    using slate_matrix_t = typename slate::Matrix<element_type>;

    using col_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::ColMajor>;
    using row_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::RowMajor>;

    using col_major_map_t = Eigen::Map<const col_major_mat_t>;
    using row_major_map_t = Eigen::Map<row_major_mat_t>;

    // Create TA PMap from SLATE metadata
    auto slate_pmap = make_pmap_from_slate(matrix, world);

    // Create TiledRange
    std::vector<size_t> row_tiling(matrix.mt()+1), col_tiling(matrix.nt()+1);

    row_tiling[0] = 0;
    for(auto i = 0; i < matrix.mt(); ++i) 
        row_tiling[i+1] = row_tiling[i] + matrix.tileMb(i);
    
    col_tiling[0] = 0;
    for(auto i = 0; i < matrix.nt(); ++i) 
        col_tiling[i+1] = col_tiling[i] + matrix.tileNb(i);


    std::vector<TA::TiledRange1> ranges = {
        TA::TiledRange1(row_tiling.begin(), row_tiling.end()),
        TA::TiledRange1(col_tiling.begin(), col_tiling.end())
    };
    TA::TiledRange trange(ranges.begin(), ranges.end());

    // Create TArray
#if 0
    Array array(world, trange, slate_pmap);
    for (int64_t it = 0; it < matrix.mt(); ++it)
    for (int64_t jt = 0; jt < matrix.nt(); ++jt) 
    if( matrix.tileIsLocal(it,jt) ) {
        auto local_ordinal = trange.tiles_range().ordinal(it,jt);

        auto tile = world.taskq.add(
            [=](slate::Tile<double> slate_tile, TA::Range const& range) {
                // Create tile
                value_type tile(range, 0.0);

                // Create Maps                
                auto local_m = slate_tile.mb();
                auto local_n = slate_tile.nb();
                col_major_map_t slate_map(slate_tile.data(), local_m, local_n);

                auto  local_m_ta = range.dim(0).extent();
                auto  local_n_ta = range.dim(1).extent();
                row_major_map_t ta_map(tile.data(), local_m_ta, local_n_ta);

                // Copy data
                ta_map = slate_map;

                return tile;
            }, matrix(it,jt), trange.make_tile_range(local_ordinal));
        
        array.set(local_ordinal, tile);
    }
#else
    Array array;
    if constexpr (is_dense<Array>::value) {
      array = detail::make_array_from_slate_dense<Array>(matrix, 
          trange, slate_pmap, world);
    } else {
      array = detail::make_array_from_slate_sparse<Array>(matrix, 
          trange, slate_pmap, world);
    }
#endif

    world.gop.fence();
    return array;
}

} // namespace TiledArray

#endif // TILEDARRAY_HAS_SLATE
#endif // TILEDARRAY_CONVERSIONS_SLATE_H
