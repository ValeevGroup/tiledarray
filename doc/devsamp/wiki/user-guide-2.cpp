#include <tiledarray.h>

// Construct a Tensor<T> filled with v
template <typename T>
TA::Tensor<T> make_tile(const TA::Range& range, const double v) {
  // Allocate a tile
  TA::Tensor<T> tile(range);
  std::fill(tile.begin(), tile.end(), v);

  return tile;
}

// Fill array x with value v
void init_array(TA::TArrayD& x, const double v) {
  // Add local tiles to a
  for(auto it = begin(x); it != end(x); ++it) {
    // Construct a tile using a MADNESS task.
    auto tile = x.world().taskq.add(make_tile<double>, x.trange().make_tile_range(it.index()), v);

    // Insert the tile into the array
    *it = tile;
  }
}

int main(int argc, char* argv[]) {
  // Initialize runtime
  auto& world = TA::initialize(argc, argv);

  // Construct tile boundary vector
  std::vector<std::size_t> tile_boundaries;
  for(std::size_t i = 0; i <= 16; i += 4)
    tile_boundaries.push_back(i);

  // Construct a set of TiledRange1's
  std::vector<TA::TiledRange1>
    ranges(2, TA::TiledRange1(tile_boundaries.begin(), tile_boundaries.end()));

  // Construct the 2D TiledRange
  TA::TiledRange trange(ranges.begin(), ranges.end());

  // Construct array objects.
  TA::TArrayD a(world, trange);
  TA::TArrayD b(world, trange);
  TA::TArrayD c(world, trange);

  // Initialize a and b.
  init_array(a, 3.0);
  init_array(b, 2.0);

  // Print the content of input tensors, a and b.
  std::cout << "a = \n" << a << "\n";
  std::cout << "b = \n" << b << "\n";

  // Compute the contraction c(m,n) = sum_k a(m,k) * b(k,n)
  c("m,n") = a("m,k") * b("k,n");

  // Print the result tensor, c.
  std::cout << "c = \n" << c << "\n";

  // Wait for all the computation to complete before exiting
  world.gop.fence();

  TA::finalize();
  return 0;
}
