#include <TiledArray/util/bug.h>
#include <tiledarray.h>

// Construct a Tensor<T> filled with v
template <typename T>
TA::Tensor<T> make_tile(const TA::Range& range, const double v) {
  // Allocate and fill a tile
  TA::Tensor<T> tile(range);
  std::fill(tile.begin(), tile.end(), v);

  return tile;
}

// Construct a 2-dimensional Tensor<T> filled with v
// tests explicit iteration over elements
template <typename T>
TA::Tensor<T> make_tile2(const TA::Range& range, const double v) {
  assert(range.rank() == 2);

  // Allocate a tile
  TA::Tensor<T> tile(range);

  // Store a reference to the start and finish array of the tile
  const auto& lobound = tile.range().lobound();
  const auto& upbound = tile.range().upbound();

  // Fill the tile
  std::size_t i[] = {0, 0};
  //  or, std::vector<std::size_t> i = {0,0};
  //  or, std::array<std::size_t, 2> i = {0,0};
  for (i[0] = lobound[0]; i[0] != upbound[0]; ++i[0])
    for (i[1] = lobound[1]; i[1] != upbound[1]; ++i[1]) tile[i] = v;

  return tile;
}

// Fill array x with value v
void init_array(TA::TArrayD& x, const double v) {
  // Add local tiles to a
  for (auto it = begin(x); it != end(x); ++it) {
    // Construct a tile using a MADNESS task.
    auto tile = x.world().taskq.add(make_tile<double>,
                                    x.trange().make_tile_range(it.index()), v);

    // Insert the tile into the array
    *it = tile;
  }
}

int main(int argc, char* argv[]) {
  // Initialize runtime
  auto& world = TA::initialize(argc, argv);

  //  N.B. uncomment to launch via LLDB:
  //  using TiledArray::Debugger;
  //  auto debugger = std::make_shared<Debugger>("user-guide-2");
  //  Debugger::set_default_debugger(debugger);
  //  debugger->set_prefix(world.rank());
  //  debugger->set_cmd("lldb_xterm");
  //  //debugger->set_cmd("gdb_xterm");
  //  debugger->debug("ready to run");

  // Construct tile boundary vector
  std::vector<std::size_t> tile_boundaries;
  for (std::size_t i = 0; i <= 16; i += 4) tile_boundaries.push_back(i);

  // Construct a set of TiledRange1's
  std::vector<TA::TiledRange1> ranges(
      2, TA::TiledRange1(tile_boundaries.begin(), tile_boundaries.end()));

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
