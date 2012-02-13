#include <iostream>
#include <tiled_array.h>

#define MEMORY 1000000000

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  madness::World world(MPI::COMM_WORLD);

  const std::size_t block_size = 32ul;
  const std::size_t num_blocks = (0.8 * std::sqrt(double(MEMORY / sizeof(double)) / 6.0) ) / block_size;

  if(world.rank() == 0)
    std::cout << "Matrix size = " << num_blocks * block_size << "x" << num_blocks * block_size << "\n"
        << "Memory per matrix = " << double(num_blocks * block_size * num_blocks * block_size * sizeof(double)) / 1000000000.0 << "GB" << std::endl;

  std::vector<unsigned int> blocking;
  blocking.reserve(num_blocks + 1);
  blocking.push_back(0);
  for(std::size_t i = 1; i < num_blocks; ++i)
    blocking.push_back(blocking[i-1] + block_size);

  std::array<TiledArray::TiledRange1, 2> blocking2 =
    {{ TiledArray::TiledRange1(blocking.begin(), blocking.end()),
        TiledArray::TiledRange1(blocking.begin(), blocking.end()) }};

  TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<2> >
    trange(blocking2.begin(), blocking2.end());

  TiledArray::Array<double, TiledArray::CoordinateSystem<2> > a(world, trange);
  TiledArray::Array<double, TiledArray::CoordinateSystem<2> > b(world, trange);
  TiledArray::Array<double, TiledArray::CoordinateSystem<2> > c(world, trange);
  a.set_all_local(1.0);
  b.set_all_local(1.0);
  c.set_all_local(0.0);
  std::cout << a.size() << std::endl;

  double avg_time = 0.0;
  for(int i = 0; i < 5; ++i) {
    const double start = madness::wall_time();
    c("m,n") = a("m,i") * b("i,n");
    world.gop.fence();
    const double stop = madness::wall_time();

    if(world.rank() == 0)
      std::cout << "Iteration: " << i << " time = " << stop - start << "\n";
    avg_time += stop - start;
  }

  if(world.rank() == 0)
    std::cout << "Average time = " << avg_time * 0.2 << "\nAverge GFLOPS ="
        << 2.0 * double(c.trange().elements().volume() * a.trange().elements().size()[1]) / (avg_time * 0.2) / 1000000000.0 << "\n";



  madness::finalize();
  return 0;
}
