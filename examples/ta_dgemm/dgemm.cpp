#include <iostream>
#include <tiled_array.h>

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  madness::World world(MPI::COMM_WORLD);

  std::array<unsigned int, 901> blocking;
  blocking.front() = 0;
  for(std::size_t i = 1; i < 901; ++i)
    blocking[i] = blocking[i-1] + 16;

  std::array<TiledArray::TiledRange1, 2> blocking2 =
    {{ TiledArray::TiledRange1(blocking.begin(), blocking.end()),
        TiledArray::TiledRange1(blocking.begin(), blocking.end()) }};

  TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<2> >
    trange(blocking2.begin(), blocking2.end());

  TiledArray::Array<double, TiledArray::CoordinateSystem<4> > a(world, trange);
  TiledArray::Array<double, TiledArray::CoordinateSystem<4> > b(world, trange);
  TiledArray::Array<double, TiledArray::CoordinateSystem<4> > c(world, trange);
  a.set_all_local(1.0);
  b.set_all_local(1.0);
  c.set_all_local(0.0);

  double avg_time = 0.0;
  for(int i = 0; i < 5; ++i) {
    const double start = madness::wall_time();
    c("m,n") = c("m,n") + a("m,i") * b("i,n");
    const double stop = madness::wall_time();

    if(world.rank() == 0)
      std::cout << "Iteration: " << i << " time = " << stop - start << "\n";
    avg_time += stop - start;
  }

  if(world.rank() == 0)
    std::cout << "Average time = " << avg_time / 5.0 << "\n";

  madness::finalize();
  return 0;
}
