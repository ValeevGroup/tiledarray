#include <iostream>
#include <tiled_array.h>
#ifdef __INTEL_COMPILER
#include <mkl.h>
#else
#include <cblas.h>
#endif

#define MEMORY 1000000000

void ta_dgemm(madness::World& world, const std::size_t block_size) {
  const std::size_t num_blocks = 4096 / block_size;
  const std::size_t size = 4096;

  if(world.rank() == 0)
    std::cout << "Matrix size = " << size << "x" << size << "\n"
        << "Memory per matrix = " << double(size * size * sizeof(double)) / 1000000000.0 << "GB" << std::endl;

  std::vector<unsigned int> blocking;
  blocking.reserve(num_blocks + 1);
  blocking.push_back(0);
  for(std::size_t i = 1; i <= num_blocks; ++i)
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
  std::cout << "Number of blocks = " << a.trange().tiles().volume() << std::endl;

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
        << 2.0 * double(size * size * size) / (avg_time * 0.2) / 1000000000.0 << "\n";

}

void eigen_dgemm(madness::World& world) {
  const std::size_t size = 4096;

  if(world.rank() == 0)
    std::cout << "Eigen instruction set: " << Eigen::SimdInstructionSetsInUse()
        << "\nMatrix size = " << size << "x" << size << "\n"
        << "Memory per matrix = " << double(size * size * sizeof(double)) / 1000000000.0 << "GB" << std::endl;

  Eigen::MatrixXd a(size, size);
  Eigen::MatrixXd b(size, size);
  Eigen::MatrixXd c(size, size);
  a.fill(1.0);
  b.fill(1.0);
  c.fill(0.0);

  double avg_time = 0.0;
  for(int i = 0; i < 5; ++i) {
    const double start = madness::wall_time();
    c.noalias() += a * b;
    const double stop = madness::wall_time();

    if(world.rank() == 0)
      std::cout << "Iteration: " << i << " time = " << stop - start << "\n";
    avg_time += stop - start;
  }

  if(world.rank() == 0)
    std::cout << "Average time = " << avg_time * 0.2 << "\nAverge GFLOPS ="
        << 2.0 * double(size * size * size) / (avg_time * 0.2) / 1000000000.0 << "\n";

}

void blas_dgemm(madness::World& world) {
  const std::size_t size = 4096;

  if(world.rank() == 0)
    std::cout << "Matrix size = " << size << "x" << size << "\n"
        << "Memory per matrix = " << double(size * size * sizeof(double)) / 1000000000.0 << "GB" << std::endl;

  double* a = new double[size * size];
  double* b = new double[size * size];
  double* c = new double[size * size];
  std::fill_n(a, size * size, 1.0);
  std::fill_n(b, size * size, 1.0);
  std::fill_n(c, size * size, 0.0);

  double avg_time = 0.0;
  for(int i = 0; i < 5; ++i) {
    const double start = madness::wall_time();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0,
        a, size, b, size, 0.0, c, size);
    const double stop = madness::wall_time();

    if(world.rank() == 0)
      std::cout << "Iteration: " << i << " time = " << stop - start << "\n";
    avg_time += stop - start;
  }

  if(world.rank() == 0)
    std::cout << "Average time = " << avg_time * 0.2 << "\nAverge GFLOPS ="
        << 2.0 * double(size * size * size) / (avg_time * 0.2) / 1000000000.0 << "\n";

}

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  madness::World world(MPI::COMM_WORLD);

  if(world.rank() == 0)
    std::cout << "TiledArray:" << std::endl;
  ta_dgemm(world, 32);
  ta_dgemm(world, 64);
  ta_dgemm(world, 128);
  ta_dgemm(world, 256);
  ta_dgemm(world, 512);
  ta_dgemm(world, 1024);

  if(world.rank() == 0) {
    std::cout << "Eigen:" << std::endl;
    eigen_dgemm(world);
  }

  if(world.rank() == 0) {
    std::cout << "Blas:" << std::endl;
    blas_dgemm(world);
  }

  madness::finalize();
  return 0;
}
