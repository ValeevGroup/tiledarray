#include <iostream>
#include <tiled_array.h>
#include <sys/resource.h>
#include <TiledArray/epik.h>

#define MATRIX_SIZE 8192

double cpu_time() {
  rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);

  return r_usage.ru_utime.tv_sec + 1e-6*r_usage.ru_utime.tv_usec + r_usage.ru_stime.tv_sec + 1e-6*r_usage.ru_stime.tv_usec;
}

void ta_dgemm(madness::World& world, const std::size_t block_size) {
  EPIK_USER_REG(ta_dgemm_iteration,"TiledArray DGEMM iteration loop");
  const std::size_t size = MATRIX_SIZE;
  const std::size_t num_blocks = size / block_size;

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
  if(world.rank() == 0)
    std::cout << "Number of blocks = " << a.trange().tiles().volume() << std::endl;

  double avg_wall_time = 0.0;
  double avg_cpu_time = 0.0;
  double avg_efficiency = 0.0;
  for(int i = 0; i < 5; ++i) {
    EPIK_USER_START(ta_dgemm_iteration);
    const double wall_time_start = madness::wall_time();
    const double cpu_time_start = cpu_time();
//    madness::RMI::set_debug(true);
    c("m,n") = a("m,i") * b("i,n");
//    madness::RMI::set_debug(false);

    world.gop.fence();

    EPIK_USER_END(ta_dgemm_iteration);
    const double wall_time_stop = madness::wall_time();
    const double cpu_time_stop = cpu_time();

    double times[2] = { wall_time_stop - wall_time_start,  cpu_time_stop - cpu_time_start };
    world.gop.reduce(times,2,std::plus<double>());
    times[0] /= world.size();
    times[1] /= world.size();
    avg_wall_time += times[0];
    avg_cpu_time += times[1];
    avg_efficiency += times[1] / times[0] / madness::ThreadPool::size();
    if(world.rank() == 0) {
      std::cout << "Iteration " << i << ": wall time = " << times[0]
        << ", cpu time = " << times[1] << ", efficiency = "
        << times[1] / times[0] / madness::ThreadPool::size() << "\n";
    }
  }

  if(world.rank() == 0)
    std::cout << "Average wall time = " << 0.2 * avg_wall_time
        << ", Average cpu time = " << 0.2 * avg_cpu_time
        << ", Average efficiency = " << 0.2 * avg_efficiency
        << "\nAverage GFLOPS =" << 2.0 * double(size * size * size) / (avg_wall_time * 0.2) / 1000000000.0 << "\n";

}

void eigen_dgemm(madness::World& world) {
  const std::size_t size = MATRIX_SIZE;

  if(world.rank() == 0) {
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
      const double wall_time_start = madness::wall_time();
      const double cpu_time_start = cpu_time();
      c.noalias() += a * b;
      const double wall_time_stop = madness::wall_time();
      const double cpu_time_stop = cpu_time();

      const double wall_time = wall_time_stop - wall_time_start;
      const double cpu_time = cpu_time_stop - cpu_time_start;
      std::cout << "Iteration " << i << ": wall time = " << wall_time
        << ", cpu time = " << cpu_time << ", efficiency = "
        << cpu_time / wall_time / madness::ThreadPool::size() << "\n";
      avg_time += cpu_time;
    }

    std::cout << "Average time = " << avg_time * 0.2 << "\nAverge GFLOPS ="
        << 2.0 * double(size * size * size) / (avg_time * 0.2) / 1000000000.0 << "\n";
  }
}

#ifdef TILEDARRAY_HAS_CBLAS
void blas_dgemm(madness::World& world) {
  const std::size_t size = MATRIX_SIZE;

  if(world.rank() == 0) {
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
      const double wall_time_start = madness::wall_time();
      const double cpu_time_start = cpu_time();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0,
          a, size, b, size, 0.0, c, size);
      const double wall_time_stop = madness::wall_time();
      const double cpu_time_stop = cpu_time();

      const double wall_time = wall_time_stop - wall_time_start;
      const double cpu_time = cpu_time_stop - cpu_time_start;
      std::cout << "Iteration " << i << ": wall time = " << wall_time
        << ", cpu time = " << cpu_time << ", efficiency = " << cpu_time / wall_time / madness::ThreadPool::size() << "\n";
      avg_time += cpu_time;
    }

    std::cout << "Average time = " << avg_time * 0.2 << "\nAverge GFLOPS ="
        << 2.0 * double(size * size * size) / (avg_time * 0.2) / 1000000000.0 << "\n";

    delete [] a;
    delete [] b;
    delete [] c;
  }
}
#endif // TILEDARRAY_HAS_CBLAS

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  madness::World world(SafeMPI::COMM_WORLD);
  EPIK_FUNC_START();

  if(world.rank() == 0)
    std::cout << "Number of nodes = " << world.size() << "\n";

  if(world.rank() == 0)
    std::cout << "TiledArray:" << std::endl;
//  ta_dgemm(world, 32);
//  ta_dgemm(world, 64);
//  ta_dgemm(world, 128);
//  ta_dgemm(world, 256);
  ta_dgemm(world, 512);
//  ta_dgemm(world, 1024);

//  if(world.rank() == 0) {
//    std::cout << "Eigen:" << std::endl;
//    eigen_dgemm(world);
//  }
#ifdef TILEDARRAY_HAS_CBLAS
//  if(world.rank() == 0) {
//    std::cout << "Blas:" << std::endl;
//    blas_dgemm(world);
//  }
#endif // TILEDARRAY_HAS_CBLAS

  EPIK_FUNC_END();
  madness::finalize();
  return 0;
}
