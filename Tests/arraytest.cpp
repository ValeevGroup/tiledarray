#include <iostream>
#include <array.h>

using namespace TiledArray;

void ArrayTest(int argc, char* argv[]) {
  
  // Create an orthotope that will be used for shape tests.
  // Test with C-style Range Array constructor.
  Orthotope<3>::index_t dim0[] = {0,4,6};
  Orthotope<3>::index_t dim1[] = {0,4,6,9};
  Orthotope<3>::index_t dim2[] = {0,4,6,9,10};
  Orthotope<3>::index_t tiles[3] = {2,3,4};
  const size_t* dim_set[] = {dim0,dim1,dim2};
  Orthotope<3> ortho3(dim_set, tiles);
  boost::shared_ptr< AbstractShape<3> > shp0(new Shape<3, OffTupleFilter<3> >(&ortho3));
  
  std::cout << "Orthotope used in shape tests." << std::endl;
  std::cout << "ortho3 = " << ortho3 << std::endl;
  std::cout << "Constructed shp" << std::endl << "shp = " << *shp0 << std::endl;
  
  MPI::Init(argc, argv);
  DistributedRuntime runtime(MPI::COMM_WORLD);
  Array<double,3> a(runtime,shp0);
  std::cout << "Constructed a = Array<double,3>" << std::endl;
  
  
  MPI::Finalize();
  
}
