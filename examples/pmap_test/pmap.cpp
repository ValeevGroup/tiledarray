#include <tiled_array.h>
#include <TiledArray/hash_pmap.h>

std::vector<ProcessID> make_map(std::size_t m, std::size_t n, std::shared_ptr<TiledArray::Pmap<std::size_t> >& pmap) {
  std::vector<ProcessID> map;

  const std::size_t end = m * n;
  map.reserve(end);
  for(std::size_t i = 0ul; i < end; ++i)
    map.push_back(pmap->owner(i));

  return map;
}

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  {
    madness::World world(MPI::COMM_WORLD);

    std::size_t m = 20;
    std::size_t n = 20;

    std::shared_ptr<TiledArray::Pmap<std::size_t> > blockd_pmap(new TiledArray::detail::BlockedPmap(world, m * n));


    std::shared_ptr<TiledArray::Pmap<std::size_t> > cyclic_pmap(new TiledArray::detail::CyclicPmap(world, m, n));


    std::shared_ptr<TiledArray::Pmap<std::size_t> > hash_pmap(new TiledArray::detail::HashPmap(world, m * n));

  }
  madness::finalize();

  return 0;
}
