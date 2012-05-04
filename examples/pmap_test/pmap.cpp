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

void print(std::size_t m, std::size_t n, const std::vector<ProcessID>& map) {
  for(std::size_t i = 0ul; i < m; ++i) {
    for(std::size_t j = 0ul; j < n; ++j)
      std::cout << map[i * n + j] << " ";
    std::cout << "\n";
  }
}

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  {
    madness::World world(MPI::COMM_WORLD);

    if(world.rank() == 0) {
      std::size_t m = 20;
      std::size_t n = 20;

      std::shared_ptr<TiledArray::Pmap<std::size_t> > blocked_pmap(new TiledArray::detail::BlockedPmap(world, m * n));
      std::vector<ProcessID> blocked_map = make_map(m, n, blocked_pmap);
      std::cout << "Block\n";
      print(m, n, blocked_map);



      std::shared_ptr<TiledArray::Pmap<std::size_t> > cyclic_pmap(new TiledArray::detail::CyclicPmap(world, m, n));
      std::vector<ProcessID> cyclic_map = make_map(m, n, cyclic_pmap);
      std::cout << "\n\nCyclic\n";
      print(m, n, cyclic_map);

      std::shared_ptr<TiledArray::Pmap<std::size_t> > hash_pmap(new TiledArray::detail::HashPmap(world, m * n));
      std::vector<ProcessID> hash_map = make_map(m, n, hash_pmap);
      std::cout << "\n\nHash\n";
      print(m, n, hash_map);
    }
  }
  madness::finalize();

  return 0;
}
