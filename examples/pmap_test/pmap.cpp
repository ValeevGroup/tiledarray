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

void print_map(std::size_t m, std::size_t n, const std::vector<ProcessID>& map) {
  for(std::size_t i = 0ul; i < m; ++i) {
    for(std::size_t j = 0ul; j < n; ++j)
      std::cout << map[i * n + j] << " ";
    std::cout << "\n";
  }
}

void print_local(madness::World& world, const std::shared_ptr<TiledArray::Pmap<std::size_t> >& pmap) {
  for(ProcessID r = 0; r < world.size(); ++r) {
    world.gop.fence();
    if(r == world.rank()) {
      std::cout << r << ": { ";
      for(TiledArray::Pmap<std::size_t>::const_iterator it = pmap->begin(); it != pmap->end(); ++it)
        std::cout << *it << " ";
      std::cout << "}\n";
    }
  }
}

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  {
    madness::World world(MPI::COMM_WORLD);

    std::size_t m = 20;
    std::size_t n = 10;

    std::shared_ptr<TiledArray::Pmap<std::size_t> > blocked_pmap(new TiledArray::detail::BlockedPmap(world, m * n));
    blocked_pmap->set_seed(0ul);
    std::vector<ProcessID> blocked_map = make_map(m, n, blocked_pmap);

    std::shared_ptr<TiledArray::Pmap<std::size_t> > cyclic_pmap(new TiledArray::detail::CyclicPmap(world, m, n));
    cyclic_pmap->set_seed(0ul);
    std::vector<ProcessID> cyclic_map = make_map(m, n, cyclic_pmap);

    std::shared_ptr<TiledArray::Pmap<std::size_t> > hash_pmap(new TiledArray::detail::HashPmap(world, m * n));
    hash_pmap->set_seed(0ul);
    std::vector<ProcessID> hash_map = make_map(m, n, hash_pmap);

    if(world.rank() == 0) {
      std::cout << "Block\n";
      print_map(m, n, blocked_map);
      std::cout << "\n";
    }

    print_local(world, blocked_pmap);

    world.gop.fence();

    if(world.rank() == 0) {

      std::cout << "\n\nCyclic\n";
      print_map(m, n, cyclic_map);
      std::cout << "\n";
    }

    print_local(world, cyclic_pmap);

    world.gop.fence();

    if(world.rank() == 0) {
      std::cout << "\n\nHash\n";
      print_map(m, n, hash_map);
      std::cout << "\n";
    }

    print_local(world, hash_pmap);
  }
  madness::finalize();

  return 0;
}
