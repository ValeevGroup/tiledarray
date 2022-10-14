/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "TiledArray/pmap/cyclic_pmap.h"
#include "TiledArray/pmap/hash_pmap.h"
#include "tiledarray.h"

std::vector<ProcessID> make_map(std::size_t m, std::size_t n,
                                std::shared_ptr<TiledArray::Pmap>& pmap) {
  std::vector<ProcessID> map;

  const std::size_t end = m * n;
  map.reserve(end);
  for (std::size_t i = 0ul; i < end; ++i) map.push_back(pmap->owner(i));

  return map;
}

void print_map(std::size_t m, std::size_t n,
               const std::vector<ProcessID>& map) {
  for (std::size_t i = 0ul; i < m; ++i) {
    for (std::size_t j = 0ul; j < n; ++j) std::cout << map[i * n + j] << " ";
    std::cout << "\n";
  }
}

void print_local(TiledArray::World& world,
                 const std::shared_ptr<TiledArray::Pmap>& pmap) {
  for (ProcessID r = 0; r < world.size(); ++r) {
    world.gop.fence();
    if (r == world.rank()) {
      std::cout << r << ": { ";
      for (TiledArray::Pmap::const_iterator it = pmap->begin();
           it != pmap->end(); ++it)
        std::cout << *it << " ";
      std::cout << "}\n";
    }
  }
}

int main(int argc, char** argv) {
  TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  std::size_t m = 20;
  std::size_t n = 10;
  std::size_t M = 200;
  std::size_t N = 100;

  std::shared_ptr<TiledArray::Pmap> blocked_pmap(
      new TiledArray::detail::BlockedPmap(world, m * n));
  std::vector<ProcessID> blocked_map = make_map(m, n, blocked_pmap);

  std::shared_ptr<TiledArray::Pmap> cyclic_pmap(
      new TiledArray::detail::CyclicPmap(world, m, n, M, N));
  std::vector<ProcessID> cyclic_map = make_map(m, n, cyclic_pmap);

  std::shared_ptr<TiledArray::Pmap> hash_pmap(
      new TiledArray::detail::HashPmap(world, m * n));
  std::vector<ProcessID> hash_map = make_map(m, n, hash_pmap);

  if (world.rank() == 0) {
    std::cout << "Block\n";
    print_map(m, n, blocked_map);
    std::cout << "\n";
  }

  print_local(world, blocked_pmap);

  world.gop.fence();

  if (world.rank() == 0) {
    std::cout << "\n\nCyclic\n";
    print_map(m, n, cyclic_map);
    std::cout << "\n";
  }

  print_local(world, cyclic_pmap);

  world.gop.fence();

  if (world.rank() == 0) {
    std::cout << "\n\nHash\n";
    print_map(m, n, hash_map);
    std::cout << "\n";
  }

  print_local(world, hash_pmap);

  return 0;
}
