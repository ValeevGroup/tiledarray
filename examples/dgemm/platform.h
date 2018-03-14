//
// Created by Eduard Valeyev on 3/13/18.
//

#ifndef TILEDARRAY_PLATFORM_H
#define TILEDARRAY_PLATFORM_H

namespace TiledArray {

/// enumerates the memory spaces
enum class MemorySpace {
  // MemorySpace is represented as a bitfield to compute unions and intersections easier
  Null = 0b00,
  CPU = 0b01,
  CUDA = 0b10,
  CUDA_UM = CPU | CUDA  // union of CPU and CUDA spaces
};

// customization point: in_memory_space<S>(O) -> bool
// it can be used to query if object O is in space S

/// @return intersection of @c space1 and @c space2
constexpr MemorySpace operator&(MemorySpace space1, MemorySpace space2) {
  return static_cast<MemorySpace>( static_cast<int>(space1) & static_cast<int>(space2) );
}
/// @return union of @c space1 and @c space2
constexpr MemorySpace operator|(MemorySpace space1, MemorySpace space2) {
  return static_cast<MemorySpace>( static_cast<int>(space1) | static_cast<int>(space2) );
}
/// @return true if intersection of @c space1 and @c space2 is nonnull
constexpr bool overlap(MemorySpace space1, MemorySpace space2) {
  return (space1 & space2) != MemorySpace::Null;
}

/// enumerates the execution spaces
enum class ExecutionSpace {
  Null = 0,
  CPU,
  CUDA
};

}  // namespace TiledArray

#endif //TILEDARRAY_PLATFORM_H
