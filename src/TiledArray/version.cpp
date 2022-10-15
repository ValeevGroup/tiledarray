//
// Created by Eduard Valeyev on 10/14/22.
//

#include <TiledArray/version.h>

namespace TiledArray {

const char* revision() noexcept {
  static const char revision[] = TILEDARRAY_REVISION;
  return revision;
}

}  // namespace TiledArray
