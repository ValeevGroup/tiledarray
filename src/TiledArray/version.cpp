//
// Created by Eduard Valeyev on 10/14/22.
//

#include <TiledArray/version.h>

namespace TiledArray {

const char* git_revision() noexcept {
  static const char revision[] = TILEDARRAY_GIT_REVISION;
  return revision;
}

const char* git_description() noexcept {
  static const char revision[] = TILEDARRAY_GIT_DESCRIPTION;
  return revision;
}

}  // namespace TiledArray
