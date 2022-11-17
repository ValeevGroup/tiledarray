// Samuel R. Powell, 2021
#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/string.h"
#include "TiledArray/util/annotation.h"

namespace Einsum::index {

std::vector<std::string> validate(const std::vector<std::string> &v) {
  return v;
}

small_vector<std::string> tokenize(const std::string &s) {
  // std::vector<std::string> r;
  // boost::split(r, s, boost::is_any_of(", \t"));
  // return r;
  auto r = TiledArray::detail::tokenize_index(s, ',');
  if (r == std::vector<std::string>{""}) return {};
  return small_vector<std::string> (r.begin(), r.end()); // correct?
}

std::string join(const small_vector<std::string> &v) {
  return string::join(v, ",");
}

}  // namespace TiledArray::Einsum::index
