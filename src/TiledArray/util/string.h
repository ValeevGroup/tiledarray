//
// Created by Samuel R. Powell on 4/15/21.
//
#pragma once

#ifndef TILEDARRAY_STRING_H
#define TILEDARRAY_STRING_H

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/join.hpp>
#include <string>
#include <vector>

namespace TiledArray::string {

  // Split delimiter must match completely
  std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> res;
    return boost::split(res, s, [&d](char c) { return c == d; } /*boost::is_any_of(d)*/);
  }

  std::string trim(const std::string& s) {
    return boost::trim_copy(s);
  }

  template<typename T>
  std::string join(const T &s, const std::string& j = "") {
    return boost::join(s, j);
  }

  template <typename T>
  std::string str(const T& obj) {
    std::stringstream ss;
    ss << obj;
    return ss.str();
  }

}

#endif //TILEDARRAY_STRING_H
