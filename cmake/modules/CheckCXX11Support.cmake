
include(CheckCXXSourceCompiles)
include(AppendFlags)

set(STD_CXX11_TEST_CODE "
#if (__cplusplus < 201103L) && ! defined(__INTEL_CXX11_MODE__)
#error C++11 support not enabled.
#endif

#include <functional>
#include <tuple>
#include <array>
#include <type_traits>
#include <memory>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <initializer_list>

template <typename T>
struct check
{
  static_assert(sizeof(int) <= sizeof(T), \"not big enough\");
};

struct Base {
  virtual void f() {}
};

struct Child : public Base {
  virtual void f() override {}
};

typedef check<check<bool>> right_angle_brackets;

int a;
decltype(a) b;

typedef check<int> check_type;
check_type c;
check_type&& cr = static_cast<check_type&&>(c);

auto d = a;
auto l = [](){};


typedef std::is_same<int, double> same_type;


inline int sum() { return 0; }

template <typename T, typename... U> 
inline int sum(const T& t, const U&... u) {
  return t + sum(u...);
}

template <typename... args>
void print_numbers(const args&... numbers) {
  constexpr std::size_t n = sizeof...(args);
  int nums[n] = { static_cast<int>(numbers)... };
  std::copy(nums, nums+n, std::ostream_iterator<int>(std::cout, \" \"));
}

int main(int, char**){

  std::tuple<int, double, unsigned long> t(1, 2.0, 3ul);

  std::hash<int> h;
  int h1 = h(1);

  std::array<int,10> aint10 = { 0,1,2,3,4,5,6,7,8,9 };
  
  std::shared_ptr<int> sptr = std::make_shared<int>(1);
  std::unique_ptr<int> uptr{new int(2)};

  const int s123 = sum(1, 2, 3);
  
  print_numbers(-1);
  print_numbers(0, 1, 2, 3, 4);

  return 0; 
}
")

macro(check_cxx11_support outvar)

  # Check for default C++11 support
  check_cxx_source_compiles("${STD_CXX11_TEST_CODE}" ${outvar})
  
  if(NOT ${outvar})
    # Check for and add, ADD additional test flags as necessary
    foreach(_cxx11_test_flag "-std=c++11" "-std=c++0x")
      cmake_push_check_state()
      
      set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${_cxx11_test_flag}")
      
      # Check for default C++11 support with _cxx11_test_flag
      unset(${outvar})
      check_cxx_source_compiles("${STD_CXX11_TEST_CODE}" ${outvar})
      
      cmake_pop_check_state()
      
      if(${outvar})
        append_flags(CMAKE_CXX_FLAGS "${_cxx11_test_flag}")
        break()
      endif()
      
    endforeach()
  endif()

endmacro(check_cxx11_support)