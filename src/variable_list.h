#ifndef TILEDARRAY_VARIABLE_LIST_H__INCLUDED
#define TILEDARRAY_VARIABLE_LIST_H__INCLUDED

#include <error.h>
#include <coordinate_system.h>
#include <vector>
#include <string>
#include <map>
#include <boost/iterator/transform_iterator.hpp>

namespace TiledArray {

  namespace detail {

    /// Variable list manages a list variable strings.

    /// Each variable is separated by commas. All spaces are ignored and removed
    /// from variable list. So, "a c" will be converted to "ac" and will be
    /// considered a single variable. All variables must be unique.
    class VariableList {
    public:
      typedef std::vector<std::string>::const_iterator const_iterator;

      /// Constructs an empty variable list.
      VariableList() : vars_() { }

      /// constructs a variable lists
      explicit VariableList(const std::string& vars) {
        init_(vars);
      }

      template<typename InIter>
      VariableList(InIter first, InIter last) {
        TA_ASSERT( unique(first, last),
            std::runtime_error("VariableList::VariableList(...): Duplicate variable names not allowed."));

        for(; first != last; ++first)
          vars_.push_back(trim_spaces_(first->begin(), first->end()));

      }

      VariableList(const VariableList& other) : vars_(other.vars_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      VariableList(VariableList&& other) : vars_(std::move(other.vars_)) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      VariableList& operator =(const VariableList& other) {
        vars_ = other.vars_;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      VariableList& operator =(VariableList&& other) {
        vars_ = std::move(other.vars_);

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      VariableList& operator =(const std::string& vars) {
        init_(vars);
        return *this;
      }

      /// Returns an iterator to the first variable.
      const_iterator begin() const { return vars_.begin(); }

      /// Returns an iterator to the end of the variable list.
      const_iterator end() const { return vars_.end(); }

      /// Returns the n-th string in the variable list.
      const std::string& get(const std::size_t n) const { return vars_.at(n); }

      /// Returns the number of strings in the variable list.
      std::size_t count() const { return vars_.size(); }


    private:

      /// Copies a comma separated list into a vector of strings. All spaces are
      /// removed from the sub-strings.
      void init_(const std::string& vars) {
        std::string::const_iterator start = vars.begin();
        std::string::const_iterator finish = vars.begin();
        for(; finish != vars.end(); ++finish) {
          if(*finish == ',') {
            vars_.push_back(trim_spaces_(start, finish));
            start = finish + 1;
          }
        }
        vars_.push_back(trim_spaces_(start, finish));

        TA_ASSERT( (unique(vars_.begin(), vars_.end())),
                  std::runtime_error("VariableList::init_(...): Duplicate variables not allowed in variable list.") );
      }

      /// Returns a string with all the spaces ( ' ' ) removed from the string
      /// defined by the start and finish iterators.
      static std::string trim_spaces_(std::string::const_iterator start, std::string::const_iterator finish) {
        TA_ASSERT( (start != finish),
            std::runtime_error("VariableList::trim_spaces_(...): Zero length variable string not allowed.") );
        std::string result = "";
        for(;start != finish; ++start)
          if(*start != ' ')
            result.append(1, *start);

        TA_ASSERT( (result.length() != 0) ,
            std::runtime_error("VariableList::trim_spaces_(...): Blank variable string not allowed.") );

        return result;
      }

      /// Returns true if all vars contained by the list are unique.
      template<typename InIter>
      bool unique(InIter first, InIter last) const {
        for(InIter it1 = first; it1 != last; ++it1) {
          InIter it2 = it1;
          for(++it2; it2 != last; ++it2)
            if(*it1 == *it2)
              return false;
        }

        return true;
      }

      std::vector<std::string> vars_;

    }; // class VariableList

    template<typename InIter1, typename InIter2>
    void find_common(InIter1 first1, const InIter1 last1, InIter2 first2, const InIter2 last2,
        std::pair<InIter1, InIter1>& common1, std::pair<InIter2, InIter2>& common2)
    {
      common1.first = last1;
      common1.second = last1;
      common2.first = last2;
      common2.second = last2;

      for(; first1 != last1; ++first1) {
        common2.first = std::find(first2, last2, *first1);
        if(common2.first != last2)
          break;
      }
      common1.first = first1;
      first2 = common2.first;
      while(*first1 == *last1 && first1 != last1 && first2 != last2) {
        ++first1;
        ++first2;
      }
      common1.second = first1;
      common2.second = first2;
    }

    inline bool operator ==(const VariableList& v0, const VariableList& v1) {
      return (v0.count() == v1.count()) && std::equal(v0.begin(), v0.end(), v1.begin());
    }

    inline bool operator !=(const VariableList& v0, const VariableList& v1) {
      return ! operator ==(v0, v1);
    }

    template<typename T, typename U>
    const U& pair_second(const std::pair<T,U>& p) {
      return p.second;
    }

    template<typename T, typename U>
    const T& pair_first(const std::pair<T,U>& p) {
      return p.first;
    }

  } // namespace detail

} // namespace TiledArray

// Add specializations of math functors.
namespace std {

  template<>
  struct multiplies< ::TiledArray::detail::VariableList> : binary_function <
      ::TiledArray::detail::VariableList,::TiledArray::detail::VariableList,
      ::TiledArray::detail::VariableList>
  {
    const ::TiledArray::detail::VariableList operator() (
        const ::TiledArray::detail::VariableList& v0,
        const ::TiledArray::detail::VariableList& v1) const
    {
      typedef ::TiledArray::detail::VariableList::const_iterator iterator;
      typedef std::pair<iterator, iterator> it_pair;

      it_pair c0(v0.end(), v0.end());
      it_pair c1(v1.end(), v1.end());
      ::TiledArray::detail::find_common(v0.begin(), v0.end(), v1.begin(), v1.end(), c0, c1);

      double n0 = v0.count() + 0.5;
      double n1 = v1.count();

      std::map<double, std::string> v;
      for(iterator it = v0.begin(); it != c0.first; ++it, n0 -= 1.0)
        v.insert(std::make_pair(n0, *it));
      for(iterator it = v1.begin(); it != c1.first; ++it, n1 -= 1.0)
        v.insert(std::make_pair(n1, *it));
      n0 -= c0.second - c0.first;
      n1 -= c1.second - c1.first;
      for(iterator it = c0.second; it != v0.end(); ++it, n0 -= 1.0)
        v.insert(std::make_pair(n0, *it));
      for(iterator it = c1.second; it != v1.end(); ++it, n1 -= 1.0)
        v.insert(std::make_pair(n1, *it));

      return ::TiledArray::detail::VariableList(
          boost::make_transform_iterator(v.rbegin(), ::TiledArray::detail::pair_second<double,std::string>),
          boost::make_transform_iterator(v.rend(), ::TiledArray::detail::pair_second<double,std::string>));
    }
  };

} // namespace std

#endif // TILEDARRAY_VARIABLE_LIST_H__INCLUDED
