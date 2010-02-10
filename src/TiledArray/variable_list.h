#ifndef TILEDARRAY_VARIABLE_LIST_H__INCLUDED
#define TILEDARRAY_VARIABLE_LIST_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/permutation.h>
//#include <TiledArray/coordinate_system.h>
//#include <boost/iterator/transform_iterator.hpp>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iosfwd>

namespace TiledArray {

  template<unsigned int DIM>
  class Permutation;
  template <unsigned int DIM, typename T>
  std::vector<T> operator^(const Permutation<DIM>&, const std::vector<T>&);
  template <unsigned int DIM, typename T>
  std::vector<T> operator^=(std::vector<T>&, const Permutation<DIM>&);

  namespace expressions {

    class VariableList;
    template<unsigned int DIM>
    VariableList operator ^(const ::TiledArray::Permutation<DIM>&, const VariableList&);
    void swap(VariableList&, VariableList&);
    template<typename InIter1, typename InIter2>
    void find_common(InIter1, const InIter1, InIter2, const InIter2,
        std::pair<InIter1, InIter1>&, std::pair<InIter2, InIter2>&);

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
        if(vars.size() != 0)
          init_(vars);
      }

      template<typename InIter>
      VariableList(InIter first, InIter last) {
        BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
        TA_ASSERT( unique_(first, last), std::runtime_error,
            "Duplicate variable names not allowed.");

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

      template<unsigned int DIM>
      VariableList& operator ^=(const Permutation<DIM>& p) {
        TA_ASSERT(DIM == dim(), std::runtime_error,
            "The permutation dimensions are not equal to the variable list dimensions.");
        vars_ ^= p;
        return *this;
      }

      /// Returns an iterator to the first variable.
      const_iterator begin() const { return vars_.begin(); }

      /// Returns an iterator to the end of the variable list.
      const_iterator end() const { return vars_.end(); }

      /// Returns the n-th string in the variable list.
      const std::string& at(const std::size_t n) const { return vars_.at(n); }

      /// Returns the n-th string in the variable list.
      const std::string& operator [](const std::size_t n) const { return vars_[n]; }

      /// Returns the number of strings in the variable list.
      unsigned int dim() const { return vars_.size(); }

      const std::vector<std::string>& data() const { return vars_; }

      std::string string() const {
        std::string result;
        std::vector<std::string>::const_iterator it = vars_.begin();
        if(it == vars_.end())
          return result;

        for(result = *it; it != vars_.end(); ++it) {
          result += "," + *it;
        }

        return result;
      }

      void swap(VariableList& other) {
        std::swap(vars_, other.vars_);
      }

      std::vector<std::size_t> permutation(const VariableList& other) {
        TA_ASSERT(dim() == other.dim(), std::runtime_error,
            "The variable list dimensions are not equal.");
        std::vector<std::size_t> p(dim(), 0);
        const_iterator other_it;
        const_iterator this_it = begin();
        for(std::vector<std::size_t>::iterator it = p.begin(); it != p.end(); ++it, ++this_it) {
          other_it = std::find(other.begin(), other.end(), *this_it);
          TA_ASSERT(other_it != other.end(), std::runtime_error,
              "Variable name not found in other variable list.");
          *it = std::distance(other.begin(), other_it);
        }

        return p;
      }

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

        TA_ASSERT( (unique_(vars_.begin(), vars_.end())), std::runtime_error,
            "Duplicate variables not allowed in variable list.");
      }

      /// Returns a string with all the spaces ( ' ' ) removed from the string
      /// defined by the start and finish iterators.
      static std::string trim_spaces_(std::string::const_iterator first, std::string::const_iterator last) {
        TA_ASSERT( (first != last), std::runtime_error,
            "Zero length variable string not allowed.");
        std::string result = "";
        for(;first != last; ++first) {
          TA_ASSERT( valid_char_(*first), std::runtime_error,
              "Variable names may only contain letters and numbers.");
          if(*first != ' ' && *first != '\0')
            result.append(1, *first);
        }

        TA_ASSERT( (result.length() != 0) , std::runtime_error,
            "Blank variable string not allowed.");

        return result;
      }

      /// Returns true if all vars contained by the list are unique.
      template<typename InIter>
      bool unique_(InIter first, InIter last) const {
        for(; first != last; ++first) {
          InIter it2 = first;
          for(++it2; it2 != last; ++it2)
            if(first->compare(*it2) == 0)
              return false;
        }

        return true;
      }

      static bool valid_char_(char c) {
        return (c >= 'a' && c < 'z') || (c >= 'A' && c < 'Z') ||
            (c >= '0' && c < '9') || (c == ' ') || (c == ',') || (c == '\0');
      }

      friend void swap(VariableList&, VariableList&);

      std::vector<std::string> vars_;

      template<unsigned int DIM>
      friend VariableList operator ^(const ::TiledArray::Permutation<DIM>&, const VariableList&);

    }; // class VariableList

    /// Exchange the content of the two variable lists.
    inline void swap(VariableList& v0, VariableList& v1) {
      std::swap(v0.vars_, v1.vars_);
    }

    template<typename InIter1, typename InIter2>
    void find_common(InIter1 first1, const InIter1 last1, InIter2 first2, const InIter2 last2,
        std::pair<InIter1, InIter1>& common1, std::pair<InIter2, InIter2>& common2)
    {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter1>::value);
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter2>::value);
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
      while((first1 != last1) && (first2 != last2)) {
        if(!std::equal(first1->begin(), first1->end(), first2->begin()))
          break;
        ++first1;
        ++first2;
      }
      common1.second = first1;
      common2.second = first2;
    }

    inline bool operator ==(const VariableList& v0, const VariableList& v1) {
      return (v0.dim() == v1.dim()) && std::equal(v0.begin(), v0.end(), v1.begin());
    }

    inline bool operator !=(const VariableList& v0, const VariableList& v1) {
      return ! operator ==(v0, v1);
    }

    template<unsigned int DIM>
    VariableList operator ^(const ::TiledArray::Permutation<DIM>& p, const VariableList& v) {
      VariableList result;
      result.vars_ = p ^ v.vars_;

      return result;
    }

    /// ostream VariableList output orperator.
    inline std::ostream& operator <<(std::ostream& out, const VariableList& v) {
      out << "(";
      std::size_t d;
      std::size_t n = v.dim() - 1;
      for(d = 0; d < n; ++d) {
        out << v[d] << ", ";
      }
      out << v[d];
      out << ")";
      return out;
    }

  } // namespace expressions

} // namespace TiledArray

namespace madness {
  namespace archive {

    template <class Archive, typename T>
    struct ArchiveLoadImpl;

    template <class Archive, typename T>
    struct ArchiveStoreImpl;

    template <class Archive>
    struct ArchiveLoadImpl<Archive, TiledArray::expressions::VariableList > {
      static void load(const Archive& ar, TiledArray::expressions::VariableList& v) {
        std::string s;
        ar & s;
        v = s;
      }
    }; // struct ArchiveLoadImpl<Archive, TiledArray::expressions::VariableList >

    template <class Archive>
    struct ArchiveStoreImpl<Archive, TiledArray::expressions::VariableList > {
      static void store(const Archive& ar, const TiledArray::expressions::VariableList& v) {
        ar & v.string();
      }
    }; // struct ArchiveLoadImpl<Archive, TiledArray::expressions::VariableList >

  } // namespace archive
} // namespace madness


// Add specializations of math functors.
namespace std {

  template<>
  struct plus< ::TiledArray::expressions::VariableList> : public binary_function <
      ::TiledArray::expressions::VariableList,::TiledArray::expressions::VariableList,
      ::TiledArray::expressions::VariableList>
  {
    const ::TiledArray::expressions::VariableList& operator() (
        const ::TiledArray::expressions::VariableList& v0,
        const ::TiledArray::expressions::VariableList& v1) const
    {
      TA_ASSERT(v0 == v1, std::runtime_error,
          "variable lists must be identical for addition operations.");

      return v0;
    }
  };

  template<>
  struct minus< ::TiledArray::expressions::VariableList> : public binary_function <
      ::TiledArray::expressions::VariableList,::TiledArray::expressions::VariableList,
      ::TiledArray::expressions::VariableList>
  {
    const ::TiledArray::expressions::VariableList& operator() (
        const ::TiledArray::expressions::VariableList& v0,
        const ::TiledArray::expressions::VariableList& v1) const
    {
      TA_ASSERT(v0 == v1, std::runtime_error,
          "variable lists must be identical for addition operations.");

      return v0;
    }
  };

  template<>
  struct multiplies< ::TiledArray::expressions::VariableList> : public binary_function <
      ::TiledArray::expressions::VariableList,::TiledArray::expressions::VariableList,
      ::TiledArray::expressions::VariableList>
  {
    const ::TiledArray::expressions::VariableList operator() (
        const ::TiledArray::expressions::VariableList& v0,
        const ::TiledArray::expressions::VariableList& v1) const
    {
      typedef ::TiledArray::expressions::VariableList::const_iterator iterator;
      typedef std::pair<iterator, iterator> it_pair;

      it_pair c0(v0.end(), v0.end());
      it_pair c1(v1.end(), v1.end());
      ::TiledArray::expressions::find_common(v0.begin(), v0.end(), v1.begin(), v1.end(), c0, c1);

      std::size_t n0 = 2 * v0.dim() + 1;
      std::size_t n1 = 2 * v1.dim();

      std::map<std::size_t, std::string> v;
      std::pair<std::size_t, std::string> p;
      for(iterator it = v0.begin(); it != c0.first; ++it, n0 -= 2) {
        p.first = n0;
        p.second = *it;
        v.insert(p);
      }
      for(iterator it = v1.begin(); it != c1.first; ++it, n1 -= 2) {
        p.first = n1;
        p.second = *it;
        v.insert(p);
      }
      n0 -= 2 * (c0.second - c0.first);
      n1 -= 2 * (c1.second - c1.first);
      for(iterator it = c0.second; it != v0.end(); ++it, n0 -= 2) {
        p.first = n0;
        p.second = *it;
        v.insert(p);
      }
      for(iterator it = c1.second; it != v1.end(); ++it, n1 -= 2) {
        p.first = n1;
        p.second = *it;
        v.insert(p);
      }

      std::vector<std::string> result;
      for(std::map<std::size_t, std::string>::reverse_iterator it = v.rbegin(); it != v.rend(); ++it)
        result.push_back(it->second);

      return ::TiledArray::expressions::VariableList(result.begin(), result.end());
    }
  };

} // namespace std

#endif // TILEDARRAY_VARIABLE_LIST_H__INCLUDED
