#ifndef TILEDARRAY_VARIABLE_LIST_MATH_H__INCLUDED
#define TILEDARRAY_VARIABLE_LIST_MATH_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/math.h>

namespace TiledArray {


  namespace expressions {

    template <typename>
    class AnnotatedArray;

  } // namespace expressions

  namespace math {

    /// Default binary operation for \c VariableList objects

    /// \tparam The operation type to be performed on two data elements.
    template <typename ArrayType, template <typename> class Op>
    class BinaryOp<
        expressions::VariableList,
        expressions::AnnotatedArray<ArrayType>,
        expressions::AnnotatedArray<ArrayType>,
        Op, typename boost::disable_if<std::is_same<Op<int>, std::multiplies<int> > >::type >
    {
    public:
      typedef const expressions::AnnotatedArray<ArrayType>& first_argument_type;
      typedef const expressions::AnnotatedArray<ArrayType>& second_argument_type;
      typedef expressions::VariableList& result_type;
      typedef expressions::VariableList VarList;

      /// Set \c result to hold the resulting variable list for this operation

      /// The default behavior for this operation is to set copy left into
      /// result.
      /// \param result The result variable list
      /// \param left The variable list for the left-hand argument.
      /// \param right The variable list for the right-hand argument.
      /// \throw std::runtime_error When \c left is not equal to \c right.
      result_type operator ()(result_type result, first_argument_type left,
          second_argument_type right) const
      {
        TA_ASSERT(left.vars() == right.vars(), std::runtime_error,
            "Left and right variable lists must match");
        result = left.vars();
        return result;
      }
    }; // class BinaryOp -- expressions::VariableList


    /// Contraction operation for \c VariableList objects
    template <typename ArrayType1, typename ArrayType2>
    class BinaryOp<
        expressions::VariableList,
        expressions::AnnotatedArray<ArrayType1>,
        expressions::AnnotatedArray<ArrayType2>,
        std::multiplies>
    {
    public:
      typedef const expressions::AnnotatedArray<ArrayType1>& first_argument_type;
      typedef const expressions::AnnotatedArray<ArrayType2>& second_argument_type;
      typedef expressions::VariableList& result_type;
      typedef expressions::VariableList VarList;

      /// Set \c result to the correct variable list for the contraction operation

      /// \param result The result variable list
      /// \param left The variable list for the left-hand argument.
      /// \param right The variable list for the right-hand argument.
      /// \return A reference to the modified \c result
      result_type operator ()(result_type result, first_argument_type left,
          second_argument_type right) const
      {
        typedef expressions::VariableList::const_iterator iterator;
        typedef std::pair<iterator, iterator> it_pair;

        it_pair c0(left.vars().end(), left.vars().end());
        it_pair c1(right.vars().end(), right.vars().end());
        find_common(left.vars().begin(), left.vars().end(), right.vars().begin(), right.vars().end(), c0, c1);

        std::size_t n0 = 2 * left.vars().dim() + 1;
        std::size_t n1 = 2 * right.vars().dim();

        std::map<std::size_t, std::string> v;
        std::pair<std::size_t, std::string> p;
        for(iterator it = left.vars().begin(); it != c0.first; ++it, n0 -= 2) {
          p.first = n0;
          p.second = *it;
          v.insert(p);
        }
        for(iterator it = right.vars().begin(); it != c1.first; ++it, n1 -= 2) {
          p.first = n1;
          p.second = *it;
          v.insert(p);
        }
        n0 -= 2 * (c0.second - c0.first);
        n1 -= 2 * (c1.second - c1.first);
        for(iterator it = c0.second; it != left.vars().end(); ++it, n0 -= 2) {
          p.first = n0;
          p.second = *it;
          v.insert(p);
        }
        for(iterator it = c1.second; it != right.vars().end(); ++it, n1 -= 2) {
          p.first = n1;
          p.second = *it;
          v.insert(p);
        }

        std::vector<std::string> temp;
        for(std::map<std::size_t, std::string>::reverse_iterator it = v.rbegin(); it != v.rend(); ++it)
          temp.push_back(it->second);

        expressions::VariableList(temp.begin(), temp.end()).swap(result);

        return result;
      }

      /// Set \c result to the correct variable list for the contraction operation

      /// \param left The variable list for the left-hand argument.
      /// \param right The variable list for the right-hand argument.
      /// \return The contracted variable list.
      expressions::VariableList operator ()(
          const expressions::VariableList& left, const expressions::VariableList& right) const
      {
        expressions::VariableList result;
        operator()(result, left, right);
        return result;
      }

    private:

      /// Finds the range of common elements for two sets of iterators.

      /// This function finds the first contiguous set of elements equivalent
      /// in two lists. Two pairs of iterators are returned via output parameters
      /// \c common1 and \c common2. These two sets of output iterators point to
      /// the first range of contiguous, equivalent elements in the two lists.
      /// If no common elements far found; then \c common1.first and \c
      /// common1.second both are equal to last1, and likewise for common2.
      /// \tparam InIter1 The input iterator type for the first range of
      /// elements.
      /// \tparam InIter2 The input iterator type for the second range of
      /// elements.
      /// \param[in] first1 An input iterator pointing to the beginning of the
      /// first range of elements to be compared.
      /// \param[in] last1 An input iterator pointing to one past the end of the
      /// first range of elements to be compared.
      /// \param[in] first2 An input iterator pointing to the beginning of the
      /// second range of elements to be compared.
      /// \param[in] last2 An input iterator pointing to one past the end of the
      /// second range of elements to be compared.
      /// \param[out] common1 A pair of iterators where \c common1.first points
      /// to the first common element, and \c common1.second points to one past
      /// the last common element in the first list.
      /// \param[out] common2 A pair of iterators where \c common1.first points
      /// to the first common element, and \c common1.second points to one past
      /// the last common element in the second list.
      template<typename InIter1, typename InIter2>
      static void find_common(InIter1 first1, const InIter1 last1, InIter2 first2, const InIter2 last2,
          std::pair<InIter1, InIter1>& common1, std::pair<InIter2, InIter2>& common2)
      {
        TA_STATIC_ASSERT(detail::is_input_iterator<InIter1>::value);
        TA_STATIC_ASSERT(detail::is_input_iterator<InIter2>::value);
        common1.first = last1;
        common1.second = last1;
        common2.first = last2;
        common2.second = last2;

        // find the first common element in the in the two ranges
        for(; first1 != last1; ++first1) {
          common2.first = std::find(first2, last2, *first1);
          if(common2.first != last2)
            break;
        }
        common1.first = first1;
        first2 = common2.first;

        // find the last common element in the two ranges
        while((first1 != last1) && (first2 != last2)) {
          if(*first1 != *first2)
            break;
          ++first1;
          ++first2;
        }
        common1.second = first1;
        common2.second = first2;
      }

    }; // class BinaryOp -- expressions::VariableList, std::multiplies


    /// Default unary operation for \c VariableList objects

    /// \tparam The operation type to be performed on the data elements.
    template <typename ArrayType, template <typename> class Op>
    class UnaryOp<expressions::VariableList, expressions::AnnotatedArray<ArrayType>, Op>
    {
    public:
      typedef const expressions::AnnotatedArray<ArrayType>& argument_type;
      typedef expressions::VariableList& result_type;

      result_type operator ()(result_type result, argument_type arg) const {
        result = arg.vars();
        return result;
      }
    }; // class UnaryOp -- expressions::VariableList

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_VARIABLE_LIST_MATH_H__INCLUDED
