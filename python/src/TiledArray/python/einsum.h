/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
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

#ifndef TA_PYTHON_EINSUM_H
#define TA_PYTHON_EINSUM_H

#include <tiledarray.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/algorithm/string/join.hpp>

#include <tuple>
#include <vector>

#include "expression.h"

namespace TiledArray {
namespace python {
namespace einsum {

  using index_tuple = std::vector<char>;

  struct Expression {
    std::vector<index_tuple> terms;
    index_tuple result;
  };

  namespace qi = boost::spirit::qi;
  namespace ascii = boost::spirit::ascii;

  template <typename Iterator>
  struct grammar : qi::grammar<Iterator, Expression(), ascii::space_type> {
    grammar()
      : grammar::base_type(expression)
    {
      indices %= +(ascii::alpha);
      terms %= (indices % ',');
      expression %= terms >> qi::lit("->") >> indices;
    }

    qi::rule<Iterator, Expression(), ascii::space_type> expression;
    qi::rule<Iterator, index_tuple(), ascii::space_type> indices, lhs;
    qi::rule<Iterator, std::vector<index_tuple>(), ascii::space_type> terms;

  };

  inline Expression parse(std::string expr) {
    typedef std::string::const_iterator Iterator;
    Iterator iter = expr.begin();
    Iterator end = expr.end();
    grammar<Iterator> g;
    boost::spirit::ascii::space_type space;
    Expression expression;
    bool status = phrase_parse(iter, end, g, space, expression);
    if (!status || iter != end) {
      throw std::domain_error("einsum: invalid expression: " + expr);
    }
    return expression;
  }

}
}
}


BOOST_FUSION_ADAPT_STRUCT(
  TiledArray::python::einsum::Expression,
  (decltype(TiledArray::python::einsum::Expression::terms), terms),
  (decltype(TiledArray::python::einsum::Expression::result), result)
);


namespace TiledArray {
namespace python {
namespace einsum {

  template<class A>
  inline auto evaluate(std::tuple<A*,index_tuple> term) {
    std::vector<std::string> v;
    for (char c : std::get<1>(term)) { v.push_back(std::string(1,c)); }
    auto index = boost::algorithm::join(v, ",");
    return (*std::get<0>(term))(index);
  }

  template<class Array, size_t ... Idx>
  auto evaluate(
    std::tuple< Array*, index_tuple > result,
    std::vector< std::tuple< const Array*, index_tuple> > terms,
    std::integer_sequence<size_t,Idx...>)
  {
    constexpr size_t N = sizeof...(Idx);
    if (N == terms.size()) {
      return (evaluate(result) = (evaluate(terms.at(Idx)) * ...));
    }
    if constexpr (N < TA_PYTHON_MAX_EXPRESSION) {
        return evaluate(result, terms, std::integer_sequence<size_t,Idx...,N>{});
    }
    throw std::domain_error(
      "Expression exceeds TA_PYTHON_MAX_EXPRESSION=" + std::to_string(TA_PYTHON_MAX_EXPRESSION)
    );
  }

  template<class Array>
  void evaluate(
    std::tuple< Array*, index_tuple > result,
    std::vector< std::tuple< const Array*, index_tuple> > terms)
  {
    evaluate<Array>(result, terms, std::integer_sequence<size_t,0>{});
  }

  template<class Array>
  void evaluate(std::string expr, std::vector<Array*> args) {

    auto expression = parse(expr);

    size_t n = args.size();

    if (1 + expression.terms.size() != n) {
      throw std::domain_error("einsum: number of args does not match expression");
    }

    std::vector< std::tuple<const Array*, index_tuple> > terms;
    for (size_t i = 0; i < n-1; ++i) {
      terms.emplace_back(args[i], expression.terms.at(i));
    }

    std::tuple<Array*, index_tuple> result = { args[n-1], expression.result };

    evaluate<Array>(result, terms);

  }

  template<class Array>
  void einsum(std::string expr, Array* a0, py::args args) {
    std::vector<Array*> argv{a0};
    for (auto o : args) {
      auto *ptr = py::cast<Array*>(o);
      if (!ptr) {
        throw std::runtime_error("einsum: tensor arguments must of same type");
      }
      argv.push_back(ptr);
    }
    evaluate<Array>(expr, argv);
  }

  void __init__(py::module m) {
    m.def("einsum", &einsum::einsum< TArray<double> >);
    m.def("einsum", &einsum::einsum< TSpArray<double> >);
  }

}
}
}

#endif // TA_PYTHON_EINSUM_H
