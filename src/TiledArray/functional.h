#ifndef TILEDARRAY_FUNCTIONAL_H__INCLUDED
#define TILEDARRAY_FUNCTIONAL_H__INCLUDED

#include <boost/iterator/zip_iterator.hpp>
//#include <boost/tuple/tuple.hpp>

namespace TiledArray {

  namespace detail {

    /// Zip operator adapter.

    /// This adapter is used convert a binary operation to a unary operation that
    /// operates on a two element tuple.
    template<typename Left, typename Right, typename Res, typename Op >
    struct ZipOp : public std::unary_function<const boost::tuple<const Left&, const Right&>&, Res>
    {
      typedef Op op_type;

      ZipOp() : op_(op_type()) { }
      ZipOp(op_type op) : op_(op) { }

      Res operator()(const boost::tuple<const Left&, const Left&>& t) const {
        return op_(boost::get<0>(t), boost::get<1>(t));
      }

    private:
      op_type op_;
    }; // struct ZipOp

    template<typename Container, typename Op >
    ZipOp<typename Container::const_iterator, typename Container::const_iterator, typename Op::result_type, Op>
    make_zip_op(const Op& op = Op()) {
      return ZipOp<typename Container::const_iterator, typename Container::const_iterator, typename Op::result_type, Op>(op);
    }

  }  // namespace detail

}  // namespace TiledArray



#endif // TILEDARRAY_FUNCTIONAL_H__INCLUDED
