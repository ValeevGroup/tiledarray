#ifndef TILEDARRAY_FUNCTIONAL_H__INCLUDED
#define TILEDARRAY_FUNCTIONAL_H__INCLUDED

#include <boost/functional.hpp>

namespace TiledArray {
  namespace detail {

    template <class Op>
    class Binder1st : public boost::binder1st<Op> {
    public:
      typedef typename boost::binder1st<Op>::argument_type argument_type;
      typedef typename boost::binder1st<Op>::result_type result_type;
    public:
      Binder1st() : boost::binder1st<Op>(Op(), typename boost::binary_traits<Op>::first_argument_type()) { }
      Binder1st(typename boost::binary_traits<Op>::param_type x,
                typename boost::call_traits<typename boost::binary_traits<Op>::first_argument_type>::param_type y)
          :
            boost::binder1st<Op>(x,y)
      {}

      using boost::binder1st<Op>::operator();

      template <typename Archive>
      void serialize(Archive& ar) const {
        ar & boost::binder1st<Op>::op & boost::binder1st<Op>::value;
      }
    }; // class Binder1st

    template <class Op>
    class Binder2nd : public boost::binder2nd<Op> {
    public:
      using boost::binder1st<Op>::argument_type;
      using boost::binder1st<Op>::result_type;
    public:
      Binder2nd() : boost::binder2nd<Op>(Op(), typename boost::binary_traits<Op>::second_argument_type()) { }
      Binder2nd(typename boost::binary_traits<Op>::param_type x,
                typename boost::call_traits<typename boost::binary_traits<Op>::second_argument_type>::param_type y) :
          boost::binder2nd<Op>(x,y)
      { }

      using boost::binder1st<Op>::operator();

      template <typename Archive>
      void serialize(Archive& ar) const {
        ar & boost::binder1st<Op>::op & boost::binder1st<Op>::value;
      }
    }; // class Binder2nd

    template <typename Op, typename T>
    inline Binder1st<Op> bind1st(const Op& op, const T& t) {
      return Binder1st<Op>(op, t);
    }

    template <typename Op, typename T>
    inline Binder2nd<Op> bind2nd(const Op& op, const T& t) {
      return Binder2nd<Op>(op, t);
    }

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_FUNCTIONAL_H__INCLUDED
