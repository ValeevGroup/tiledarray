#ifndef TILEDARRAY_MATH_H__INCLUDED
#define TILEDARRAY_MATH_H__INCLUDED

#include <world/typestuff.h>
#include <functional>

namespace TiledArray {
  namespace math {

    // These tags represent the iteration ordering of the operation.
    typedef enum {
      linear_tag,
      contract_tag,
      permute_tag
    } OpTag;

    template <typename T>
    struct permute { };

    template <template <typename> class Op>
    struct OpOrder {
      static const OpTag tag = linear_tag;
    };

    template <>
    struct OpOrder<std::multiplies> {
      static const OpTag tag = contract_tag;
    };

    template <>
    struct OpOrder<permute> {
      static const OpTag tag = permute_tag;
    };

    // These classes are place holders. You must add specializations to do anything
    // useful with them.

    template <typename, typename, typename, template <typename> class, OpTag>
    class BinaryOpImpl;

    template <typename, typename, template <typename> class, OpTag>
    class UnaryOpImpl;

    template <typename Res, typename LeftArg, typename RightArg, template <typename> class Op>
    class BinaryOp : public BinaryOpImpl<Res, LeftArg, RightArg, Op, OpOrder<Op>::tag > { };

    template <typename Res, typename Arg, template <typename> class Op>
    class UnaryOp : public UnaryOpImpl<Res, Arg, Op, OpOrder<Op>::tag > { };

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_H__INCLUDED
