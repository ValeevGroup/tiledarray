#ifndef TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED

#include <TiledArray/summa.h>
#include <TiledArray/vspgemm.h>

namespace TiledArray {
  namespace expressions {

    template <typename LExp, typename RExp>
    typename detail::ContractionExp<LExp, RExp>::type
    make_contraction_tensor(const LExp& left, const RExp& right) {
      // Define the base impl type
      typedef detail::TensorExpressionImpl<typename detail::ContractionResult<LExp, RExp>::type> impl_type;

      // Create the implementation pointer
      impl_type* pimpl = NULL;
      if(left.is_dense() && right.is_dense())
        pimpl = new Summa<LExp, RExp>(left, right);
      else
        pimpl = new VSpGemm<LExp, RExp>(left, right);

      return typename detail::ContractionExp<LExp, RExp>::type(
          std::shared_ptr<impl_type>(pimpl,
              madness::make_deferred_deleter<impl_type>(left.get_world())));
    }

  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_TENSOR_H__INCLUDED
