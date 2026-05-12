/// Locks plain-tensor zero-overhead from the arena plan storage field.

#include "TiledArray/tensor.h"
#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tile_op/contract_reduce.h"
#include "TiledArray/util/function.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <cstddef>
#include <type_traits>
#include <variant>

namespace TA = TiledArray;

namespace {

using PlainResult = TA::Tensor<double>;
using PlainLeft = TA::Tensor<double>;
using PlainRight = TA::Tensor<double>;
using PlainScalar = double;

using PlainContractReduceBase =
    TA::detail::ContractReduceBase<PlainResult, PlainLeft, PlainRight,
                                   PlainScalar>;

using PlainArenaPlanStorage =
    TA::detail::arena_plan_storage_t<PlainResult, PlainLeft, PlainRight>;

using PlainElemMulAddOp =
    TA::function_ref<void(double&, const double&, const double&)>;

/// Shadows the public field order of ContractReduceBase::Impl on master.
struct ImplLayoutMaster {
  TA::math::GemmHelper gemm_helper_;
  PlainScalar alpha_;
  TA::BipartitePermutation perm_;
  PlainElemMulAddOp elem_muladd_op_;
};

/// Same as ImplLayoutMaster + trailing TA_NO_UNIQUE_ADDRESS arena_plan_.
struct ImplLayoutAllocator {
  TA::math::GemmHelper gemm_helper_;
  PlainScalar alpha_;
  TA::BipartitePermutation perm_;
  PlainElemMulAddOp elem_muladd_op_;
  TA_NO_UNIQUE_ADDRESS PlainArenaPlanStorage arena_plan_;
};

/// Sizes captured against master; re-baseline if the toolchain changes.
constexpr std::size_t kTensorDoubleSizeMaster = 328;
constexpr std::size_t kContractReduceBaseSizeMaster = 16;
constexpr std::size_t kImplLayoutSizeMaster = 248;

static_assert(std::is_same_v<PlainArenaPlanStorage, std::monostate>,
              "plain-tensor arena_plan_storage_t must be std::monostate");

static_assert(sizeof(ImplLayoutAllocator) == sizeof(ImplLayoutMaster),
              "TA_NO_UNIQUE_ADDRESS failed to fold arena_plan_ into padding");

}

BOOST_AUTO_TEST_SUITE(arena_sizeof_invariant_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(tensor_double_sizeof_matches_master) {
  BOOST_CHECK_EQUAL(sizeof(TA::Tensor<double>), kTensorDoubleSizeMaster);
}

BOOST_AUTO_TEST_CASE(contract_reduce_base_sizeof_matches_master) {
  BOOST_CHECK_EQUAL(sizeof(PlainContractReduceBase),
                    kContractReduceBaseSizeMaster);
}

BOOST_AUTO_TEST_CASE(impl_layout_no_unique_address_invariant) {
  BOOST_CHECK_EQUAL(sizeof(ImplLayoutMaster), kImplLayoutSizeMaster);
  BOOST_CHECK_EQUAL(sizeof(ImplLayoutAllocator), kImplLayoutSizeMaster);
  BOOST_CHECK_EQUAL(sizeof(ImplLayoutAllocator), sizeof(ImplLayoutMaster));
}

BOOST_AUTO_TEST_CASE(plain_arena_plan_storage_is_monostate) {
  BOOST_CHECK((std::is_same_v<PlainArenaPlanStorage, std::monostate>));
  BOOST_CHECK_EQUAL(sizeof(PlainArenaPlanStorage), sizeof(std::monostate));
}

BOOST_AUTO_TEST_SUITE_END()
