/*
*  This file is a part of TiledArray.
*  Copyright (C) 2025  Virginia Tech
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
*  Karl Pierce
*  Department of Chemistry, Virginia Tech
*
*  thc_lt_thc_als.h
*  June 5, 2025
*
 */

#ifndef TILEDARRAY_MATH_SOLVERS_CP_THC_LT_THC_ALS__H
#define TILEDARRAY_MATH_SOLVERS_CP_THC_LT_THC_ALS__H

#include <TiledArray/math/solvers/cp/cp.h>
#include <TiledArray/expressions/einsum.h>
#include <TiledArray/math/solvers/cp/cp_reconstruct.h>

namespace TiledArray::math::cp {

/**
* This is a tensor hyper-contraction (THC) optimization class which
* takes a reference order-N tensor that is expressed in the THC format contracted
* With a LT (i.e. a THC X CPD)
* and decomposes it into a set of order-2 tensors all coupled by
* a core matrix. These factors are optimized
* using an alternating least squares algorithm.
*
* @tparam Tile typing for the DistArray tiles
* @tparam Policy policy of the DistArray
**/
template <typename Tile, typename Policy>
class THC_LT_THC_ALS : public CP<Tile, Policy> {
 public:
  using CP<Tile, Policy>::ndim;
  using CP<Tile, Policy>::cp_factors;

  /// Default CP_ALS constructor
  THC_LT_THC_ALS() = default;

  /// CP_ALS constructor function
  /// takes, as a constant reference, the tensor to be decomposed
  /// \param[in] tref A constant reference to the tensor to be decomposed.
  // for now I am going to assume an order-4 THC but later this will be used for
  // arbitrary order.
  THC_LT_THC_ALS(const DistArray<Tile, Policy>& tref1, const DistArray<Tile, Policy>& tref2, const DistArray<Tile, Policy>& tref3)
      : CP<Tile, Policy>(5), ref_orb_a(tref1), ref_orb_b(tref2), ref_core(tref3), world(get_default_world()),
        ref_orb_c(tref1), ref_orb_d(tref2) {

    DistArray<Tile, Policy> pr, pq;
    pr("m,r,rp") = (ref_orb_a("a,m,r") * ref_orb_a("a,m,rp")) * (ref_orb_b("i,m,r") * ref_orb_b("i,m,rp"));
    pq("m,p,q") = ref_core("p,r") * pr("m,r,rp") * ref_core("q,rp");
    this->norm_ref_sq = pq("m,r,rp").dot(pr("m,r,rp")).get();

    //    pr("m,p,r") = ((ref_orb_a("a,m,p") * ref_orb_a("a,m,r")) * (ref_orb_b("i,m,p") * ref_orb_b("i,m,r")));
    //    pq("m,p,q") = pr("m,p,r") * ref_core("r,q");
    //    auto n = pq("m,p,q").dot(pq("m,p,q")).get();
    //    std::cout << n - this->norm_ref_sq << std::endl;
    this->norm_reference = sqrt(this->norm_ref_sq);
    symmetric = true;
  }

  THC_LT_THC_ALS(const DistArray<Tile, Policy>& tref1, const DistArray<Tile, Policy>& tref2, const DistArray<Tile, Policy>& core,
                 const DistArray<Tile, Policy>& tref3, const DistArray<Tile, Policy>& tref4)
      : CP<Tile, Policy>(2 * rank(tref3)), ref_orb_a(tref1), ref_orb_b(tref2), ref_core(core), world(get_default_world()),
        ref_orb_c(tref3), ref_orb_d(tref4) {
    DistArray<Tile, Policy> pr, pq;
    // I need two things that are r_{ai} x r_{ab} which I am calling p x q
    pr("m,p,n,r") = TA::einsum(ref_orb_a("a,m,p"), ref_orb_a("a,n,r"), "m,p,n,r")("m,p,n,r") * TA::einsum(ref_orb_b("i,m,p"), ref_orb_b("i,n,r"), "m,p,n,r")("m,p,n,r");
    pq("m,p,n,r") = TA::einsum(ref_orb_c("a,m,p"), ref_orb_c("a,n,r"), "m,p,n,r")("m,p,n,r") * TA::einsum(ref_orb_d("i,m,p"), ref_orb_d("i,n,r"), "m,p,n,r")("m,p,n,r");
    this->norm_ref_sq = TA::dot(ref_core("p,q"),  pr("m,p,n,r") * (pq("m,q,n,rp") * ref_core("r,rp"))).get();
    this->norm_reference = sqrt(this->norm_ref_sq);
  }

  // This will just assume you've done the correct thing. If you haven't
  // an error will be thrown somewhere else.
  void set_factor_matrices(std::vector<DistArray<Tile, Policy>> & factors){
    cp_factors = factors;
    factors_set = true;
  }

 protected:
  const DistArray<Tile, Policy>& ref_orb_a, ref_orb_b, ref_core, ref_orb_c, ref_orb_d;
  DistArray<Tile, Policy> UnNormalizedLeft, UnNormalizedRight;
  madness::World& world;
  std::vector<typename Tile::value_type> lambda;
  std::vector<DistArray<Tile, Policy>> THC_times_CPD;
  TiledRange1 rank_trange1;
  size_t size_of_core;
  bool factors_set = false;
  bool symmetric = false;

  /// This function constructs the initial CP factor matrices
  /// stores them in CP::cp_factors vector.
  /// In general the initial guess is constructed using quasi-random numbers
  /// generated between [-1, 1]
  /// \param[in] rank rank of the CP approximation
  /// \param[in] rank_trange TiledRange1 of the rank dimension.
  void build_guess(const size_t rank, const TiledRange1 rank_trange) override {
    rank_trange1 = rank_trange;
    if (cp_factors.size() == 0) {
      auto core = this->construct_random_factor(
          world, rank, rank, rank_trange, rank_trange
      );
      core("a,b") = 0.5 * (core("a,b") + core("b,a"));
      cp_factors.emplace_back(this->construct_random_factor(
          world, rank, ref_orb_a.trange().elements_range().extent(0),
          rank_trange, ref_orb_a.trange().data()[0]));
      cp_factors.emplace_back(this->construct_random_factor(
          world, rank, ref_orb_b.trange().elements_range().extent(0),
          rank_trange, ref_orb_b.trange().data()[0]));
      cp_factors.emplace_back(core);
      cp_factors.emplace_back(this->construct_random_factor(
          world, rank, ref_orb_c.trange().elements_range().extent(0),
          rank_trange, ref_orb_c.trange().data()[0]));
      cp_factors.emplace_back(this->construct_random_factor(
          world, rank, ref_orb_d.trange().elements_range().extent(0),
          rank_trange, ref_orb_d.trange().data()[0]));
    } else if(factors_set) {
      // Do nothing and don't throw an error.
//      UnNormalizedLeft = cp_factors[2].clone();
//      UnNormalizedRight = cp_factors[5].clone();
//      this->unNormalized_Factor("r,rp") = UnNormalizedLeft("r,X") * UnNormalizedRight("rp,X");
//      this->normalize_factor(cp_factors[2]);
//      this->normalize_factor(cp_factors[5]);
    }else {
      TA_EXCEPTION("Currently no implementation to increase or change rank");
    }

    // check to see how bad the guess is
    if(false){
      DistArray<Tile, Policy> abr, cdr, tref, cp, diff;
      abr = einsum(ref_orb_a("a,m,r"),  ref_orb_b("b,m,r"), "a,b,m,r");
      cdr = einsum(ref_orb_c("c,m,rp"), ref_orb_d("d,m,rp"), "c,d,m,rp");
      tref("a,b,c,d") = abr("a,b,m,r") * ref_core("r,rp") * cdr("c,d,m,rp");

      abr = einsum(cp_factors[0]("r,a"),  cp_factors[1]("r,b"), "a,b,r");
      cdr = einsum(cp_factors[3]("rp,c"), cp_factors[4]("rp,d"), "c,d,rp");
      cp("a,b,c,d") = abr("a,b,r") * cp_factors[2]("r,rp") * cdr("c,d,rp");

      diff("a,b,c,d") = tref("a,b,c,d") - cp("a,b,c,d");
      std::cout << "Error in initial guess: " << TA::norm2(diff) / TA::norm2(tref) << std::endl;
    }
    return;
  }

  /// This function is specified by the CP solver
  /// optimizes the rank @c rank CP approximation
  /// stored in cp_factors.
  /// \param[in] rank rank of the CP approximation
  /// \param[in] max_iter max number of ALS iterations
  /// \param[in] verbose Should ALS print fit information while running?
  void ALS(size_t rank, size_t max_iter, bool verbose = true) override {
    size_t iter = 0;
    bool converged = false;
    auto nthc = TA::rank(ref_core);
    // initialize partial grammians
    this->partial_grammian.reserve(4);
    {
      auto ptr = this->partial_grammian.data(),
           ptr_facs = cp_factors.data();
      (*(ptr))("P,L") = (*(ptr_facs))("P,n") * (*(ptr_facs))("L,n");
      (*(ptr + 1))("P,L") = (*(ptr_facs + 1))("P,n") * (*(ptr_facs + 1))("L,n");
      (*(ptr + 2))("Q,M") = (*(ptr_facs + 3))("Q,n") * (*(ptr_facs + 3))("M,n");
      (*(ptr + 3))("Q,M") = (*(ptr_facs + 4))("Q,n") * (*(ptr_facs + 4))("M,n");

      DistArray<Tile, Policy> pq;
      pq("m,Q,M") = ref_orb_c("a,m,Q") * cp_factors[3]("M,a");
      THC_times_CPD.emplace_back(pq);
      pq("m,Q,M") *= ref_orb_d("b,m,Q") * cp_factors[4]("M,b");
      pq.truncate();
      THC_times_CPD.emplace_back(pq);

    }

    do {
      // Checking the convergence of the thing

      update_factors_left();
      /*{
        DistArray<Tile, Policy> abcd_old, abcd_new, diff;
        abcd_old("a,b,c,d") = TA::einsum(ref_orb_a("a,m,P"), ref_orb_b("b,m,P"), "a,b,m,P")("a,b,m,P") *
                              ref_core("P,Q") * TA::einsum(ref_orb_c("c,m,Q"), ref_orb_d("d,m,Q"), "c,d,m,Q")("c,d,m,Q");
        abcd_new("a,b,c,d") = TA::einsum(cp_factors[0]("P,a"), UnNormalizedLeft("P,b"), "P,a,b")("P,a,b") *
                              //(UnNormalizedLeft("P,X") * UnNormalizedRight("Q,X")) *
                              cp_factors[2]("P,Q") *
                              TA::einsum(cp_factors[3]("Q,c"), cp_factors[4]("Q,d"), "Q,c,d")("Q,c,d");
        diff("a,b,c,d") = abcd_new("a,b,c,d") - abcd_old("a,b,c,d");
        std::cout << "Norm old: " << TA::norm2(abcd_old) << std::endl;
        std::cout << "Norm new: " << TA::norm2(abcd_new) << std::endl;
        std::cout << "Error: " << TA::norm2(diff) / TA::norm2(abcd_old) << std::endl;
      }*/
      // Preserve symmetry in the structure
      {
        cp_factors[3] = cp_factors[0].clone();
        THC_times_CPD[1]("m,Q,M") = ref_orb_c("a,m,Q") * cp_factors[3]("M,a");
        this->partial_grammian[2]("r,rp") = cp_factors[3]("r,c") * cp_factors[3]("rp,c");

        cp_factors[4] = cp_factors[1].clone();
        THC_times_CPD[1]("m,Q,M") *= ref_orb_d("b,m,Q") * cp_factors[4]("M,b");
        this->partial_grammian[3]("r,rp") = cp_factors[4]("r,c") * cp_factors[4]("rp,c");
      }
      // Update the core tensor and don't rescale to normalized
      update_core();
      /*{
        DistArray<Tile, Policy> abcd_old, abcd_new, diff;
        abcd_old("a,b,c,d") = TA::einsum(ref_orb_a("a,m,P"), ref_orb_b("b,m,P"), "a,b,m,P")("a,b,m,P") *
                              ref_core("P,Q") * TA::einsum(ref_orb_c("c,m,Q"), ref_orb_d("d,m,Q"), "c,d,m,Q")("c,d,m,Q");
        abcd_new("a,b,c,d") = TA::einsum(cp_factors[0]("P,a"), cp_factors[1]("P,b"), "P,a,b")("P,a,b") *
                              //(UnNormalizedLeft("P,X") * UnNormalizedRight("Q,X")) *
                              cp_factors[2]("P,Q") *
                              TA::einsum(cp_factors[3]("Q,c"), cp_factors[4]("Q,d"), "Q,c,d")("Q,c,d");
        diff("a,b,c,d") = abcd_new("a,b,c,d") - abcd_old("a,b,c,d");
        std::cout << "Norm old: " << TA::norm2(abcd_old) << std::endl;
        std::cout << "Norm new: " << TA::norm2(abcd_new) << std::endl;
        std::cout << "Error: " << TA::norm2(diff) / TA::norm2(abcd_old) << std::endl;
      }*/
      //update_factors_right();
      /*{
        DistArray<Tile, Policy> abcd_old, abcd_new, diff;
        abcd_old("a,b,c,d") = TA::einsum(ref_orb_a("a,m,P"), ref_orb_b("b,m,P"), "a,b,m,P")("a,b,m,P") *
                              ref_core("P,Q") * TA::einsum(ref_orb_c("c,m,Q"), ref_orb_d("d,m,Q"), "c,d,m,Q")("c,d,m,Q");
        abcd_new("a,b,c,d") = TA::einsum(cp_factors[0]("P,a"), cp_factors[1]("P,b"), "P,a,b")("P,a,b") *
                              (UnNormalizedLeft("P,X") * UnNormalizedRight("Q,X")) * TA::einsum(cp_factors[3]("Q,c"), cp_factors[4]("Q,d"), "Q,c,d")("Q,c,d");
        diff("a,b,c,d") = abcd_new("a,b,c,d") - abcd_old("a,b,c,d");
        std::cout << "Norm old: " << TA::norm2(abcd_old) << std::endl;
        std::cout << "Norm new: " << TA::norm2(abcd_new) << std::endl;
        std::cout << "Error: " << TA::norm2(diff) / TA::norm2(abcd_old) << std::endl;
      }*/

      converged = this->check_thc_fit(verbose);

      ++iter;
    } while (iter < max_iter && !converged);
    this->unNormalized_Factor = cp_factors[4];
  }

  // These assume the center is a sqrt of the core tensor.
  /*void update_factors_left() {
    DistArray<Tile, Policy> W, MTtKRP, env, core_gram;

    // compute the right hand side of the problem
    // Start forming the grammian from the right hand side.
    core_gram("P,L") = cp_factors[2]("P,X") * UnNormalizedRight("L,X");
    W("P,Q") = core_gram("P,L") * TA::einsum(this->partial_grammian[2]("L,M"),
                                             this->partial_grammian[3]("L,M"), "L,M")("L,M") *
               core_gram("Q,M");

    // Effectively this is the effective environment from the target tensor times the right hand side problem.
    env("m,P,L") = ref_core("P,Q") * (THC_times_CPD[1]("m,Q,M") * core_gram("L,M"));
    // Set up the order 3 problem by hadamard contracting the B factor matrix.
    MTtKRP("L,a") = TA::einsum(TA::einsum(ref_orb_b("b,m,P"), cp_factors[1]("L,b"), "m,P,L")("m,P,L"), env("m,P,L"), "m,P,L")("m,P,L") * ref_orb_a("a,m,P");
    // gather the contributions to the grammian from B.

    // Invert the grammian and contract with MTtKRP.
    this->cholesky_inverse(MTtKRP, TA::einsum(this->partial_grammian[1]("r,rp"), W("r,rp"), "r,rp"));
    world.gop.fence();  // N.B. seems to deadlock without this

    this->normalize_factor(MTtKRP);
    cp_factors[0] = MTtKRP;
    this->partial_grammian[0]("r,rp") = MTtKRP("r,n") * MTtKRP("rp,n");
    this->partial_grammian[0].truncate();
    THC_times_CPD[0]("m,P,Q") = ref_orb_a("a,m,P") * MTtKRP("Q,a");

    // Take the environment "tmp" and contract now with A.
    MTtKRP("L,b") = TA::einsum(THC_times_CPD[0]("m,P,L"), env("m,P,L"), "m,P,L")("m,P,L") * ref_orb_b("b,m,P");

    // Form the contributions to the grammian from A
    this->cholesky_inverse(MTtKRP, TA::einsum(this->partial_grammian[0]("r,rp"), W("r,rp"), "r,rp"));
    world.gop.fence();  // N.B. seems to deadlock without this

    this->normalize_factor(MTtKRP);
    cp_factors[1] = MTtKRP;
    this->partial_grammian[1]("r,rp") = MTtKRP("r,n") * MTtKRP("rp,n");
    this->partial_grammian[1].truncate();
    THC_times_CPD[0]("m,P,Q") *= ref_orb_b("b,m,P") * MTtKRP("Q,b");

    DistArray<Tile, Policy> abr, Wr, R;

    MTtKRP("P,X") = THC_times_CPD[0]("m,L,P") * ref_core("L,M") * THC_times_CPD[1]("m,M,Q") * UnNormalizedRight("Q,X");
    W = TA::einsum(this->partial_grammian[0]("r,rp"), this->partial_grammian[1]("r,rp"), "r,rp");//abr("r,a,b") * abr("rp,a,b");
    Wr("X,Y") = UnNormalizedRight("r,X") * TA::einsum(this->partial_grammian[2]("r,rp"),
                                                      this->partial_grammian[3]("r,rp"), "r,rp")("r,rp") *
                UnNormalizedRight("rp,Y");

    W = lu_inv(W);
    Wr = lu_inv(Wr);
    MTtKRP("Q,Y") = W("Q,P") * MTtKRP("P,X") * Wr("Y,X");
    MTtKRP.truncate();
    UnNormalizedLeft = MTtKRP.clone();
    this->normalize_factor(MTtKRP);
    cp_factors[2] = MTtKRP;
  }

  void update_factors_right(){
    DistArray<Tile, Policy> W, MTtKRP, env, core_gram;

    // compute the right hand side of the problem
    // Start forming the grammian from the right hand side.
    core_gram("P,L") = cp_factors[5]("P,X") * UnNormalizedLeft("L,X");
    core_gram.truncate();
    W("P,Q") = core_gram("P,L") * TA::einsum(this->partial_grammian[0]("L,M"),
                                             this->partial_grammian[1]("L,M"), "L,M")("L,M") *
               core_gram("Q,M");

    // Effectively this is the effective environment from the target tensor times the right hand side problem.
    env("m,P,L") = ref_core("Q,P") * (THC_times_CPD[0]("m,Q,M") * core_gram("L,M"));
    // Set up the order 3 problem by hadamard contracting the B factor matrix.
    MTtKRP("L,a") = TA::einsum(TA::einsum(ref_orb_d("b,m,P"), cp_factors[4]("L,b"), "m,P,L")("m,P,L"), env("m,P,L"), "m,P,L")("m,P,L") * ref_orb_c("a,m,P");
    // gather the contributions to the grammian from B.

    // Invert the grammian and contract with MTtKRP.
    this->cholesky_inverse(MTtKRP, TA::einsum(this->partial_grammian[3]("r,rp"), W("r,rp"), "r,rp"));
    world.gop.fence();  // N.B. seems to deadlock without this

    this->normalize_factor(MTtKRP);
    cp_factors[3] = MTtKRP;
    this->partial_grammian[2]("r,rp") = MTtKRP("r,n") * MTtKRP("rp,n");
    THC_times_CPD[1]("m,P,Q") = ref_orb_c("a,m,P") * MTtKRP("Q,a");

    // Take the environment "tmp" and contract now with A.
    MTtKRP("L,b") = TA::einsum(THC_times_CPD[1]("m,P,L"), env("m,P,L"), "m,P,L")("m,P,L") * ref_orb_d("b,m,P");

    // Form the contributions to the grammian from A
    this->cholesky_inverse(MTtKRP, TA::einsum(this->partial_grammian[2]("r,rp"), W("r,rp"), "r,rp"));
    world.gop.fence();  // N.B. seems to deadlock without this

    this->normalize_factor(MTtKRP);
    cp_factors[4] = MTtKRP;
    this->partial_grammian[3]("r,rp") = MTtKRP("r,n") * MTtKRP("rp,n");
    THC_times_CPD[1]("m,P,Q") *= ref_orb_d("b,m,P") * MTtKRP("Q,b");

    DistArray<Tile, Policy> Wr;
    MTtKRP("Q,X") = THC_times_CPD[0]("m,L,P") * ref_core("L,M") * THC_times_CPD[1]("m,M,Q") * UnNormalizedLeft("P,X");
    W = TA::einsum(this->partial_grammian[1]("r,rp"), this->partial_grammian[2]("r,rp"), "r,rp");
    Wr("X,Y") = UnNormalizedLeft("r,X") * TA::einsum(this->partial_grammian[0]("r,rp"),
                                                      this->partial_grammian[1]("r,rp"), "r,rp")("r,rp") *
                UnNormalizedLeft("rp,Y");
    Wr.truncate();

    W = lu_inv(W);
    Wr = lu_inv(Wr);
    MTtKRP("Q,Y") = W("Q,P") * MTtKRP("P,X") * Wr("Y,X");
    MTtKRP.truncate();
    UnNormalizedRight = MTtKRP.clone();
    this->normalize_factor(MTtKRP);
    cp_factors[5] = MTtKRP;
  }*/

  //
  void update_factors_left(){
    DistArray<Tile, Policy> env, b_mON, MttKRP, W, W_env, pq;

    // solve for A
    env("m,M,P") = ref_core("M,N") * THC_times_CPD[1]("m,N,Q") * cp_factors[2]("P,Q");
    b_mON("m,M,P") = ref_orb_b("b,m,M") * cp_factors[1]("P,b");
    env.truncate();
    b_mON.truncate();

    MttKRP("P,a") = TA::einsum(env("m,M,P"), b_mON("m,M,P"), "m,M,P")("m,M,P") * ref_orb_a("a,m,M");

    DistArray<Tile, Policy> temp;
    temp = TA::einsum(this->partial_grammian[2]("r,rp"),
                      this->partial_grammian[3]("r,rp"), "r,rp");
    W_env("L,rp") = (cp_factors[2]("L,r") * temp("r,rp"));
    W_env("L,M") = W_env("L,rp") * cp_factors[2]("rp,M");
    W("L,M") = this->partial_grammian[1]("L,M") * W_env("L,M");

    MttKRP = math::linalg::lu_solve(W, MttKRP);
    //this->cholesky_inverse(MttKRP, W);

    world.gop.fence();  // N.B. seems to deadlock without this

    this->normalize_factor(MttKRP);
    MttKRP.truncate();
    cp_factors[0] = MttKRP;
    this->partial_grammian[0]("r,rp") = MttKRP("r,n") * MttKRP("rp,n");
    pq("m,M,P") = ref_orb_a("a,m,M") * MttKRP("P,a");

    // Solve for B
    MttKRP("P,b") = TA::einsum(env("m,M,P"), pq("m,M,P"), "m,M,P")("m,M,P") * ref_orb_b("b,m,M");
    W("L,M") = this->partial_grammian[0]("L,M") * W_env("L,M");

    //this->cholesky_inverse(MttKRP, W);
    MttKRP = math::linalg::lu_solve(W, MttKRP);
    world.gop.fence();  // N.B. seems to deadlock without this

    UnNormalizedLeft = MttKRP.clone();
    this->normalize_factor(MttKRP);
    MttKRP.truncate();
    cp_factors[1] = MttKRP;
    this->partial_grammian[1]("r,rp") = MttKRP("r,n") * MttKRP("rp,n");
    THC_times_CPD[0]("m,M,P") = pq("m,M,P") * (ref_orb_b("b,m,M") * MttKRP("P,b"));
  }

  void update_core(){
    DistArray<Tile, Policy> a_mMP, b_mMP, L, R, MttKRP;
    MttKRP("P,Q") = THC_times_CPD[0]("m,M,P") * (ref_core("M,N") * THC_times_CPD[1]("m,N,Q"));
    this->MTtKRP = MttKRP;

    L = math::linalg::lu_inv(TA::einsum(this->partial_grammian[0]("P,Q"), this->partial_grammian[1]("P,Q"),"P,Q"));
    R = math::linalg::lu_inv(TA::einsum(this->partial_grammian[2]("P,Q"), this->partial_grammian[3]("P,Q"),"P,Q"));

    cp_factors[2]("P,Q") = L("P,L") * MttKRP("L,M") * R("Q,M");
    cp_factors[2].truncate();
  }

  void update_factors_right(){
    DistArray<Tile, Policy> env, b_mON, MttKRP, W, W_env, pq;

    // solve for A
    env("m,M,P") = ref_core("M,N") * THC_times_CPD[0]("m,N,Q") * cp_factors[2]("P,Q");
    b_mON("m,M,P") = ref_orb_d("b,m,M") * cp_factors[4]("P,b");
    MttKRP("P,a") = TA::einsum(env("m,M,P"), b_mON("m,M,P"), "m,M,P")("m,M,P") * ref_orb_c("a,m,M");

    W_env("L,M") = cp_factors[2]("L,r") *
                   TA::einsum(this->partial_grammian[0]("r,rp"),
                              this->partial_grammian[1]("r,rp"), "r,rp")("r,rp") *
                   cp_factors[2]("M,rp");
    W("L,M") = this->partial_grammian[3]("L,M") * W_env("L,M");

    this->cholesky_inverse(MttKRP, W);
    world.gop.fence();  // N.B. seems to deadlock without this

    this->normalize_factor(MttKRP);
    cp_factors[3] = MttKRP;
    this->partial_grammian[2]("r,rp") = MttKRP("r,n") * MttKRP("rp,n");
    pq("m,M,P") = ref_orb_c("a,m,M") * MttKRP("P,a");

    // Solve for B
    MttKRP("P,b") = TA::einsum(env("m,M,P"), pq("m,M,P"), "m,M,P")("m,M,P") * ref_orb_d("b,m,M");
    W("L,M") = this->partial_grammian[2]("L,M") * W_env("L,M");

    this->cholesky_inverse(MttKRP, W);
    world.gop.fence();  // N.B. seems to deadlock without this

    this->normalize_factor(MttKRP);
    cp_factors[4] = MttKRP;
    this->partial_grammian[3]("r,rp") = MttKRP("r,n") * MttKRP("rp,n");
    THC_times_CPD[1]("m,M,P") = pq("m,M,P") * (ref_orb_d("b,m,M") * MttKRP("P,b"));
  }
};

}  // namespace TiledArray::math::cp

#endif  // TILEDARRAY_MATH_SOLVERS_CP_THC_LT_THC_ALS__H
