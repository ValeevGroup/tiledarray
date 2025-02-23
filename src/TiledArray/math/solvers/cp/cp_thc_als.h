/*
*  This file is a part of TiledArray.
*  Copyright (C) 2023  Virginia Tech
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
*  cp.h
*  April 17, 2022
*
*/

#ifndef TILEDARRAY_MATH_SOLVERS_CP_CP_THC_ALS__H
#define TILEDARRAY_MATH_SOLVERS_CP_CP_THC_ALS__H

#include <TiledArray/math/solvers/cp/cp.h>
#include <TiledArray/expressions/einsum.h>
#include <TiledArray/math/solvers/cp/cp_reconstruct.h>

namespace TiledArray::math::cp {

/**
* This is a canonical polyadic (CP) optimization class which
* takes a reference order-N tensor that is expressed in the THC format
* and decomposes it into a set of order-2 tensors all coupled by
* a hyperdimension called the rank. These factors are optimized
* using an alternating least squares algorithm.
*
* @tparam Tile typing for the DistArray tiles
* @tparam Policy policy of the DistArray
**/
template <typename Tile, typename Policy>
class CP_THC_ALS : public CP<Tile, Policy> {
public:
 using CP<Tile, Policy>::ndim;
 using CP<Tile, Policy>::cp_factors;

 /// Default CP_ALS constructor
 CP_THC_ALS() = default;

 /// CP_ALS constructor function
 /// takes, as a constant reference, the tensor to be decomposed
 /// \param[in] tref A constant reference to the tensor to be decomposed.
 // for now I am going to assume an order-4 THC but later this will be used for
 // arbitrary order.
 CP_THC_ALS(const DistArray<Tile, Policy>& tref1, const DistArray<Tile, Policy>& tref2, const DistArray<Tile, Policy>& tref3)
     : CP<Tile, Policy>(2 * rank(tref3)), ref_orb_a(tref1), ref_orb_b(tref2), ref_core(tref3), world(tref1.world()),
        ref_orb_c(tref1), ref_orb_d(tref2) {

   DistArray<Tile, Policy> pr, pq;
   pr("r,rp") = (ref_orb_a("a,r") * ref_orb_a("a,rp")) * (ref_orb_b("i,r") * ref_orb_b("i,rp"));
   pq("p,q") = ref_core("p,r") * pr("r,rp") * ref_core("q,rp");
   this->norm_ref_sq = pq("r,rp").dot(pr("r,rp")).get();
   this->norm_reference = sqrt(this->norm_ref_sq);
   symmetric = true;
 }

 CP_THC_ALS(const DistArray<Tile, Policy>& tref1, const DistArray<Tile, Policy>& tref2, const DistArray<Tile, Policy>& core,
            const DistArray<Tile, Policy>& tref3, const DistArray<Tile, Policy>& tref4)
     : CP<Tile, Policy>(2 * rank(tref3)), ref_orb_a(tref1), ref_orb_b(tref2), ref_core(core), world(tref1.world()),
       ref_orb_c(tref3), ref_orb_d(tref4) {
   DistArray<Tile, Policy> pr, pq;
   // I need two things that are r_{ai} x r_{ab} which I am calling p x q
   pr("p,r") = ((ref_orb_a("a,p") * ref_orb_a("a,r")) * (ref_orb_b("i,p") * ref_orb_b("i,r")));
   pq("p,q") = pr("p,r") * ref_core("r,q");
   pr("p,q") = ref_core("p,r") * ((ref_orb_c("b,r") * ref_orb_c("b,q")) * (ref_orb_d("j,r") * ref_orb_d("j,q")));
   this->norm_ref_sq = pq("p,q").dot(pr("p,q")).get();
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
     cp_factors.emplace_back(this->construct_random_factor(
         world, rank, ref_orb_a.trange().elements_range().extent(0),
         rank_trange, ref_orb_a.trange().data()[0]));
     cp_factors.emplace_back(this->construct_random_factor(
         world, rank, ref_orb_b.trange().elements_range().extent(0),
         rank_trange, ref_orb_b.trange().data()[0]));
     cp_factors.emplace_back(this->construct_random_factor(
         world, rank, ref_orb_c.trange().elements_range().extent(0),
         rank_trange, ref_orb_c.trange().data()[0]));
     cp_factors.emplace_back(this->construct_random_factor(
         world, rank, ref_orb_d.trange().elements_range().extent(0),
         rank_trange, ref_orb_d.trange().data()[0]));
   } else if(factors_set) {
     // Do nothing and don't throw an error.
   }else {
     TA_EXCEPTION("Currently no implementation to increase or change rank");
   }

   return;
 }

 /// This function is specified by the CP solver
 /// optimizes the rank @c rank CP approximation
 /// stored in cp_factors.
 /// \param[in] rank rank of the CP approximation
 /// \param[in] max_iter max number of ALS iterations
 /// \param[in] verbose Should ALS print fit information while running?
 void ALS(size_t rank, size_t max_iter, bool verbose = false) override {
   size_t iter = 0;
   bool converged = false;
   auto nthc = TA::rank(ref_core);
   // initialize partial grammians
   {
     auto ptr = this->partial_grammian.begin();
     for (auto& i : cp_factors) {
       (*ptr)("r,rp") = i("r,n") * i("rp, n");
       ++ptr;
     }
     DistArray<Tile, Policy> pq;
     pq("p,q") = ref_orb_c("a,p") * cp_factors[2]("q,a");
     THC_times_CPD.emplace_back(pq);
     pq("p,q") *= ref_orb_d("b,p") * cp_factors[3]("q,b");
     pq.truncate();
     THC_times_CPD.emplace_back(pq);

   }
//   auto factor_begin = cp_factors.data(),
//        gram_begin = this->partial_grammian.data();
//   DistArray<Tile, Policy> abr, tref;
//   abr = einsum(ref_orb_a("a,r"),  ref_orb_b("b,r"), "a,b,r");
//   tref("a,b,c,d") = abr("a,b,r") * ref_core("r,rp") * abr("c,d,rp");
//
//   std::cout << "Norm2 : " << norm2(tref) << std::endl;
//   std::cout << "set: " << this->norm_reference << std::endl;
   do {
     update_factors_left();
     update_factors_right();
     converged = this->check_fit(verbose);
     //     for (auto i = 0; i < nthc; ++i) {
     //       update_factor(i, rank);
     //     }
     ++iter;
   } while (iter < max_iter && !converged);
 }

 void update_factor(size_t mode, size_t rank){

   size_t pos = 2 * mode, pos_plus_one = pos + 1;
   // going through the core 0 is associated with factors 0 and 1
   // core 1 associated with factors 2 and 3 ...
   // First we need to take other side of the problem and contract it with the core
   // if core is greater than 2 I need to contract all the centers but
   // here there's only one so just do one contract
   size_t other_side = (mode + 1) % 2;
   DistArray<Tile, Policy> env, pq, W_env, W;
   {
     DistArray<Tile, Policy> An;
     env("p,q") = ref_core("p,r") * THC_times_CPD[other_side]("r,q");

     pq("p,q") = ref_orb_b("b,p") * cp_factors[pos_plus_one]("q,b");
     An("q,a") = (pq("p,q") * env("p,q")) *  ref_orb_a("a,p");

     // TODO check to see if the Cholesky will fail. If it does
     // use SVD
     DistArray<Tile, Policy> W;
     other_side *= 2;
     W_env("p,q") = this->partial_grammian[other_side]("p,q") *
                    this->partial_grammian[other_side + 1]("p,q");
     W("p,q") = this->partial_grammian[pos_plus_one]("p,q") * W_env("p,q");

     this->cholesky_inverse(An, W);
     world.gop.fence();  // N.B. seems to deadlock without this

     this->normalize_factor(An);
     cp_factors[pos] = An;
     auto& gram = this->partial_grammian[pos];
     gram("r,rp") = An("r,n") * An("rp,n");
     pq("p,q") = ref_orb_a("a,p") * cp_factors[pos]("q,a");
     THC_times_CPD[mode] = pq;
   }

   // Finished with the first factor in THC.
   // Starting the second factor
   {
     DistArray<Tile, Policy> Bn;
     Bn = DistArray<Tile, Policy>();
     Bn("q,b") = einsum(pq("p,q"), env("p,q"), "p,q")("p,q") * ref_orb_b("b,p");

     this->MTtKRP = Bn;

     // TODO check to see if the Cholesky will fail. If it does
     // use SVD
     W("p,q") = this->partial_grammian[pos]("p,q") * W_env("p,q");
     this->cholesky_inverse(Bn, W);
     world.gop.fence();  // N.B. seems to deadlock without this

     this->unNormalized_Factor = Bn.clone();
     this->normalize_factor(Bn);
     cp_factors[pos_plus_one] = Bn;
     auto& gram = this->partial_grammian[pos_plus_one];
     gram("r,rp") = Bn("r,n") * Bn("rp,n");
     THC_times_CPD[mode]("p,q") *= ref_orb_b("b,p") * cp_factors[pos_plus_one]("q,b");
   }
 }
 void update_factors_left(){

   // going through the core 0 is associated with factors 0 and 1
   // core 1 associated with factors 2 and 3 ...
   // First we need to take other side of the problem and contract it with the core
   // if core is greater than 2 I need to contract all the centers but
   // here there's only one so just do one contract
   DistArray<Tile, Policy> env, pq, W_env, W;
   {
     DistArray<Tile, Policy> An;
     env("p,q") = ref_core("p,r") * THC_times_CPD[1]("r,q");

     pq("p,q") = ref_orb_b("b,p") * cp_factors[1]("q,b");
     An("q,a") = (pq("p,q") *  env("p,q")) *  ref_orb_a("a,p");

     // TODO check to see if the Cholesky will fail. If it does
     // use SVD
     W_env("p,q") = this->partial_grammian[2]("p,q") *
                    this->partial_grammian[3]("p,q");
     W("p,q") = this->partial_grammian[1]("p,q") * W_env("p,q");

     this->cholesky_inverse(An, W);
     world.gop.fence();  // N.B. seems to deadlock without this

     this->normalize_factor(An);
     cp_factors[0] = An;
     this->partial_grammian[0]("r,rp") = An("r,n") * An("rp,n");
     pq("p,q") = ref_orb_a("a,p") * An("q,a");
     THC_times_CPD[0] = pq;
   }

   // Finished with the first factor in THC.
   // Starting the second factor
   {
     DistArray<Tile, Policy> Bn;
     Bn = DistArray<Tile, Policy>();
     Bn("q,b") = (pq("p,q") * env("p,q")) * ref_orb_b("b,p");

     // TODO check to see if the Cholesky will fail. If it does
     // use SVD
     W("p,q") = this->partial_grammian[0]("p,q") * W_env("p,q");
     this->cholesky_inverse(Bn, W);
     world.gop.fence();  // N.B. seems to deadlock without this

     this->normalize_factor(Bn);
     cp_factors[1] = Bn;
     this->partial_grammian[1]("r,rp") = Bn("r,n") * Bn("rp,n");
     THC_times_CPD[0]("p,q") *= ref_orb_b("b,p") * Bn("q,b");
   }
 }
 void update_factors_right(){

   // going through the core 0 is associated with factors 0 and 1
   // core 1 associated with factors 2 and 3 ...
   // First we need to take other side of the problem and contract it with the core
   // if core is greater than 2 I need to contract all the centers but
   // here there's only one so just do one contract
   DistArray<Tile, Policy> env, pq, W_env, W;
   {
     DistArray<Tile, Policy> An;
     env("p,q") = ref_core("r,p") * THC_times_CPD[0]("r,q");

     pq("p,q") = ref_orb_d("b,p") * cp_factors[3]("q,b");
     An("q,a") = (pq("p,q") *  env("p,q")) *  ref_orb_c("a,p");

     // TODO check to see if the Cholesky will fail. If it does
     // use SVD
     W_env("p,q") = this->partial_grammian[0]("p,q") *
                    this->partial_grammian[1]("p,q");
     W("p,q") = this->partial_grammian[3]("p,q") * W_env("p,q");

     this->cholesky_inverse(An, W);
     world.gop.fence();  // N.B. seems to deadlock without this

     this->normalize_factor(An);
     cp_factors[2] = An;
     this->partial_grammian[2]("r,rp") = An("r,n") * An("rp,n");
     pq("p,q") = ref_orb_c("a,p") * An("q,a");
     THC_times_CPD[1] = pq;
   }

   // Finished with the first factor in THC.
   // Starting the second factor
   {
     DistArray<Tile, Policy> Bn;
     Bn = DistArray<Tile, Policy>();
     Bn("q,b") = (pq("p,q") * env("p,q")) * ref_orb_d("b,p");

     this->MTtKRP = Bn;

     // TODO check to see if the Cholesky will fail. If it does
     // use SVD
     W("p,q") = this->partial_grammian[2]("p,q") * W_env("p,q");
     this->cholesky_inverse(Bn, W);
     world.gop.fence();  // N.B. seems to deadlock without this

     this->unNormalized_Factor = Bn.clone();
     this->normalize_factor(Bn);
     cp_factors[3] = Bn;
     this->partial_grammian[3]("r,rp") = Bn("r,n") * Bn("rp,n");
     THC_times_CPD[1]("p,q") *= ref_orb_d("b,p") * Bn("q,b");
   }
 }
};

}  // namespace TiledArray::math::cp

#endif  // TILEDARRAY_MATH_SOLVERS_CP_CP_THC_ALS__H
