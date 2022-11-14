/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#include <tiledarray.h>
#include <iomanip>
#include "input_data.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

int main(int argc, char** argv) {
  // Initialize runtime
  TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  std::string file_name = argv[1];

  // Open input file.
  std::ifstream input(file_name.c_str());

  if (!input.fail()) {
    // Read input data.

    if (world.rank() == 0) std::cout << "Reading input...";

    InputData data(input);
    input.close();

    if (world.rank() == 0) std::cout << " done.\nConstructing Fock tensors...";

    // Construct Fock tensor
    //    TArray2s f_a_oo = data.make_f(world, alpha, occ, occ);
    //    TArray2s f_a_vo = data.make_f(world, alpha, vir, occ);
    //    TArray2s f_a_ov = data.make_f(world, alpha, occ, vir);
    //    TArray2s f_a_vv = data.make_f(world, alpha, vir, vir);
    //    // Just make references to the data since the input is closed shell.
    //    TArray2s& f_b_oo = f_a_oo;
    //    TArray2s& f_b_vo = f_a_vo;
    //    TArray2s& f_b_ov = f_a_ov;
    //    TArray2s& f_b_vv = f_a_vv;
    //
    //    // Fence to make sure Fock tensors are initialized on all nodes
    //    world.gop.fence();
    //
    //    if(world.rank() == 0)
    //      std::cout << " done.\nConstructing v_ab tensors...";
    //
    //    // Construct the integral tensors
    //    TArray4s v_ab_oooo = data.make_v_ab(world, occ, occ, occ, occ);
    //    TArray4s v_ab_ooov = data.make_v_ab(world, occ, occ, occ, vir);
    //    TArray4s v_ab_oovo = data.make_v_ab(world, occ, occ, vir, occ);
    //    TArray4s v_ab_ovoo = data.make_v_ab(world, occ, vir, occ, occ);
    //    TArray4s v_ab_vooo = data.make_v_ab(world, vir, occ, occ, occ);
    //    TArray4s v_ab_vvoo = data.make_v_ab(world, vir, vir, occ, occ);
    //    TArray4s v_ab_oovv = data.make_v_ab(world, occ, occ, vir, vir);
    //    TArray4s v_ab_vovo = data.make_v_ab(world, vir, occ, vir, occ);
    //    TArray4s v_ab_ovov = data.make_v_ab(world, occ, vir, occ, vir);
    //    TArray4s v_ab_voov = data.make_v_ab(world, vir, occ, occ, vir);
    //    TArray4s v_ab_ovvo = data.make_v_ab(world, occ, vir, vir, occ);
    //    TArray4s v_ab_vvvo = data.make_v_ab(world, vir, vir, vir, occ);
    //    TArray4s v_ab_vvov = data.make_v_ab(world, vir, vir, occ, vir);
    //    TArray4s v_ab_vovv = data.make_v_ab(world, vir, occ, vir, vir);
    //    TArray4s v_ab_ovvv = data.make_v_ab(world, occ, vir, vir, vir);
    //    TArray4s v_ab_vvvv = data.make_v_ab(world, vir, vir, vir, vir);
    //
    //    // Fence to make sure data on all nodes has been initialized
    //    world.gop.fence();
    //
    //    if(world.rank() == 0)
    //      std::cout << " done.\nConstructing v_aa and v_bb tensors...";
    //
    //    TArray4s v_aa_oooo = v_ab_oooo("i,j,k,l") - v_ab_oooo("i,j,l,k");
    //    TArray4s v_aa_ooov = v_ab_ooov("i,j,k,a") - v_ab_oovo("i,j,a,k");
    //    TArray4s v_aa_vooo = v_ab_vooo("a,i,j,k") - v_ab_vooo("a,i,k,j");
    //    TArray4s v_aa_vvoo = v_ab_vvoo("a,b,i,j") - v_ab_vvoo("a,b,j,i");
    //    TArray4s v_aa_vovo = v_ab_vovo("a,i,b,j") - v_ab_voov("a,i,j,b");
    //    TArray4s v_aa_oovv = v_ab_oovv("i,j,a,b") - v_ab_oovv("i,j,b,a");
    //    TArray4s v_aa_voov = v_ab_voov("a,i,j,b") - v_ab_vovo("a,i,b,j");
    //    TArray4s v_aa_ovvo = v_ab_ovvo("i,a,b,j") - v_ab_ovov("i,a,j,b");
    //    TArray4s v_aa_vovv = v_ab_vovv("a,i,b,c") - v_ab_vovv("a,i,c,b");
    //    TArray4s v_aa_vvov = v_ab_vvov("a,b,i,c") - v_ab_vvvo("a,b,c,i");
    //    TArray4s v_aa_vvvv = v_ab_vvvv("a,b,c,d") - v_ab_vvvv("a,b,d,c");
    //    // Just make references to the data since the input is closed shell.
    //    TArray4s& v_bb_oooo = v_aa_oooo;
    //    TArray4s& v_bb_ooov = v_aa_ooov;
    //    TArray4s& v_bb_vvoo = v_aa_vvoo;
    //    TArray4s& v_bb_vovo = v_aa_vovo;
    //    TArray4s& v_bb_oovv = v_aa_oovv;
    //    TArray4s& v_bb_voov = v_aa_voov;
    //    TArray4s& v_bb_vovv = v_aa_vovv;
    //    TArray4s& v_bb_vvvv = v_aa_vvvv;
    //
    //    // Fence again to make sure data all the integral tensors have been
    //    initialized world.gop.fence();
    //
    //    if(world.rank() == 0)
    //      std::cout << " done.\n";
    //
    //
    //    TArray2s t_a_vo(world, f_a_vo.trange(), f_a_vo.shape());
    //    t_a_vo.set_all_local(0.0);
    //
    //    TArray2s& t_b_vo = t_a_vo;
    //
    //    TArray4s t_aa_vvoo(world, v_aa_vvoo.trange(), v_aa_vvoo.shape());
    //    t_aa_vvoo.set_all_local(0.0);
    //
    //    TArray4s t_ab_vvoo(world, v_ab_vvoo.trange(), v_ab_vvoo.shape());
    //    t_ab_vvoo.set_all_local(0.0);
    //
    //    TArray4s t_bb_vvoo(world, v_bb_vvoo.trange(), v_bb_vvoo.shape());
    //    t_bb_vvoo.set_all_local(0.0);
    //
    //
    //    TArray2s D_vo(world, f_a_vo.trange(), f_a_vo.shape());
    //    for(TArray2s::range_type::const_iterator it =
    //    D_vo.tiles_range().begin(); it
    //    != D_vo.tiles_range().end(); ++it)
    //      if(D_vo.is_local(*it) && (! D_vo.is_zero(*it)))
    //        D_vo.set(*it, world.taskq.add(data, & InputData::make_D_vo_tile,
    //        D_vo.trange().make_tile_range(*it)));
    //
    //    TArray4s D_vvoo(world, v_ab_vvoo.trange(), v_ab_vvoo.shape());
    //    for(TArray4s::range_type::const_iterator it =
    //    D_vvoo.tiles_range().begin(); it != D_vvoo.tiles_range().end(); ++it)
    //      if(D_vvoo.is_local(*it) && (! D_vvoo.is_zero(*it)))
    //        D_vvoo.set(*it, world.taskq.add(data, &
    //        InputData::make_D_vvoo_tile,
    //        D_vvoo.trange().make_tile_range(*it)));

    world.gop.fence();

    data.clear();

    if (world.rank() == 0) std::cout << "Calculating t amplitudes...\n";

    double energy = 0.0;

    for (unsigned int i = 0ul; i < 100; ++i) {
      if (world.rank() == 0) std::cout << "Iteration " << i << "\n";

      //      TArray2s r_a_vo = f_a_vo("p1a,h1a");
      //
      //      world.gop.fence();
      //
      //      r_a_vo("p1a,h1a") = r_a_vo("p1a,h1a")
      //          -0.5*(
      //              t_aa_vvoo("p1a,p2a,h2a,h5a")*v_aa_ooov("h2a,h5a,h1a,p2a")
      //              +t_aa_vvoo("p1a,p3a,h2a,h3a")*t_a_vo("p2a,h1a")*v_aa_oovv("h2a,h3a,p2a,p3a")
      //              +t_aa_vvoo("p2a,p3a,h1a,h3a")*t_a_vo("p1a,h2a")*v_aa_oovv("h2a,h3a,p2a,p3a")
      //              -t_aa_vvoo("p2a,p5a,h1a,h2a")*v_aa_vovv("p1a,h2a,p2a,p5a")
      //          );
      //
      //      world.gop.fence();
      //
      //      r_a_vo("p1a,h1a") = r_a_vo("p1a,h1a")
      //          -f_a_oo("h2a,h1a")*t_a_vo("p1a,h2a")
      //          -f_a_ov("h2a,p2a")*t_a_vo("p1a,h2a")*t_a_vo("p2a,h1a")
      //          -t_ab_vvoo("p1a,p4b,h2a,h4b")*t_a_vo("p2a,h1a")*v_ab_oovv("h2a,h4b,p2a,p4b")
      //          -t_ab_vvoo("p1a,p4b,h2a,h4b")*v_ab_ooov("h2a,h4b,h1a,p4b");
      //
      //      world.gop.fence();
      //
      //      r_a_vo("p1a,h1a") = r_a_vo("p1a,h1a")
      //          -t_ab_vvoo("p2a,p4b,h1a,h4b")*t_a_vo("p1a,h2a")*v_ab_oovv("h2a,h4b,p2a,p4b")
      //          -t_a_vo("p1a,h2a")*t_a_vo("p2a,h1a")*t_a_vo("p3a,h3a")*v_aa_oovv("h2a,h3a,p2a,p3a")
      //          -t_a_vo("p1a,h2a")*t_a_vo("p2a,h1a")*t_b_vo("p4b,h4b")*v_ab_oovv("h2a,h4b,p2a,p4b")
      //          -t_a_vo("p1a,h2a")*t_a_vo("p2a,h3a")*v_aa_ooov("h2a,h3a,h1a,p2a");
      //
      //      world.gop.fence();
      //
      //      r_a_vo("p1a,h1a") = r_a_vo("p1a,h1a")
      //          -t_a_vo("p1a,h2a")*t_b_vo("p4b,h4b")*v_ab_ooov("h2a,h4b,h1a,p4b")
      //          +f_a_ov("h2a,p2a")*t_aa_vvoo("p1a,p2a,h1a,h2a")
      //          +f_a_vv("p1a,p2a")*t_a_vo("p2a,h1a")
      //          +f_b_ov("h4b,p4b")*t_ab_vvoo("p1a,p4b,h1a,h4b");
      //
      //      world.gop.fence();
      //
      //      r_a_vo("p1a,h1a") = r_a_vo("p1a,h1a")
      //          +t_aa_vvoo("p1a,p2a,h1a,h2a")*t_a_vo("p3a,h3a")*v_aa_oovv("h2a,h3a,p2a,p3a")
      //          +t_aa_vvoo("p1a,p2a,h1a,h2a")*t_b_vo("p4b,h4b")*v_ab_oovv("h2a,h4b,p2a,p4b")
      //          +t_ab_vvoo("p1a,p4b,h1a,h4b")*t_a_vo("p2a,h2a")*v_ab_oovv("h2a,h4b,p2a,p4b")
      //          +t_ab_vvoo("p1a,p4b,h1a,h4b")*t_b_vo("p6b,h6b")*v_bb_oovv("h4b,h6b,p4b,p6b");
      //
      //      world.gop.fence();
      //
      //      r_a_vo("p1a,h1a") = r_a_vo("p1a,h1a")
      //          +t_ab_vvoo("p2a,p4b,h1a,h4b")*v_ab_vovv("p1a,h4b,p2a,p4b")
      //          +t_a_vo("p2a,h1a")*t_a_vo("p3a,h2a")*v_aa_vovv("p1a,h2a,p2a,p3a")
      //          +t_a_vo("p2a,h1a")*t_b_vo("p4b,h4b")*v_ab_vovv("p1a,h4b,p2a,p4b")
      //          +t_a_vo("p2a,h2a")*v_aa_voov("p1a,h2a,h1a,p2a")
      //          +t_b_vo("p4b,h4b")*v_ab_voov("p1a,h4b,h1a,p4b");
      //
      //      TArray2s& r_b_vo = r_a_vo;
      //
      //      world.gop.fence();
      //
      //      TArray4s r_aa_vvoo = v_aa_vvoo("p7a,p8a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +0.25*t_aa_vvoo("p7a,p8a,h9a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a")*t_aa_vvoo("p9a,p10a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +0.5*t_aa_vvoo("p7a,p10a,h9a,h10a")*t_aa_vvoo("p8a,p9a,h7a,h8a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            -0.5*t_aa_vvoo("p7a,p8a,h7a,h9a")*t_aa_vvoo("p9a,p10a,h8a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            +0.5*t_aa_vvoo("p7a,p8a,h8a,h9a")*t_aa_vvoo("p9a,p10a,h7a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +0.5*t_aa_vvoo("p7a,p8a,h9a,h10a")*t_a_vo("p10a,h8a")*t_a_vo("p9a,h7a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            -0.5*t_aa_vvoo("p7a,p8a,h9a,h10a")*t_a_vo("p9a,h7a")*v_aa_ooov("h9a,h10a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            +0.5*t_aa_vvoo("p7a,p8a,h9a,h10a")*t_a_vo("p9a,h8a")*v_aa_ooov("h9a,h10a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +0.5*t_aa_vvoo("p7a,p8a,h9a,h12a")*v_aa_oooo("h9a,h12a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            -0.5*t_aa_vvoo("p7a,p9a,h7a,h8a")*t_aa_vvoo("p8a,p10a,h9a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            +0.5*t_aa_vvoo("p9a,p10a,h7a,h8a")*t_a_vo("p7a,h9a")*t_a_vo("p8a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +0.5*t_aa_vvoo("p9a,p10a,h7a,h8a")*t_a_vo("p7a,h9a")*v_aa_vovv("p8a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            -0.5*t_aa_vvoo("p9a,p10a,h7a,h8a")*t_a_vo("p8a,h9a")*v_aa_vovv("p7a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            +0.5*t_aa_vvoo("p9a,p12a,h7a,h8a")*v_aa_vvvv("p7a,p8a,p9a,p12a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -0.5*t_a_vo("p10a,h7a")*t_a_vo("p9a,h8a")*v_aa_vvvv("p7a,p8a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            +0.5*t_a_vo("p10a,h8a")*t_a_vo("p9a,h7a")*v_aa_vvvv("p7a,p8a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            -0.5*t_a_vo("p7a,h10a")*t_a_vo("p8a,h9a")*v_aa_oooo("h9a,h10a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //            +0.5*t_a_vo("p7a,h9a")*t_a_vo("p8a,h10a")*v_aa_oooo("h9a,h10a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -f_a_oo("h9a,h8a")*t_aa_vvoo("p7a,p8a,h7a,h9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -f_a_ov("h9a,p9a")*t_aa_vvoo("p7a,p8a,h7a,h9a")*t_a_vo("p9a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -f_a_ov("h9a,p9a")*t_aa_vvoo("p7a,p9a,h7a,h8a")*t_a_vo("p8a,h9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -f_a_vv("p7a,p9a")*t_aa_vvoo("p8a,p9a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p8a,h7a,h9a")*t_ab_vvoo("p9a,p11b,h8a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p8a,h7a,h9a")*t_a_vo("p10a,h10a")*t_a_vo("p9a,h8a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p8a,h7a,h9a")*t_a_vo("p9a,h10a")*v_aa_ooov("h9a,h10a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p8a,h7a,h9a")*t_a_vo("p9a,h8a")*t_b_vo("p11b,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p8a,h7a,h9a")*t_b_vo("p11b,h11b")*v_ab_ooov("h9a,h11b,h8a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h7a,h8a")*t_ab_vvoo("p8a,p11b,h9a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h7a,h8a")*t_a_vo("p10a,h10a")*t_a_vo("p8a,h9a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h7a,h8a")*t_a_vo("p8a,h9a")*t_b_vo("p11b,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h7a,h9a")*t_a_vo("p10a,h8a")*t_a_vo("p8a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h7a,h9a")*t_a_vo("p10a,h8a")*v_aa_vovv("p8a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h8a,h9a")*t_aa_vvoo("p8a,p10a,h7a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h8a,h9a")*t_ab_vvoo("p8a,p11b,h7a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h8a,h9a")*t_a_vo("p8a,h10a")*v_aa_ooov("h9a,h10a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p7a,p9a,h8a,h9a")*v_aa_voov("p8a,h9a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p8a,p9a,h7a,h8a")*t_a_vo("p10a,h9a")*v_aa_vovv("p7a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p8a,p9a,h7a,h8a")*t_b_vo("p11b,h11b")*v_ab_vovv("p7a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p8a,p9a,h7a,h9a")*t_ab_vvoo("p7a,p11b,h8a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p8a,p9a,h7a,h9a")*t_a_vo("p7a,h10a")*v_aa_ooov("h9a,h10a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p8a,p9a,h7a,h9a")*v_aa_voov("p7a,h9a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p8a,p9a,h8a,h9a")*t_a_vo("p10a,h7a")*t_a_vo("p7a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_aa_vvoo("p8a,p9a,h8a,h9a")*t_a_vo("p10a,h7a")*v_aa_vovv("p7a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p7a,p11b,h7a,h11b")*t_a_vo("p8a,h9a")*t_a_vo("p9a,h8a")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p7a,p11b,h7a,h11b")*t_a_vo("p8a,h9a")*v_ab_ooov("h9a,h11b,h8a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p7a,p11b,h8a,h11b")*t_ab_vvoo("p8a,p14b,h7a,h14b")*v_bb_oovv("h11b,h14b,p11b,p14b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p7a,p11b,h8a,h11b")*t_a_vo("p9a,h7a")*v_ab_vovv("p8a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p7a,p13b,h8a,h13b")*v_ab_voov("p8a,h13b,h7a,p13b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p8a,p11b,h7a,h11b")*t_a_vo("p9a,h8a")*v_ab_vovv("p7a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p8a,p11b,h8a,h11b")*t_a_vo("p7a,h9a")*t_a_vo("p9a,h7a")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p8a,p11b,h8a,h11b")*t_a_vo("p7a,h9a")*v_ab_ooov("h9a,h11b,h7a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_ab_vvoo("p8a,p13b,h7a,h13b")*v_ab_voov("p7a,h13b,h8a,p13b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_a_vo("p10a,h8a")*t_a_vo("p8a,h9a")*t_a_vo("p9a,h7a")*v_aa_vovv("p7a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_a_vo("p7a,h9a")*t_a_vo("p8a,h10a")*t_a_vo("p9a,h7a")*v_aa_ooov("h9a,h10a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_a_vo("p7a,h9a")*t_a_vo("p9a,h7a")*v_aa_voov("p8a,h9a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_a_vo("p8a,h9a")*t_a_vo("p9a,h8a")*v_aa_voov("p7a,h9a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_a_vo("p8a,h9a")*v_aa_vooo("p7a,h9a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          -t_a_vo("p9a,h7a")*v_aa_vvov("p7a,p8a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +f_a_oo("h9a,h7a")*t_aa_vvoo("p7a,p8a,h8a,h9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +f_a_ov("h9a,p9a")*t_aa_vvoo("p7a,p8a,h8a,h9a")*t_a_vo("p9a,h7a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +f_a_ov("h9a,p9a")*t_aa_vvoo("p8a,p9a,h7a,h8a")*t_a_vo("p7a,h9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +f_a_vv("p8a,p9a")*t_aa_vvoo("p7a,p9a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p8a,h8a,h9a")*t_ab_vvoo("p9a,p11b,h7a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p8a,h8a,h9a")*t_a_vo("p10a,h10a")*t_a_vo("p9a,h7a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p8a,h8a,h9a")*t_a_vo("p9a,h10a")*v_aa_ooov("h9a,h10a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p8a,h8a,h9a")*t_a_vo("p9a,h7a")*t_b_vo("p11b,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p8a,h8a,h9a")*t_b_vo("p11b,h11b")*v_ab_ooov("h9a,h11b,h7a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h7a,h8a")*t_a_vo("p10a,h9a")*v_aa_vovv("p8a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h7a,h8a")*t_b_vo("p11b,h11b")*v_ab_vovv("p8a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h7a,h9a")*t_aa_vvoo("p8a,p10a,h8a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h7a,h9a")*t_ab_vvoo("p8a,p11b,h8a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h7a,h9a")*t_a_vo("p8a,h10a")*v_aa_ooov("h9a,h10a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h7a,h9a")*v_aa_voov("p8a,h9a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h8a,h9a")*t_a_vo("p10a,h7a")*t_a_vo("p8a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p7a,p9a,h8a,h9a")*t_a_vo("p10a,h7a")*v_aa_vovv("p8a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h7a,h8a")*t_ab_vvoo("p7a,p11b,h9a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h7a,h8a")*t_a_vo("p10a,h10a")*t_a_vo("p7a,h9a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h7a,h8a")*t_a_vo("p7a,h9a")*t_b_vo("p11b,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h7a,h9a")*t_a_vo("p10a,h8a")*t_a_vo("p7a,h10a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h7a,h9a")*t_a_vo("p10a,h8a")*v_aa_vovv("p7a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h8a,h9a")*t_ab_vvoo("p7a,p11b,h7a,h11b")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h8a,h9a")*t_a_vo("p7a,h10a")*v_aa_ooov("h9a,h10a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_aa_vvoo("p8a,p9a,h8a,h9a")*v_aa_voov("p7a,h9a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p7a,p11b,h7a,h11b")*t_ab_vvoo("p8a,p14b,h8a,h14b")*v_bb_oovv("h11b,h14b,p11b,p14b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p7a,p11b,h7a,h11b")*t_a_vo("p9a,h8a")*v_ab_vovv("p8a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p7a,p11b,h8a,h11b")*t_a_vo("p8a,h9a")*t_a_vo("p9a,h7a")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p7a,p11b,h8a,h11b")*t_a_vo("p8a,h9a")*v_ab_ooov("h9a,h11b,h7a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p7a,p13b,h7a,h13b")*v_ab_voov("p8a,h13b,h8a,p13b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p8a,p11b,h7a,h11b")*t_a_vo("p7a,h9a")*t_a_vo("p9a,h8a")*v_ab_oovv("h9a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p8a,p11b,h7a,h11b")*t_a_vo("p7a,h9a")*v_ab_ooov("h9a,h11b,h8a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p8a,p11b,h8a,h11b")*t_a_vo("p9a,h7a")*v_ab_vovv("p7a,h11b,p9a,p11b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_ab_vvoo("p8a,p13b,h8a,h13b")*v_ab_voov("p7a,h13b,h7a,p13b");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_a_vo("p10a,h8a")*t_a_vo("p7a,h9a")*t_a_vo("p8a,h10a")*t_a_vo("p9a,h7a")*v_aa_oovv("h9a,h10a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_a_vo("p10a,h8a")*t_a_vo("p7a,h9a")*t_a_vo("p9a,h7a")*v_aa_vovv("p8a,h9a,p9a,p10a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_a_vo("p7a,h9a")*t_a_vo("p8a,h10a")*t_a_vo("p9a,h8a")*v_aa_ooov("h9a,h10a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_a_vo("p7a,h9a")*t_a_vo("p9a,h8a")*v_aa_voov("p8a,h9a,h7a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_a_vo("p7a,h9a")*v_aa_vooo("p8a,h9a,h7a,h8a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_a_vo("p8a,h9a")*t_a_vo("p9a,h7a")*v_aa_voov("p7a,h9a,h8a,p9a");
      //
      //      world.gop.fence();
      //
      //      r_aa_vvoo("p7a,p8a,h7a,h8a") = r_aa_vvoo("p7a,p8a,h7a,h8a")
      //          +t_a_vo("p9a,h8a")*v_aa_vvov("p7a,p8a,h7a,p9a");
      //
      //      TArray4s& r_bb_vvoo = r_aa_vvoo;
      //
      //      TArray4s r_ab_vvoo = v_ab_vvoo("p17a,p16b,h17a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -0.5*t_aa_vvoo("p17a,p20a,h18a,h20a")*t_ab_vvoo("p18a,p16b,h17a,h15b")*v_aa_oovv("h18a,h20a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //              -0.5*t_aa_vvoo("p18a,p20a,h17a,h20a")*t_ab_vvoo("p17a,p16b,h18a,h15b")*v_aa_oovv("h18a,h20a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //              -0.5*t_ab_vvoo("p17a,p15b,h17a,h15b")*t_bb_vvoo("p16b,p19b,h16b,h19b")*v_bb_oovv("h16b,h19b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //              -0.5*t_ab_vvoo("p17a,p16b,h17a,h16b")*t_bb_vvoo("p15b,p19b,h15b,h19b")*v_bb_oovv("h16b,h19b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -f_a_oo("h18a,h17a")*t_ab_vvoo("p17a,p16b,h18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -f_a_ov("h18a,p18a")*t_ab_vvoo("p17a,p16b,h18a,h15b")*t_a_vo("p18a,h17a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -f_a_ov("h18a,p18a")*t_ab_vvoo("p18a,p16b,h17a,h15b")*t_a_vo("p17a,h18a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -f_b_oo("h16b,h15b")*t_ab_vvoo("p17a,p16b,h17a,h16b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -f_b_ov("h16b,p15b")*t_ab_vvoo("p17a,p15b,h17a,h15b")*t_b_vo("p16b,h16b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -f_b_ov("h16b,p15b")*t_ab_vvoo("p17a,p16b,h17a,h16b")*t_b_vo("p15b,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_aa_vvoo("p17a,p18a,h17a,h18a")*t_b_vo("p15b,h15b")*t_b_vo("p16b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_aa_vvoo("p17a,p18a,h17a,h18a")*t_b_vo("p16b,h16b")*v_ab_oovo("h18a,h16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h17a,h15b")*t_ab_vvoo("p18a,p16b,h18a,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h17a,h15b")*t_a_vo("p18a,h18a")*t_b_vo("p16b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h17a,h15b")*t_b_vo("p16b,h16b")*t_b_vo("p19b,h19b")*v_bb_oovv("h16b,h19b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h17a,h16b")*t_b_vo("p16b,h19b")*t_b_vo("p19b,h15b")*v_bb_oovv("h16b,h19b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h17a,h16b")*t_b_vo("p19b,h15b")*v_bb_vovv("p16b,h16b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h18a,h15b")*t_a_vo("p18a,h17a")*v_ab_ovvv("h18a,p16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h18a,h15b")*v_ab_ovov("h18a,p16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p15b,h18a,h16b")*t_ab_vvoo("p18a,p16b,h17a,h15b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h17a,h16b")*t_ab_vvoo("p18a,p15b,h18a,h15b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h17a,h16b")*t_a_vo("p18a,h18a")*t_b_vo("p15b,h15b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h17a,h16b")*t_a_vo("p18a,h18a")*v_ab_oovo("h18a,h16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h17a,h16b")*t_b_vo("p15b,h15b")*t_b_vo("p19b,h19b")*v_bb_oovv("h16b,h19b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h17a,h16b")*t_b_vo("p15b,h19b")*v_bb_ooov("h16b,h19b,h15b,p15b")
      //          -t_ab_vvoo("p17a,p16b,h18a,h15b")*t_ab_vvoo("p18a,p15b,h17a,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h18a,h15b")*t_a_vo("p18a,h17a")*t_a_vo("p20a,h20a")*v_aa_oovv("h18a,h20a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h18a,h15b")*t_a_vo("p18a,h17a")*t_b_vo("p15b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h18a,h15b")*t_a_vo("p18a,h20a")*v_aa_ooov("h18a,h20a,h17a,p18a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p17a,p16b,h18a,h15b")*t_b_vo("p15b,h16b")*v_ab_ooov("h18a,h16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p15b,h17a,h15b")*t_a_vo("p17a,h18a")*v_ab_ovvv("h18a,p16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p15b,h17a,h15b")*t_b_vo("p16b,h16b")*v_ab_vovv("p17a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p16b,h17a,h15b")*t_a_vo("p17a,h18a")*t_a_vo("p20a,h20a")*v_aa_oovv("h18a,h20a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p16b,h17a,h15b")*t_a_vo("p17a,h18a")*t_b_vo("p15b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p16b,h17a,h16b")*t_b_vo("p15b,h15b")*v_ab_vovv("p17a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p16b,h17a,h16b")*v_ab_vovo("p17a,h16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p16b,h18a,h15b")*t_a_vo("p17a,h20a")*t_a_vo("p20a,h17a")*v_aa_oovv("h18a,h20a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_ab_vvoo("p18a,p16b,h18a,h15b")*t_a_vo("p20a,h17a")*v_aa_vovv("p17a,h18a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p17a,h18a")*t_a_vo("p18a,h17a")*t_bb_vvoo("p16b,p15b,h15b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p17a,h18a")*t_a_vo("p18a,h17a")*t_b_vo("p15b,h15b")*v_ab_ovvv("h18a,p16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p17a,h18a")*t_a_vo("p18a,h17a")*v_ab_ovvo("h18a,p16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p17a,h18a")*t_bb_vvoo("p16b,p15b,h15b,h16b")*v_ab_ooov("h18a,h16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p17a,h18a")*t_b_vo("p15b,h15b")*v_ab_ovov("h18a,p16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p17a,h18a")*v_ab_ovoo("h18a,p16b,h17a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p18a,h17a")*t_b_vo("p15b,h15b")*t_b_vo("p16b,h16b")*v_ab_vovv("p17a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_a_vo("p18a,h17a")*t_b_vo("p16b,h16b")*v_ab_vovo("p17a,h16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_b_vo("p15b,h15b")*t_b_vo("p16b,h16b")*v_ab_voov("p17a,h16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          -t_b_vo("p16b,h16b")*v_ab_vooo("p17a,h16b,h17a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +f_a_vv("p17a,p18a")*t_ab_vvoo("p18a,p16b,h17a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +f_b_vv("p16b,p15b")*t_ab_vvoo("p17a,p15b,h17a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_aa_vvoo("p17a,p18a,h17a,h18a")*t_ab_vvoo("p20a,p16b,h20a,h15b")*v_aa_oovv("h18a,h20a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_aa_vvoo("p17a,p18a,h17a,h18a")*t_bb_vvoo("p16b,p15b,h15b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_aa_vvoo("p17a,p18a,h17a,h18a")*t_b_vo("p15b,h15b")*v_ab_ovvv("h18a,p16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_aa_vvoo("p17a,p18a,h17a,h18a")*v_ab_ovvo("h18a,p16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h17a,h15b")*t_a_vo("p18a,h18a")*v_ab_ovvv("h18a,p16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h17a,h15b")*t_b_vo("p19b,h16b")*v_bb_vovv("p16b,h16b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h17a,h16b")*t_ab_vvoo("p18a,p16b,h18a,h15b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h17a,h16b")*t_b_vo("p16b,h19b")*v_bb_ooov("h16b,h19b,h15b,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h17a,h16b")*v_bb_voov("p16b,h16b,h15b,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h18a,h15b")*t_ab_vvoo("p18a,p16b,h17a,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h18a,h15b")*t_a_vo("p18a,h17a")*t_b_vo("p16b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p15b,h18a,h15b")*t_b_vo("p16b,h16b")*v_ab_ooov("h18a,h16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p16b,h18a,h16b")*t_ab_vvoo("p18a,p15b,h17a,h15b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p16b,h18a,h16b")*t_a_vo("p18a,h17a")*t_b_vo("p15b,h15b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p16b,h18a,h16b")*t_a_vo("p18a,h17a")*v_ab_oovo("h18a,h16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p16b,h18a,h16b")*t_b_vo("p15b,h15b")*v_ab_ooov("h18a,h16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p16b,h18a,h16b")*v_ab_oooo("h18a,h16b,h17a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p17a,p19b,h17a,h19b")*t_bb_vvoo("p16b,p15b,h15b,h16b")*v_bb_oovv("h16b,h19b,p15b,p19b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p15b,h17a,h15b")*t_a_vo("p17a,h18a")*t_b_vo("p16b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p15b,h17a,h15b")*v_ab_vvvv("p17a,p16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p16b,h17a,h15b")*t_a_vo("p20a,h18a")*v_aa_vovv("p17a,h18a,p18a,p20a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p16b,h17a,h15b")*t_b_vo("p15b,h16b")*v_ab_vovv("p17a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p16b,h17a,h16b")*t_a_vo("p17a,h18a")*t_b_vo("p15b,h15b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p16b,h17a,h16b")*t_a_vo("p17a,h18a")*v_ab_oovo("h18a,h16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p16b,h18a,h15b")*t_a_vo("p17a,h20a")*v_aa_ooov("h18a,h20a,h17a,p18a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_ab_vvoo("p18a,p16b,h18a,h15b")*v_aa_voov("p17a,h18a,h17a,p18a");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_a_vo("p17a,h18a")*t_a_vo("p18a,h17a")*t_b_vo("p15b,h15b")*t_b_vo("p16b,h16b")*v_ab_oovv("h18a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_a_vo("p17a,h18a")*t_a_vo("p18a,h17a")*t_b_vo("p16b,h16b")*v_ab_oovo("h18a,h16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_a_vo("p17a,h18a")*t_b_vo("p15b,h15b")*t_b_vo("p16b,h16b")*v_ab_ooov("h18a,h16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_a_vo("p17a,h18a")*t_b_vo("p16b,h16b")*v_ab_oooo("h18a,h16b,h17a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_a_vo("p18a,h17a")*t_bb_vvoo("p16b,p15b,h15b,h16b")*v_ab_vovv("p17a,h16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_a_vo("p18a,h17a")*t_b_vo("p15b,h15b")*v_ab_vvvv("p17a,p16b,p18a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_a_vo("p18a,h17a")*v_ab_vvvo("p17a,p16b,p18a,h15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_bb_vvoo("p16b,p15b,h15b,h16b")*v_ab_voov("p17a,h16b,h17a,p15b");
      //
      //      world.gop.fence();
      //
      //      r_ab_vvoo = r_ab_vvoo("p17a,p16b,h17a,h15b")
      //          +t_b_vo("p15b,h15b")*v_ab_vvov("p17a,p16b,h17a,p15b");
      //
      //
      //      world.gop.fence();
      //
      //      t_aa_vvoo("a,b,i,j") =
      //          make_binary_tiled_tensor(D_vvoo("a,b,i,j"),
      //          r_aa_vvoo("a,b,i,j"), std::multiplies<double>())
      //          + t_aa_vvoo("a,b,i,j");
      //
      //      t_ab_vvoo("a,b,i,j") =
      //          make_binary_tiled_tensor(D_vvoo("a,b,i,j"),
      //          r_ab_vvoo("a,b,i,j"), std::multiplies<double>())
      //          + t_ab_vvoo("a,b,i,j");
      //
      //      t_bb_vvoo("a,b,i,j") =
      //          make_binary_tiled_tensor(D_vvoo("a,b,i,j"),
      //          r_bb_vvoo("a,b,i,j"), std::multiplies<double>())
      //          + t_bb_vvoo("a,b,i,j");
      //
      //
      //      world.gop.fence();
      //
      //      TArray4s tau_aa_vvoo =
      //          t_aa_vvoo("a,b,i,j") + t_a_vo("a,i") * t_a_vo("b,j") -
      //          t_a_vo("b,i") * t_a_vo("a,j");
      //
      //      TArray4s tau_bb_vvoo =
      //          t_bb_vvoo("a,b,i,j") + t_b_vo("a,i") * t_b_vo("b,j") -
      //          t_b_vo("b,i") * t_b_vo("a,j");
      //
      //      TArray4s tau_ab_vvoo =
      //          t_ab_vvoo("a,b,i,j") + t_a_vo("a,i") * t_b_vo("b,j");
      //
      //
      //      world.gop.fence();
      //
      //      const double error =
      //      TiledArray::expressions::norm2(r_aa_vvoo("a,b,i,j")
      //          + r_ab_vvoo("a,b,i,j") + r_bb_vvoo("a,b,i,j"));
      //
      //      energy = 0.25 * (dot(tau_aa_vvoo("a,b,i,j"), v_aa_vvoo("a,b,i,j"))
      //          +dot(tau_bb_vvoo("a,b,i,j"), v_bb_vvoo("a,b,i,j")) )
      //          +dot(tau_ab_vvoo("a,b,i,j"), v_ab_vvoo("a,b,i,j"))
      //          +dot(t_a_vo("a,i"), f_a_vo("a,i"))
      //          +dot(t_b_vo("a,i"), f_b_vo("a,i"));
      //
      //      world.gop.fence();
      //
      //      if(world.rank() == 0)
      //        std::cout << " error  = "  << std::setprecision(12) << error <<
      //        "\n"
      //                  << " energy = " << std::setprecision(12) << energy <<
      //                  "\n";
      //
      //      if(error < 1.0e-10)
      //        break;
    }

    if (world.rank() == 0) {
      std::cout << "CCSD energy = " << std::setprecision(12) << energy << "\n";
      std::cout << "Done!\n";
    }

  } else {
    std::cout << "Unable to open file: " << file_name << "\n";
    return 1;
  }

  return 0;
}
