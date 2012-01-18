//============================================================================
// Name        : CCD.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   :
// Description : Hello World in C, Ansi-style
//============================================================================

#include <iomanip>
#include <tiled_array.h>
#include "input_data.h"

using namespace TiledArray;

int main(int argc, char** argv) {
  // Initialize madness runtime
  madness::initialize(argc,argv);
  std::string file_name = argv[1];

  // Open input file.
  std::ifstream input(file_name.c_str());

  if(! input.fail()) {
    // Make the world
    madness::World world(MPI::COMM_WORLD);

    // Read input data.

    std::cout << "Reading input...";
    InputData data(input);
    input.close();

    std::cout << " done.\nConstructing Fock tensors...";

    // Construct Fock tensor
    Array<double, CoordinateSystem<2> > f_a_oo = data.make_f(world, alpha, occ, occ);
    Array<double, CoordinateSystem<2> > f_a_vv = data.make_f(world, alpha, vir, vir);
    // Just make references to the data since the input is closed shell.
    Array<double, CoordinateSystem<2> >& f_b_oo = f_a_oo;
    Array<double, CoordinateSystem<2> >& f_b_vv = f_a_vv;

    // Fence to make sure Fock tensors are initialized on all nodes
    world.gop.fence();
    std::cout << " done.\nConstructing v_ab tensors...";

    // Construct the integral tensors
    Array<double, CoordinateSystem<4> > v_ab_oooo = data.make_v_ab(world, occ, occ, occ, occ);
    Array<double, CoordinateSystem<4> > v_ab_vvoo = data.make_v_ab(world, vir, vir, occ, occ);
    Array<double, CoordinateSystem<4> > v_ab_vovo = data.make_v_ab(world, vir, occ, vir, occ);
    Array<double, CoordinateSystem<4> > v_ab_voov = data.make_v_ab(world, vir, occ, occ, vir);
    Array<double, CoordinateSystem<4> > v_ab_vvvv = data.make_v_ab(world, vir, vir, vir, vir);

    // Fence to make sure data on all nodes has been initialized
    world.gop.fence();
    std::cout << " done.\nConstructing v_aa and v_bb tensors...";

    Array<double, CoordinateSystem<4> > v_aa_oooo = v_ab_oooo("i,j,k,l") - v_ab_oooo("i,j,l,k");
    Array<double, CoordinateSystem<4> > v_aa_vvoo = v_ab_vvoo("a,b,i,j") - v_ab_vvoo("a,b,j,i");
    Array<double, CoordinateSystem<4> > v_aa_vovo = v_ab_vovo("a,i,b,j") - v_ab_voov("a,i,j,b");
    Array<double, CoordinateSystem<4> > v_aa_vvvv = v_ab_vvvv("a,b,c,d") - v_ab_vvvv("a,b,d,c");
    // Just make references to the data since the input is closed shell.
    Array<double, CoordinateSystem<4> >& v_bb_oooo = v_aa_oooo;
    Array<double, CoordinateSystem<4> >& v_bb_vvoo = v_aa_vvoo;
    Array<double, CoordinateSystem<4> >& v_bb_vovo = v_aa_vovo;
    Array<double, CoordinateSystem<4> >& v_bb_vvvv = v_aa_vvvv;

    // Fence again to make sure data all the integral tensors have been initialized
    world.gop.fence();
    std::cout << " done.\n";


    Array<double, CoordinateSystem<4> > t_aa_vvoo(world, v_aa_vvoo.trange(), v_aa_vvoo.get_shape());
    for(Array<double, CoordinateSystem<4> >::range_type::const_iterator it = t_aa_vvoo.range().begin(); it != t_aa_vvoo.range().end(); ++it)
      if(t_aa_vvoo.is_local(*it) && (! t_aa_vvoo.is_zero(*it)))
        t_aa_vvoo.set(*it, 0.0);

    Array<double, CoordinateSystem<4> > D_aa_vvoo(world, v_aa_vvoo.trange(), v_aa_vvoo.get_shape());
    for(Array<double, CoordinateSystem<4> >::range_type::const_iterator it = D_aa_vvoo.range().begin(); it != D_aa_vvoo.range().end(); ++it)
      if(D_aa_vvoo.is_local(*it) && (! D_aa_vvoo.is_zero(*it)))
        D_aa_vvoo.set(*it, world.taskq.add(data, & InputData::make_D_tile, D_aa_vvoo.trange().make_tile_range(*it)));

    Array<double, CoordinateSystem<4> > t_ab_vvoo(world, v_ab_vvoo.trange(), v_ab_vvoo.get_shape());
    for(Array<double, CoordinateSystem<4> >::range_type::const_iterator it = t_ab_vvoo.range().begin(); it != t_ab_vvoo.range().end(); ++it)
      if(t_ab_vvoo.is_local(*it) && (! t_ab_vvoo.is_zero(*it)))
        t_ab_vvoo.set(*it, 0.0);

    Array<double, CoordinateSystem<4> > D_ab_vvoo(world, v_ab_vvoo.trange(), v_ab_vvoo.get_shape());
    for(Array<double, CoordinateSystem<4> >::range_type::const_iterator it = D_ab_vvoo.range().begin(); it != D_ab_vvoo.range().end(); ++it)
      if(D_ab_vvoo.is_local(*it) && (! D_ab_vvoo.is_zero(*it)))
        D_ab_vvoo.set(*it, world.taskq.add(data, & InputData::make_D_tile, D_ab_vvoo.trange().make_tile_range(*it)));

    Array<double, CoordinateSystem<4> > t_bb_vvoo(world, v_bb_vvoo.trange(), v_bb_vvoo.get_shape());
    for(Array<double, CoordinateSystem<4> >::range_type::const_iterator it = t_bb_vvoo.range().begin(); it != t_bb_vvoo.range().end(); ++it)
      if(t_bb_vvoo.is_local(*it) && (! t_bb_vvoo.is_zero(*it)))
        t_bb_vvoo.set(*it, 0.0);

    Array<double, CoordinateSystem<4> > D_bb_vvoo(world, v_bb_vvoo.trange(), v_bb_vvoo.get_shape());
    for(Array<double, CoordinateSystem<4> >::range_type::const_iterator it = D_bb_vvoo.range().begin(); it != D_bb_vvoo.range().end(); ++it)
      if(D_bb_vvoo.is_local(*it) && (! D_bb_vvoo.is_zero(*it)))
        D_bb_vvoo.set(*it, world.taskq.add(data, & InputData::make_D_tile, D_bb_vvoo.trange().make_tile_range(*it)));


    world.gop.fence();

    data.clear();

    Array<double, CoordinateSystem<4> >r_aa_vvoo =
         v_aa_vvoo("p1a,p2a,h1a,h2a")
        -f_a_vv("p1a,p3a")*t_aa_vvoo("p2a,p3a,h1a,h2a")
        +f_a_vv("p2a,p3a")*t_aa_vvoo("p1a,p3a,h1a,h2a")
        +f_a_oo("h3a,h1a")*t_aa_vvoo("p1a,p2a,h2a,h3a")
        -f_a_oo("h3a,h2a")*t_aa_vvoo("p1a,p2a,h1a,h3a")
        ;

    Array<double, CoordinateSystem<4> >r_ab_vvoo =
         v_ab_vvoo("p1a,p2b,h1a,h2b")
        +f_a_vv("p1a,p3a")*t_ab_vvoo("p3a,p2b,h1a,h2b")
        +f_b_vv("p2b,p3b")*t_ab_vvoo("p1a,p3b,h1a,h2b")
        -f_a_oo("h3a,h1a")*t_ab_vvoo("p1a,p2b,h3a,h2b")
        -f_b_oo("h3b,h2b")*t_ab_vvoo("p1a,p2b,h1a,h3b");

    Array<double, CoordinateSystem<4> >r_bb_vvoo =
         v_bb_vvoo("p1b,p2b,h1b,h2b")
        -f_b_vv("p1b,p3b")*t_bb_vvoo("p2b,p3b,h1b,h2b")
        +f_b_vv("p2b,p3b")*t_bb_vvoo("p1b,p3b,h1b,h2b")
        +f_b_oo("h3b,h1b")*t_bb_vvoo("p1b,p2b,h2b,h3b")
        -f_b_oo("h3b,h2b")*t_bb_vvoo("p1b,p2b,h1b,h3b");

    Array<double, CoordinateSystem<4> > rr_aa_vvoo =
//        t_ab_vvoo("a,b,i,j") =
        TiledArray::expressions::make_binary_tiled_tensor(D_aa_vvoo("a,b,i,j"),
        r_aa_vvoo("a,b,i,j"), std::multiplies<double>())
        + t_aa_vvoo("a,b,i,j");

    Array<double, CoordinateSystem<4> > rr_ab_vvoo =
//    t_ab_vvoo("a,b,i,j") =
        TiledArray::expressions::make_binary_tiled_tensor(D_ab_vvoo("a,b,i,j"),
        r_ab_vvoo("a,b,i,j"), std::multiplies<double>())
        + t_ab_vvoo("a,b,i,j");

    Array<double, CoordinateSystem<4> > rr_bb_vvoo =
//    t_bb_vvoo("a,b,i,j") =
        TiledArray::expressions::make_binary_tiled_tensor(D_bb_vvoo("a,b,i,j"),
        r_bb_vvoo("a,b,i,j"), std::multiplies<double>())
        + t_bb_vvoo("a,b,i,j");

    double energy =
         0.25 * (  TiledArray::expressions::dot(rr_aa_vvoo("a,b,i,j"), v_aa_vvoo("a,b,i,j"))
                 + TiledArray::expressions::dot(rr_bb_vvoo("a,b,i,j"), v_bb_vvoo("a,b,i,j"))
                )
        + TiledArray::expressions::dot(rr_ab_vvoo("a,b,i,j"), v_ab_vvoo("a,b,i,j"))
    ;

    std::cout << "MP2 energy = " << std::setprecision(12) << energy << "\n";
    std::cout << "Done!\n";

  } else  {
    std::cout << "Unable to open file: " << file_name << "\n";
    // stop the madenss runtime
    madness::finalize();
    return 1;
  }

  // stop the madenss runtime
  madness::finalize();
	return 0;
}
