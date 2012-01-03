//============================================================================
// Name        : CCD.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   :
// Description : Hello World in C, Ansi-style
//============================================================================

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
    Array<double, CoordinateSystem<4> > v_ab_oovv = data.make_v_ab(world, occ, occ, vir, vir);
    Array<double, CoordinateSystem<4> > v_ab_vovo = data.make_v_ab(world, vir, occ, vir, occ);
    Array<double, CoordinateSystem<4> > v_ab_voov = data.make_v_ab(world, vir, occ, occ, vir);
    Array<double, CoordinateSystem<4> > v_ab_vvvv = data.make_v_ab(world, vir, vir, vir, vir);

    // Fence to make sure data on all nodes has been initialized
    world.gop.fence();
    std::cout << " done.\nConstructing v_aa and v_bb tensors...";

    Array<double, CoordinateSystem<4> > v_aa_oooo = v_ab_oooo("i,j,k,l") - v_ab_oooo("i,j,l,k");
    Array<double, CoordinateSystem<4> > v_aa_oovv = v_ab_oovv("i,j,a,b") - v_ab_oovv("i,j,b,a");
    Array<double, CoordinateSystem<4> > v_aa_vovo = v_ab_vovo("a,i,b,j") - v_ab_voov("a,i,j,b");
    Array<double, CoordinateSystem<4> > v_aa_vvvv = v_ab_vvvv("a,b,c,d") - v_ab_vvvv("a,b,d,c");
    // Just make references to the data since the input is closed shell.
    Array<double, CoordinateSystem<4> >& v_bb_oooo = v_aa_oooo;
    Array<double, CoordinateSystem<4> >& v_bb_oovv = v_aa_oovv;
    Array<double, CoordinateSystem<4> >& v_bb_vovo = v_aa_vovo;
    Array<double, CoordinateSystem<4> >& v_bb_vvvv = v_aa_vvvv;

    // Fence again to make sure data all the integral tensors have been initialized
    world.gop.fence();
    std::cout << " done.\n";

    input.clear();

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
