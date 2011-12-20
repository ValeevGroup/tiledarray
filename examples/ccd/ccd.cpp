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
  {
    madness::World world(MPI::COMM_WORLD);

    // Get input data
    std::ifstream input("input");
    InputData data(input);
    input.close();

    Array<double, CoordinateSystem<2> > f_a_oo = data.make_f(world, alpha, occ, occ);
    Array<double, CoordinateSystem<2> > f_a_vv = data.make_f(world, alpha, vir, vir);
    Array<double, CoordinateSystem<2> > f_b_oo = data.make_f(world, beta, occ, occ);
    Array<double, CoordinateSystem<2> > f_b_vv = data.make_f(world, beta, vir, vir);


    Array<double, CoordinateSystem<4> > v_ab_oooo = data.make_v_ab(world, occ, occ, occ, occ);
    Array<double, CoordinateSystem<4> > v_ab_oovv = data.make_v_ab(world, occ, occ, vir, vir);
    Array<double, CoordinateSystem<4> > v_ab_vovo = data.make_v_ab(world, vir, occ, vir, occ);
    Array<double, CoordinateSystem<4> > v_ab_vvvv = data.make_v_ab(world, vir, vir, vir, vir);

    // Fence to make sure data on all nodes has been initialized
    world.gop.fence();

    Array<double, CoordinateSystem<4> > v_aa_oooo = v_ab_oooo("a,b,c,d") - v_ab_oooo("a,b,d,c");
    Array<double, CoordinateSystem<4> > v_aa_oovv = v_ab_oovv("a,b,c,d") - v_ab_oovv("a,b,d,c");
    Array<double, CoordinateSystem<4> > v_aa_vovo = v_ab_vovo("a,b,c,d") - v_ab_vovo("a,b,d,c");
    Array<double, CoordinateSystem<4> > v_aa_vvvv = v_ab_vvvv("a,b,c,d") - v_ab_vvvv("a,b,d,c");
    Array<double, CoordinateSystem<4> >& v_bb_oooo = v_aa_oooo;
    Array<double, CoordinateSystem<4> >& v_bb_oovv = v_aa_oovv;
    Array<double, CoordinateSystem<4> >& v_bb_vovo = v_aa_vovo;
    Array<double, CoordinateSystem<4> >& v_bb_vvvv = v_aa_vvvv;

    world.gop.fence();

  }

  std::cout << "Done!\n";
  // stop the madenss runtime
  madness::finalize();
	return 0;
}
