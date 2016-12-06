/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2016  Virginia Tech
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
 *  Drew Lewis
 *  Department of Chemistry, Virginia Tech
 *
 *  evd.cpp
 *  Dec 6th, 2016
 *  
 */

#include <tiledarray.h>
#include <iostream>

int main(int argc, char** argv) {
    int rc = 0;

    try {
        // Initialize runtime
        TiledArray::World& world = TiledArray::initialize(argc, argv);

        // Get command line arguments
        if (argc < 3) {
            std::cout << "Usage: evd matrix_size block_size [repetitions] "
                         "[use_complex]\n";
            return 0;
        }
        const long matrix_size = atol(argv[1]);
        const long block_size = atol(argv[2]);
        if (matrix_size <= 0) {
            std::cerr << "Error: matrix size must greater than zero.\n";
            return 1;
        }
        if (block_size <= 0) {
            std::cerr << "Error: block size must greater than zero.\n";
            return 1;
        }
        const long repeat = (argc >= 4 ? atol(argv[3]) : 5);
        if (repeat <= 0) {
            std::cerr
                << "Error: number of repetitions must greater than zero.\n";
            return 1;
        }

        const std::size_t num_blocks = matrix_size / block_size;
        const std::size_t block_count = num_blocks * num_blocks;

        if (world.rank() == 0)
            std::cout << "TiledArray: to El evd test..."
                      << "\nNumber of nodes     = " << world.size()
                      << "\nMatrix size         = " << matrix_size << "x"
                      << matrix_size << "\nBlock size          = " << block_size
                      << "x" << block_size << "\nMemory per matrix   = "
                      << double(matrix_size * matrix_size * sizeof(double)) /
                             1.0e9
                      << " GB\nNumber of blocks    = " << block_count
                      << "\nAverage blocks/node = "
                      << double(block_count) / double(world.size()) << "\n";

        // Construct TiledRange
        std::vector<unsigned int> blocking;
        blocking.reserve(num_blocks + 1);
        for (long i = 0l; i < matrix_size; i += block_size)
            blocking.push_back(i);
        blocking.push_back(matrix_size);

        std::vector<TiledArray::TiledRange1> blocking2(
            2, TiledArray::TiledRange1(blocking.begin(), blocking.end()));

        TiledArray::TiledRange trange(blocking2.begin(), blocking2.end());

        auto array_task = [](TA::Tensor<double> &t, TA::Range const&range){
            t = TA::Tensor<double>(range, 0.0);
            auto lo = range.lobound_data();
            auto up = range.upbound_data();
            for(auto m = lo[0]; m < up[0]; ++m){
                for(auto n = lo[1]; n < up[1]; ++n){
                    t(m,n) = m + n;
                }
            }

            return t.norm();
        };

        auto A_ta = TA::make_array<TA::DistArray<TA::Tensor<double>, 
             TA::DensePolicy>>(world, trange, array_task);

        world.gop.fence();
        auto to_el_start = madness::wall_time();
        auto el_mat = TA::array_to_el(A_ta);
        world.gop.fence();
        auto to_el_end = madness::wall_time();
        auto to_el_time = to_el_end - to_el_start;

        auto A_ta_copy = TA::el_to_array(el_mat, world, A_ta.trange());
        world.gop.fence();
        auto el_to_array_time = madness::wall_time() - to_el_end;

        if(world.rank() == 0){
            std::cout << "To el time: " << to_el_time << std::endl;
            std::cout << "From el time: " << el_to_array_time << std::endl;
        }

        double norm = (A_ta("i,j") - A_ta_copy("i,j")).norm(world).get();

        if(world.rank() == 0){
            std::cout << "To el and back norm diff: " << norm << std::endl;
        }

        
        El::DistMatrix<double, El::VR, El::STAR> w(El::Grid::Default());
        El::DistMatrix<double> C(El::Grid::Default());

        std::vector<double> times;
        for(auto i = 0; i < repeat; ++i){
            El::DistMatrix<double> el_mat_copy = el_mat;
            auto start = madness::wall_time();
            El::HermitianEig(El::LOWER, el_mat_copy, w, C);
            auto time = madness::wall_time() - start;
            times.push_back(time);
        }

        auto time_avg = 0.0;
        for(auto const &t : times){
            time_avg += t;
        }
        time_avg /= times.size();
        if(world.rank() == 0){
            std::cout << "Evd time average: " << time_avg << std::endl;
        }

        auto evecs = TA::el_to_array(C, world, A_ta.trange());

        world.gop.fence();

        TiledArray::finalize();

    } catch (TiledArray::Exception& e) {
        std::cerr << "!! TiledArray exception: " << e.what() << "\n";
        rc = 1;
    } catch (madness::MadnessException& e) {
        std::cerr << "!! MADNESS exception: " << e.what() << "\n";
        rc = 1;
    } catch (SafeMPI::Exception& e) {
        std::cerr << "!! SafeMPI exception: " << e.what() << "\n";
        rc = 1;
    } catch (std::exception& e) {
        std::cerr << "!! std exception: " << e.what() << "\n";
        rc = 1;
    } catch (...) {
        std::cerr << "!! exception: unknown exception\n";
        rc = 1;
    }

    return rc;
}
