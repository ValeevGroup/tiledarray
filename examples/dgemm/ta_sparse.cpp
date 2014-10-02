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

#include <iostream>
#include <time.h> // for time
#include <tiledarray.h>
#include <iomanip>



int main(int argc, char** argv) {
  int rc = 0;

  try {
    // Initialize runtime
    madness::World& world = madness::initialize(argc, argv);

    // Get command line arguments
    if(argc < 2) {
      std::cout << "Usage: ta_sparse matrix_size block_size [repetitions]\n";
      return 0;
    }
    const long matrix_size = atol(argv[1]);
    const long block_size = atol(argv[2]);
    if(matrix_size <= 0) {
      std::cerr << "Error: matrix size must greater than zero.\n";
      return 1;
    }
    if(block_size <= 0) {
      std::cerr << "Error: block size must greater than zero.\n";
      return 1;
    }
    if((matrix_size % block_size) != 0ul) {
      std::cerr << "Error: matrix size must be evenly divisible by block size.\n";
      return 1;
    }
    const long repeat = (argc >= 4 ? atol(argv[3]) : 4);
    if(repeat <= 0) {
      std::cerr << "Error: number of repetitions must greater than zero.\n";
      return 1;
    }

    // Print information about the test
    const std::size_t num_blocks = matrix_size / block_size;
    std::vector<std::vector<double> > gflops;
    std::vector<std::vector<double> > times;

    std::cout << "TiledArray: block-sparse matrix multiply test...\n"
              << "Number of nodes    = " << world.size()
              << "\nMatrix size        = " << matrix_size << "x" << matrix_size
              << "\nBlock size         = " << block_size << "x" << block_size;

    for(unsigned int left_sparsity = 10; left_sparsity <= 100; left_sparsity += 10){
      std::vector<double> inner_gflops;
      std::vector<double> inner_times;
      for(unsigned int right_sparsity = 10; right_sparsity <= left_sparsity; right_sparsity += 10){
        const long l_block_count = (double(left_sparsity) / 100.0) * double(num_blocks * num_blocks);
        const long r_block_count = (double(right_sparsity) / 100.0) * double(num_blocks * num_blocks);
        if(world.rank() == 0)
                    std::cout << "\nMemory per left matrix  = " << double(l_block_count * block_size * block_size * sizeof(double)) / 1.0e9 << " GB"
                    << "\nMemory per right matrix  = " << double(r_block_count * block_size * block_size * sizeof(double)) / 1.0e9 << " GB"
                    << "\nNumber of left blocks   = " << l_block_count
                    << " " << left_sparsity << " percent"
                    << "\nNumber of right blocks   = " << r_block_count
                    << " " << right_sparsity << " percent"
                    << "\nAverage left blocks/node = " << double(l_block_count) / double(world.size())
                    << "\nAverage right blocks/node = " << double(r_block_count) / double(world.size()) << "\n";

        // Construct TiledRange
        std::vector<unsigned int> blocking;
        blocking.reserve(num_blocks + 1);
        for(long i = 0l; i <= matrix_size; i += block_size)
          blocking.push_back(i);

        std::vector<TiledArray::TiledRange1> blocking2(2,
            TiledArray::TiledRange1(blocking.begin(), blocking.end()));

        TiledArray::TiledRange
          trange(blocking2.begin(), blocking2.end());

        // Construct shape
        TiledArray::Tensor<float>
            a_shape_tensor(trange.tiles(), 0.0),
            b_shape_tensor(trange.tiles(), 0.0);
        if(world.rank() == 0) {
          world.srand(time(NULL));
          const long l_process_block_count = l_block_count / world.size() +
              (world.rank() < (l_block_count / world.size()) ? 1 : 0);
          const long r_process_block_count = r_block_count / world.size() +
              (world.rank() < (r_block_count / world.size()) ? 1 : 0);
          for(long i = 0; i < l_process_block_count; ++i)
            a_shape_tensor.data()[world.rand() % trange.tiles().volume()] = 1.0;

          for(long i = 0; i < r_process_block_count; ++i)
            b_shape_tensor.data()[world.rand() % trange.tiles().volume()] = 1.0;

        }
        TiledArray::SparseShape<float>
            a_shape(world, a_shape_tensor, trange),
            b_shape(world, b_shape_tensor, trange);


        typedef TiledArray::Array<double, 2, TiledArray::Tensor<double>,
            TiledArray::SparsePolicy > SpTArray2;

        // Construct and initialize arrays
        SpTArray2 a(world, trange, a_shape);
        SpTArray2 b(world, trange, b_shape);
        SpTArray2 c(world, trange);
        a.set_all_local(1.0);
        b.set_all_local(1.0);

        // Start clock
        world.gop.fence();
        const double wall_time_start = madness::wall_time();

        // Do matrix multiplication
        for(int i = 0; i < repeat; ++i) {
          c("m,n") = a("m,k") * b("k,n");
          world.gop.fence();
          if(world.rank() == 0)
            std::cout << "Iteration " << i + 1 << "\n";
        }

        // Stop clock
        const double wall_time_stop = madness::wall_time();
        const long flop = 2.0 * c("m,n").sum();
        inner_gflops.push_back(double(repeat) * double(flop) / (wall_time_stop - wall_time_start) / 1.0e9);
        inner_times.push_back((wall_time_stop - wall_time_start)/double(repeat));

        // Print results
        if(world.rank() == 0) {
          std::cout << "Average wall time = " << (wall_time_stop - wall_time_start) / double(repeat)
              << "\nAverage GFLOPS = " << double(repeat) * double(flop) / (wall_time_stop - wall_time_start) / 1.0e9 << "\n";
        }
      }
      gflops.push_back(inner_gflops);
      times.push_back(inner_times);
      world.gop.fence();
    }
    if(world.rank() == 0){
      for(unsigned int i = 0; i < gflops.size(); ++i){
        if(i == 0){
          std::cout << std::defaultfloat;
          std::cout << "   ";
          for(unsigned int j = 10; j <= 100; j+=10){
            std::cout << "        " << j;
          }
          std::cout << std::endl;
        }
        for(unsigned int j = 0; j < gflops[i].size(); ++j){
          if(j == 0){
            std::cout << std::defaultfloat;
            int num = (i+1) * 10;
            if(num < 100){
              std::cout << num << " |";
            } else { std::cout << num << "|"; }
          }

          std::cout << std::setprecision(3) << std::scientific;
          std::cout << double(gflops[i][j]) << " ";
        }
        std::cout << std::endl;
      }
    }

    if(world.rank() == 0){
      for(unsigned int i = 0; i < times.size(); ++i){
        if(i == 0){
          std::cout << std::defaultfloat;
          std::cout << "   ";
          for(unsigned int j = 10; j <= 100; j+=10){
            std::cout << "        " << j;
          }
          std::cout << std::endl;
        }
        for(unsigned int j = 0; j < times[i].size(); ++j){
          if(j == 0){
            std::cout << std::defaultfloat;
            int num = (i+1) * 10;
            if(num < 100){
              std::cout << num << " |";
            } else { std::cout << num << "|"; }
          }

          std::cout << std::setprecision(3) << std::scientific;
          std::cout << double(times[i][j]) << " ";
        }
        std::cout << std::endl;
      }
    }

    madness::finalize();

  } catch(TiledArray::Exception& e) {
    std::cerr << "!!ERROR TiledArray: " << e.what() << "\n";
    rc = 1;
  } catch(madness::MadnessException& e) {
    std::cerr << "!!ERROR MADNESS: " << e.what() << "\n";
    rc = 1;
  } catch(SafeMPI::Exception& e) {
    std::cerr << "!!ERROR SafeMPI: " << e.what() << "\n";
    rc = 1;
  } catch(std::exception& e) {
    std::cerr << "!!ERROR std: " << e.what() << "\n";
    rc = 1;
  } catch(...) {
    std::cerr << "!!ERROR: unknown exception\n";
    rc = 1;
  }


  return rc;
}
