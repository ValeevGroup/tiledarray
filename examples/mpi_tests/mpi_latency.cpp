/******************************************************************************
 * FILE: mpi_latency.cpp
 * DESCRIPTION:
 *   MPI Latency Timing Program - C Version
 *   In this example code, a MPI communication timing test is performed.
 *   MPI task 0 will send "reps" number of 1 byte messages to MPI task 1,
 *   waiting for a reply between each rep. Before and after timings are made
 *   for each rep and an average calculated when completed.
 * AUTHOR: Blaise Barney
 * LAST REVISED: 04/13/05
 * Ported to C++ by Justus Calvin on 02/27/2012
 ******************************************************************************/
#include <mpi.h>
#include <iostream>

// number of samples per test
#define NUMBER_REPS 1000

int main (int argc, char** argv) {
  char msg = 'x'; // buffer containing 1 byte message

  MPI::Init(argc,argv);
  if(MPI::COMM_WORLD.Get_rank() == 0 && MPI::COMM_WORLD.Get_size() != 2) {
    std::cout << "Number of tasks = " << MPI::COMM_WORLD.Get_size() << "\n"
              << "Only need 2 tasks - extra will be ignored...\n";
  }
  MPI::COMM_WORLD.Barrier();

  if (MPI::COMM_WORLD.Get_rank() == 0) {
    /* round-trip latency timing test */
    std::cout << "task 0 has started...\n"
              << "Beginning latency timing test. Number of reps = " << NUMBER_REPS << ".\n"
              << "***************************************************\n"
              << "Rep#\tT1\t\tT2\t\tdeltaT\n";

    double sumT = 0.0; // sum of all reps times
    for (int n = 1; n <= NUMBER_REPS; n++) {
      const double T1 = MPI::Wtime();     // start time
      // send message to worker - message tag set to 1.
      // If return code indicates error quit
      MPI::COMM_WORLD.Send(&msg, 1, MPI::BYTE, 1, 1);

      // Now wait to receive the echo reply from the worker
      // If return code indicates error quit
      MPI::COMM_WORLD.Recv(&msg, 1, MPI::BYTE, 1, 1);
      const double T2 = MPI::Wtime();     // end time

      /* calculate round trip time and print */
      const double deltaT = T2 - T1;
      std::cout <<  n << "\t" << T1 << "\t" << T2 << "\t" << deltaT << "\n";
      sumT += deltaT;
    }
    const double avgT = (sumT*1000000.0)/NUMBER_REPS;
    std::cout << "***************************************************\n"
              << "\n*** Avg round trip time = " << avgT << " microseconds\n"
              << "*** Avg one way latency = " << avgT/2 << " microseconds\n";
  } else if (MPI::COMM_WORLD.Get_rank() == 1) {
    std::cout << "task 1 has started...\n";
    for (int n = 1; n <= NUMBER_REPS; n++) {
      MPI::COMM_WORLD.Recv(&msg, 1, MPI_BYTE, 0, 1);
      MPI::COMM_WORLD.Send(&msg, 1, MPI_BYTE, 0, 1);
    }
  }

  MPI::Finalize();
  return 0;
}
