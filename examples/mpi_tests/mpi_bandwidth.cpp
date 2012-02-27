/****************************************************************************
 * FILE: mpi_bandwidth.cpp
 * DESCRIPTION:
 *   Provides point-to-point communications timings for any even
 *   number of MPI tasks.
 * AUTHOR: Blaise Barney
 * LAST REVISED: 04/13/05
 * Ported to C++ by Justus Calvin on 02/27/2012
 ****************************************************************************/
#include <mpi.h>
#include <iostream>

#define MAXTASKS      8192
// Change the next four parameters to suit your case
#define STARTSIZE     100000
#define ENDSIZE       1000000
#define INCREMENT     100000
#define ROUNDTRIPS    100

int main (int argc, char** argv) {
  // Some initializations and error checking
  MPI::Init(argc, argv);
  const int numtasks = MPI::COMM_WORLD.Get_size();
  if (numtasks % 2 != 0) {
    std::cout << "ERROR: Must be an even number of tasks!  Quitting...\n";
    MPI::COMM_WORLD.Abort(0);
    exit(0);
  }
  const int rank = MPI::COMM_WORLD.Get_rank();

  char msgbuf[ENDSIZE];
  std::fill_n(msgbuf, ENDSIZE, 'x');

  // All tasks send their host name to task 0
  int namelength = 0;
  char host[MPI::MAX_PROCESSOR_NAME];
  char hostmap[MAXTASKS][MPI::MAX_PROCESSOR_NAME];
  MPI::Get_processor_name(host, namelength);
  MPI::COMM_WORLD.Gather(&host, MPI::MAX_PROCESSOR_NAME, MPI::CHAR, &hostmap,
      MPI::MAX_PROCESSOR_NAME, MPI::CHAR, 0);

  // Determine who my send/receive partner is and tell task 0
  int src = (rank < numtasks/2 ? numtasks/2 + rank : rank - numtasks/2);
  int dest = src;
  int taskpairs[MAXTASKS];
  MPI::COMM_WORLD.Gather(&dest, 1, MPI::INT, &taskpairs, 1, MPI::INT, 0);

  if (rank == 0) {
    const double resolution = MPI::Wtick();
    std::cout << "\n******************** MPI Bandwidth Test ********************\n"
              << "Message start size= " << STARTSIZE << " bytes\n"
              << "Message finish size= " << ENDSIZE << " bytes\n"
              << "Incremented by " << INCREMENT << " bytes per iteration\n"
              << "Roundtrips per iteration= " << ROUNDTRIPS << "\n"
              << "MPI::Wtick resolution = " << resolution << "\n"
              << "************************************************************\n";
    for(int i=0; i<numtasks; i++)
      std::cout << "task " << i << " is on " << hostmap[i] << " partner=" << taskpairs[i] << "\n";
    std::cout << "************************************************************\n";
  }


  //************************** first half of tasks ****************************
  // These tasks send/receive messages with their partner task, and then do a
  // few bandwidth calculations based upon message size and timings.

  if (rank < numtasks/2) {
    for(int n = STARTSIZE; n <= ENDSIZE; n += INCREMENT) {
      double bestbw = 0.0;
      double worstbw = .99E+99;
      double totalbw = 0.0;
      const int nbytes =  sizeof(char) * n;
      for(int i = 1; i <= ROUNDTRIPS; ++i){
        const double t1 = MPI::Wtime();
        MPI::COMM_WORLD.Send(&msgbuf, n, MPI::CHAR, dest, 1);
        MPI::COMM_WORLD.Recv(&msgbuf, n, MPI::CHAR, src, 1);
        const double t2 = MPI::Wtime();
        const double thistime = t2 - t1;
        const double bw = ((double)nbytes * 2) / thistime;
        totalbw += bw;
        bestbw = std::max(bestbw, bw);
        worstbw = std::min(worstbw, bw);
      }
      // Convert to megabytes per second
      bestbw /= 1000000.0;
      const double avgbw = (totalbw/1000000.0)/double(ROUNDTRIPS);
      worstbw /= 1000000.0;

      // Task 0 collects timings from all relevant tasks
      if (rank == 0) {
        double timings[MAXTASKS/2][3];
        // Keep track of my own timings first
        timings[0][0] = bestbw;
        timings[0][1] = avgbw;
        timings[0][2] = worstbw;
        // Initialize overall averages
        double bestall = 0.0;
        double avgall = 0.0;
        double worstall = 0.0;
        // Now receive timings from other tasks and print results. Note that
        // this loop will be appropriately skipped if there are only two tasks.
        for(int j = 1; j < (numtasks/2); ++j)
          MPI::COMM_WORLD.Recv(&timings[j], 3, MPI::DOUBLE, j, 1);
        std::cout << "***Message size: " << n << " *** best  /  avg  / worst (MB/sec)\n";
        for(int j=0; j<numtasks/2; j++) {
          std::cout << "   task pair: " << j << " - " << taskpairs[j] << ":    "
              << timings[j][0] << " / " << timings[j][1] << " / " << timings[j][2] << "\n";
          bestall += timings[j][0];
          avgall += timings[j][1];
          worstall += timings[j][2];
        }
        std::cout << "   OVERALL AVERAGES:          " << bestall/(numtasks/2)
            << " / " << avgall/(numtasks/2) << " / " << worstall/(numtasks/2) << " \n\n";
      }
      else {
        double tmptimes[3];
        // Other tasks send their timings to task 0
        tmptimes[0] = bestbw;
        tmptimes[1] = avgbw;
        tmptimes[2] = worstbw;
        MPI::COMM_WORLD.Send(tmptimes, 3, MPI::DOUBLE, 0, 1);
      }
    }
  }



  //*************************** second half of tasks ***************************
  // These tasks do nothing more than send and receive with their partner task

  if(rank >= numtasks/2) {
    for(int n = STARTSIZE; n <= ENDSIZE; n += INCREMENT) {
      for(int i = 1; i <= ROUNDTRIPS; ++i){
        MPI::COMM_WORLD.Recv(&msgbuf, n, MPI::CHAR, src, 1);
        MPI::COMM_WORLD.Send(&msgbuf, n, MPI::CHAR, dest, 1);
      }
    }
  }

  MPI::Finalize();
  return 0;
}
