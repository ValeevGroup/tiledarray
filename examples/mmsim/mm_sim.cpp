#include <iostream>
#include <Eigen/core>
#include <world/world.h>

void stat(const std::size_t P, const std::size_t M, const std::size_t N,
    const std::size_t K, const std::size_t* A, const std::size_t* B, const std::size_t C) {

  std::vector<std::size_t> contract(P, 0);
  std::vector<std::size_t> reduce(P, 0);
  std::vector<std::size_t> comm_in(P, 0);
  std::vector<std::size_t> comm_out(P, 0);
  std::vector<std::vector<std::size_t> > tempC(P, std::vector<std::size_t>(M * N, 0));
  std::vector<std::vector<std::size_t> > casheA(P, std::vector<std::size_t>(M * K, 0));
  std::vector<std::vector<std::size_t> > casheB(P, std::vector<std::size_t>(K * N, 0));


  for(std::size_t m = 0; m < M; ++m) {
    for(std::size_t n = 0; n < N; ++n) {
      for(std::size_t k = 0; k < K; ++k) {
        const std::size_t left_i = m * K + k;
        const std::size_t left = A[left_i];
        const std::size_t right_i = k * N + n;
        const std::size_t right = B[right_i];
        const std::size_t res_i = m * N + n;
        const std::size_t res = C[res_i];


        if(left == right) {
          // Contraction cost
          ++contract[left];
          // Produce temp for C
          ++(tempC[left][res_i]);

          if(left != res) {
            // C is non local;
            if(tempC[left][res_i] == 1) { // Only once since multiple contributions are combine before sending
              // Add communication and reduction cost
              ++comm_out[left];
              ++comm_in[res];
              ++reduce[res];
            }
          }

        } else {
          if(left == res) {
            // Do contraction on the left node since the result goes there too.
            ++contract[left];
            // Produce temp for C
            ++(tempC[left][res_i]);

            // Communication cost to get right
            if(++(cashB[right][right_i]) == 1) {
              ++comm_out[right];
              ++comm_in[left];
            }

            // Communication and reduction cost
            if(tempC[left][res_i] == 1) { // Only once since multiple contributions are combine before sending
              // Add communication and reduction cost
              ++comm_out[left];
              ++comm_in[res];
              ++reduce[res];
            }

          } else if(right == res) {
            // Do contraction on the left node since the result goes there too.
            ++contract[right];

            // Communication cost to get left
            if(++(cashA[left][left_i]) == 1) {
              ++comm_out[left];
              ++comm_in[right];
            }

            // Communication and reduction cost
            if(tempC[right][res_i] == 1) { // Only once since multiple contributions are combine before sending
              // Add communication and reduction cost
              ++comm_out[right];
              ++comm_in[res];
              ++reduce[res];
            }

          } else {
            // Randomly pick left or right since the is no clear advantage
            std::size_t h = madness::hash_value(left);
            madness::hash_combine(h, right);
            if(h % 2) {
              // Do contraction on left node
              ++contract[left];

              // Communication cost to get right
              if(++(cashB[right][right_i]) == 1) {
                ++comm_out[right];
                ++comm_in[left];
              }

              // Communication and reduction cost
              if(tempC[left][res_i] == 1) { // Only once since multiple contributions are combine before sending
                // Add communication and reduction cost
                ++comm_out[left];
                ++comm_in[res];
                ++reduce[res];
              }

            } else {
              // Do contraction on right node
              ++contract[right];

              // Communication cost to get left
              if(++(cashA[left][left_i]) == 1) {
                ++comm_out[left];
                ++comm_in[right];
              }

              // Communication and reduction cost
              if(tempC[right][res_i] == 1) { // Only once since multiple contributions are combine before sending
                // Add communication and reduction cost
                ++comm_out[right];
                ++comm_in[res];
                ++reduce[res];
              }
            }
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
    madness::initialize(argc,argv);
    madness::World world(MPI::COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 8;

    madness::finalize();
    return 0;
}
