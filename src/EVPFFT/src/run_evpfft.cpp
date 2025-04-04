#include <cstdlib>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <random>
#include <sstream>
#include <vector>

#include "evpfft.h"
#include "matar.h"

// std::vector<double> str_comma_to_vector(const std::string &str) {
//   std::vector<double> result;
//   std::stringstream ss(str);
//   std::string item;
//   while (std::getline(ss, item, ',')) {
//     result.push_back(std::stod(item));
//   }
//   return result;
// }

std::vector<double> generate_velgrad_trajectory(const size_t num_points,
                                                const double mean = 0,
                                                const double std = 1e-5,
                                                const size_t window = 100) {
  std::vector<double> velgrad_trajectory(num_points, 0.0);

  std::vector<double> random_values(num_points, 0.0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist{mean, std};
  for (size_t idx = 0; idx < num_points; idx++) {
    random_values[idx] = dist(gen);
  }

  for (size_t idx = 1; idx < num_points; idx++) {
    for (int p_idx = std::max(0, (int)idx - (int)window); p_idx < (int)idx + 1;
         p_idx++) {
      velgrad_trajectory[idx] += random_values[(size_t)p_idx];
    }
  }
  return velgrad_trajectory;
}

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  Kokkos::initialize(Kokkos::InitializationSettings()
                         .set_disable_warnings(true)
                         .set_num_threads(1));

  MPI_Comm evpfft_mpi_comm = MPI_COMM_NULL;
  if (evpfft_mpi_comm == MPI_COMM_NULL) {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_split(MPI_COMM_WORLD, global_rank, global_rank, &evpfft_mpi_comm);
  }
  int my_rank, num_ranks;
  MPI_Comm_rank(evpfft_mpi_comm, &my_rank);
  MPI_Comm_size(evpfft_mpi_comm, &num_ranks);

  // Ensure we have the correct number of arguments
  if (argc != 6) {
    std::cout << "Length of argc is " << argc << std::endl;
    std::cerr << "Usage: run_evpfft num_points time_step mean std window"
              << std::endl;
    return 1;
  }
  const size_t num_points = std::stoi(argv[1]);
  const double time_step = std::stod(argv[2]);
  const double mean = std::stod(argv[3]);
  const double std = std::stod(argv[4]);
  const size_t window = std::stoi(argv[5]);

  std::array<std::vector<double>, 6> strain_trajectory, stress_trajectory,
      velgrad_trajectory;
  for (int i = 0; i < 6; ++i) {
    strain_trajectory[i].resize(num_points + 1, 0.0);
    stress_trajectory[i].resize(num_points + 1, 0.0);
    velgrad_trajectory[i] =
        generate_velgrad_trajectory(num_points, mean, std, window);
  }

  std::string filename = "/resnick/groups/bhatta/ccocke/CarterFierro/Fierro/"
                         "src/EVPFFT/example_input_files/tantalum_input.txt";
  CommandLineArgs cmd;
  cmd.input_filename = filename; //"evpfft.in";
  cmd.micro_filetype = 0;
  cmd.check_cmd_args();

  auto evpfft = new EVPFFT(evpfft_mpi_comm, cmd);

  for (size_t cycle = 0; cycle < num_points; cycle++) {
    // Note EVPFFT uses F-layout while Fierro uses C-layout
    FArray<double> Fvel_grad(3, 3);
    FArray<double> Fstress(3, 3);
    FArray<double> Fstrain(3, 3);
    // Transpose vel_grad
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Fstress(i, j) = 0.0;
        Fstrain(i, j) = 0.0;
      }
    }
    Fvel_grad(0, 0) = velgrad_trajectory[0][cycle];
    Fvel_grad(1, 1) = velgrad_trajectory[1][cycle];
    Fvel_grad(2, 2) = velgrad_trajectory[2][cycle];

    Fvel_grad(0, 1) = velgrad_trajectory[3][cycle];
    Fvel_grad(1, 0) = velgrad_trajectory[3][cycle];

    Fvel_grad(0, 2) = velgrad_trajectory[4][cycle];
    Fvel_grad(2, 0) = velgrad_trajectory[4][cycle];

    Fvel_grad(1, 2) = velgrad_trajectory[5][cycle];
    Fvel_grad(2, 1) = velgrad_trajectory[5][cycle];

    const double udotAccTh = 0; // Do not extrapolate

    // std::cout << "Calling solve function with velgrad: "
    //           << velgrad_trajectory[0][cycle] << " "
    //           << velgrad_trajectory[1][cycle] << " "
    //           << velgrad_trajectory[2][cycle] << " "
    //           << velgrad_trajectory[3][cycle] << " "
    //           << velgrad_trajectory[4][cycle] << " "
    //           << velgrad_trajectory[5][cycle] << std::endl;
    evpfft->solve(Fvel_grad.pointer(), Fstress.pointer(), Fstrain.pointer(),
                  time_step, cycle, 0, udotAccTh);

    stress_trajectory[0][cycle + 1] = Fstress(0, 0);
    stress_trajectory[1][cycle + 1] = Fstress(1, 1);
    stress_trajectory[2][cycle + 1] = Fstress(2, 2);
    stress_trajectory[3][cycle + 1] = Fstress(0, 1);
    stress_trajectory[4][cycle + 1] = Fstress(0, 2);
    stress_trajectory[5][cycle + 1] = Fstress(1, 2);

    strain_trajectory[0][cycle + 1] = Fstrain(0, 0);
    strain_trajectory[1][cycle + 1] = Fstrain(1, 1);
    strain_trajectory[2][cycle + 1] = Fstrain(2, 2);
    strain_trajectory[3][cycle + 1] = Fstrain(0, 1);
    strain_trajectory[4][cycle + 1] = Fstrain(0, 2);
    strain_trajectory[5][cycle + 1] = Fstrain(1, 2);
  }

  for (size_t i = 0; i < velgrad_trajectory.size(); i++) {
    std::cout << "Velgrad trajectory " << i << ": ";
    for (size_t j = 0; j < velgrad_trajectory[i].size(); j++) {
      std::cout << velgrad_trajectory[i][j] << ",";
    }
    std::cout << std::endl;
  }

  for (size_t i = 0; i < strain_trajectory.size(); i++) {
    std::cout << "Strain trajectory " << i << ": ";
    for (size_t j = 0; j < strain_trajectory[i].size(); j++) {
      std::cout << strain_trajectory[i][j] << ",";
    }
    std::cout << std::endl;
  }

  for (size_t i = 0; i < stress_trajectory.size(); i++) {
    std::cout << "Stress trajectory " << i << ": ";
    for (size_t j = 0; j < stress_trajectory[i].size(); j++) {
      std::cout << stress_trajectory[i][j] << ",";
    }
    std::cout << std::endl;
  }

  return 0;
}