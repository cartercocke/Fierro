# from mpi4py import MPI
import subprocess
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def run_cpp_code(num_points, time_step, mean, std, window):
    path = '../../src/EVPFFT/evpfft_fftw_serial/evpfft_carter'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    call_list = ['../../src/EVPFFT/evpfft_fftw_serial/evpfft_carter', str(num_points), str(time_step), str(mean), str(std), str(window)]
    # assert len(trajectories) == 6, "Expected 6 trajectories"
    # for trajectory in trajectories:
    #     call_list.append(traj_to_str(trajectory))

    result = subprocess.run(call_list, capture_output=True, text=True)
    output = result.stdout.strip()
    return output


def main():
    # anvil is 175 m/s
    n_iterations = 500
    mean = 0
    std = 2e2
    window = 0
    time_step = 1e-6
    cpp_output = run_cpp_code(n_iterations, time_step, mean, std, window)

    def line_to_array(line):
        split = line.split(': ')
        array = split[-1].split(',')
        array = [float(x) for x in array[:-1]]
        return np.array(array)

    lines = cpp_output.split('\n')
    strain_trajectories = []
    stress_trajectories = []
    velgrad_trajectories = []
    for line in lines:
        if 'Strain' in line:
            strain_trajectories.append(line_to_array(line))
        elif 'Stress' in line:
            stress_trajectories.append(line_to_array(line))
        elif 'Velgrad' in line:
            velgrad_trajectories.append(line_to_array(line))

    fig = plt.figure(layout="constrained", figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.00, hspace=0.00)
    ax = fig.add_subplot(gs[0, 0])
    for i, velgrad in enumerate(velgrad_trajectories):
        plt.plot(velgrad, label=f'Velgrad {i}')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Velgrad')

    times = np.arange(0, len(strain_trajectories[0])) * time_step * 1e3  # Convert to ms

    ax = fig.add_subplot(gs[0, 1])
    for i, strain in enumerate(strain_trajectories):
        plt.plot(times, strain, label=f'Strain {i}')
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Strain')

    ax = fig.add_subplot(gs[0, 2])
    for i, stress in enumerate(stress_trajectories):
        plt.plot(times, stress, label=f'Stress {i}')
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Stress')

    plt.savefig('output.png', bbox_inches='tight')


if __name__ == "__main__":
    main()
