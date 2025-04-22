import subprocess
import os
import time

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class TrajectoryGenerator:

    def __init__(self, trajectory_path, time_step, num_points, num_split_points=10,
                 stddev_init=1e3, stddev_final=1e3, hydro_fraction=0.01, polynomial_order=3):
        self.trajectory_path = trajectory_path
        self.time_step = time_step
        self.num_points = num_points
        self.num_split_points = num_split_points
        self.stddev_init = stddev_init
        self.stddev_final = stddev_final
        self.hydro_fraction = hydro_fraction
        self.polynomial_order = polynomial_order

    @staticmethod
    def generate_trajectory(num_points, num_spline_points, polynomial_order,
                            stddev_init, stddev_final):
        t_N = np.linspace(0, 1, num_spline_points)
        stddevs = np.linspace(stddev_init, stddev_final, num_spline_points, endpoint=True)
        traj_N = [np.random.normal(loc=0, scale=sigma) for sigma in stddevs]
        fun = interpolate.interp1d(t_N, traj_N, kind=polynomial_order)

        t_n = np.linspace(0, 1, num_points)
        traj_n = fun(t_n)
        return t_n, traj_n

    def generate_trajectories(self):
        args = [self.num_points, self.num_split_points, self.polynomial_order,
                self.stddev_init, self.stddev_final]
        args2 = args.copy()
        args2[-1] = self.hydro_fraction * self.stddev_final
        args2[-2] = self.hydro_fraction * self.stddev_init

        _, vxx = self.generate_trajectory(*args)
        _, vyy = self.generate_trajectory(*args)
        vzz = -(vxx + vyy)
        _, vxy = self.generate_trajectory(*args)
        _, vxz = self.generate_trajectory(*args)
        _, vyz = self.generate_trajectory(*args)
        _, hydro = self.generate_trajectory(*args2)

        vxx += hydro
        vyy += hydro
        vzz += hydro
        return vxx, vyy, vzz, vxy, vxz, vyz

    @staticmethod
    def plot_trajectories(vxx, vyy, vzz, vxy, vxz, vyz):
        plt.figure(figsize=(6, 4))
        plt.plot(vxx, label='vxx')
        plt.plot(vyy, label='vyy')
        plt.plot(vzz, label='vzz')
        plt.plot(vxy, label='vxy')
        plt.plot(vxz, label='vxz')
        plt.plot(vyz, label='vyz')
        plt.xlabel('Time step')
        plt.ylabel('Velocity gradient')
        plt.legend()
        plt.savefig('trajectories.png', bbox_inches='tight')

    def create_trajectory_file(self, filename, plot=False, n_iterations=None):
        vxx, vyy, vzz, vxy, vxz, vyz = self.generate_trajectories()

        if n_iterations is not None:
            n_iterations = min(n_iterations, len(vxx))
            vxx = vxx[:n_iterations]
            vyy = vyy[:n_iterations]
            vzz = vzz[:n_iterations]
            vxy = vxy[:n_iterations]
            vxz = vxz[:n_iterations]
            vyz = vyz[:n_iterations]

        with open(filename, 'w') as f:
            f.write('vxx,vyy,vzz,vxy,vxz,vyz\n')
            for i in range(len(vxx)):
                f.write(f"{vxx[i]},{vyy[i]},{vzz[i]},{vxy[i]},{vxz[i]},{vyz[i]}\n")

        if plot:
            self.plot_trajectories(vxx, vyy, vzz, vxy, vxz, vyz)

        return [vxx, vyy, vzz, vxy, vxz, vyz]


def get_trajectories_from_code_output(cpp_output):
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
    return strain_trajectories, stress_trajectories, velgrad_trajectories


def run_cpp_code(trajectory_path, time_step, res=16):
    evpfft_path = '/resnick/groups/bhatta/ccocke/CarterFierro/Fierro/src/EVPFFT/evpfft_fftw_openmp/evpfft_carter'
    input_file_path = f'/resnick/groups/bhatta/ccocke/CarterFierro/Fierro/src/EVPFFT/example_input_files/tantalum_input_{res}.txt'
    if not os.path.exists(evpfft_path):
        raise FileNotFoundError(f"Path {evpfft_path} does not exist.")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Path {input_file_path} does not exist.")
    call_list = [evpfft_path,
                 input_file_path,
                 trajectory_path,
                 str(time_step)]

    print(f"Running command: {call_list}")
    time1 = time.time()
    result = subprocess.run(call_list, capture_output=True, text=True)
    time2 = time.time()
    runtime = time2 - time1
    print(f"Time taken: {runtime:.2f} seconds")
    if result.returncode != 0:
        print(result.stderr)
    output = result.stdout.strip()
    return output, runtime


def main():
    # anvil is 175 m/s
    n_points = 100
    n_iterations = 40
    time_step = 1e-6
    reses = [16, 32, 64, 128]
    runtimes = []
    trajectory_list = []

    trajectory_path = '/resnick/groups/bhatta/ccocke/CarterFierro/Fierro/src/EVPFFT/example_input_files/trajectories.txt'
    trajectory_generator = TrajectoryGenerator(trajectory_path, time_step, n_points,
                                               num_split_points=10, stddev_init=1e2, stddev_final=1e3,
                                               hydro_fraction=0.01, polynomial_order=2)
    velgrad_trajectories_ref = trajectory_generator.create_trajectory_file(trajectory_path, n_iterations=n_iterations, plot=True)

    for res in reses:
        print(f"Running resolution {res}")
        cpp_output, runtime = run_cpp_code(trajectory_path, time_step, res=res)
        runtimes.append(runtime)
        strain_trajectories, stress_trajectories, velgrad_trajectories = get_trajectories_from_code_output(cpp_output)
        for ref, curr in zip(velgrad_trajectories_ref, velgrad_trajectories):
            assert np.allclose(ref, curr), f"Velgrad trajectories do not match for resolution {res}"
        trajectory_list.append([strain_trajectories, stress_trajectories, velgrad_trajectories])

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

        plt.savefig(f'output_{res}.png', bbox_inches='tight')
        plt.close(fig)

    for idx in range(len(reses) - 1):
        plt.figure()
        times = np.arange(0, len(strain_trajectories[0])) * time_step * 1e3  # Convert to ms
        for i, strain in enumerate(strain_trajectories):
            plt.plot(times, trajectory_list[idx][1][i] - trajectory_list[-1][1][i], label=f'Stress {i}')
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Stress difference')
        plt.savefig(f'delta_{reses[idx]}-{reses[-1]}.png', bbox_inches='tight')
        plt.close(fig)

    print("Times:", runtimes)
    print("Resolutions:", reses)
    plt.figure()
    plt.plot([x**3 for x in reses], runtimes, marker='o')
    plt.xlabel('Number of grid points')
    plt.ylabel('Time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{n_iterations} time steps test')
    plt.savefig('times.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot([x**3 for x in reses], [x / n_iterations for x in runtimes], marker='o')
    plt.xlabel('Number of grid points')
    plt.ylabel('Time per time step (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{n_iterations} time steps test')
    plt.savefig('times_per_step.png', bbox_inches='tight')
    plt.close()
    print("All done!")


if __name__ == "__main__":
    main()
