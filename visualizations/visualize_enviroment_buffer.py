import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_density_plot(data, file_name):
    plt.figure(figsize=(10, 4))

    for t, vels in zip([x for x in range(len(data))], data):
        plt.scatter([t] * len(vels), vels, color="blue", alpha=0.1, s=5)
    
    plt.xlabel("Time")
    plt.ylabel("Left Elbow Angular Velocity")
    plt.title("Environment Buffer (Info-Prop Dyna)")
    # Grid styling
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show colorbar
    plt.colorbar(label="Density")
    plt.savefig(file_name)




def plot_d_env( data): 
    
    results = [[] for x in range(1000)]


    for (start, stop) in data['trajectory_indices']:
        span = np.arange(start, stop)
        each_time_step_data = data['next_obs'][span]

        for i, val in enumerate(each_time_step_data):
            results[i].append(val[0]) # observation 0

    
    
    plot_density_plot(results, 'd_env.png')



def visualize_enviroment_buffers(path, dimension): 
    
    d_env_path = path + 'replay_buffer.npz'
    d_mod_path = path + 'd_mod.pickle'


    d_env = np.load(d_env_path)
    
    with open(d_mod_path, "rb") as f:
        d_mod = pickle.load(f)

    plot_density_plot(d_mod, 'd_mod.png')
    plot_d_env(d_env)


if __name__ == "__main__": 
    folder_path = 'exp/macura_wildcard/experiment_name/gym___InvertedDoublePendulum-v4/2025.03.13/213051/'
    dimension = 0
    visualize_enviroment_buffers(folder_path, dimension)