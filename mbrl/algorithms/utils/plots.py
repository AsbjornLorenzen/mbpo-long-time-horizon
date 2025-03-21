
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import pickle

def plot_critic_loss(work_dir, graph_name='critic_loss'):
    path = work_dir + '/train.csv'
    df = pd.read_csv(path)
    df = df[['step', 'critic_loss']]
    plt.figure(figsize=(8, 5))
    plt.plot(df['step'], df['critic_loss'], color='blue')
    plt.xlabel("Steps")
    plt.ylabel("Average Rollout Length")
    plt.savefig(f"{work_dir}/{graph_name}", dpi=300, bbox_inches="tight")






def plot_rollout_length(work_dir, graph_name='rollout_length'): 
    path = work_dir + '/results.csv'
    df = pd.read_csv(path)
    df = df[['env_step', 'rollout_length']]
    plt.figure(figsize=(8, 5))
    plt.plot(df['env_step'], df['rollout_length'], color='blue')
    plt.xlabel("Steps")
    plt.ylabel("Average Rollout Length")
    plt.savefig(f"{work_dir}/{graph_name}", dpi=300, bbox_inches="tight")


def plot_results(work_dir, graph_name='reward_to_step'): 
    path = work_dir + '/results.csv'
    df = pd.read_csv(path)
    df = df[['env_step', 'episode_reward']]
    plt.figure(figsize=(8, 5))
    plt.plot(df['env_step'], df['episode_reward'], color='blue')
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.savefig(f"{work_dir}/{graph_name}", dpi=300, bbox_inches="tight")




def save_d_mod(work_dir, state_arr, file_name='d_mod.pickle'): 
    with open(f"{work_dir}/{file_name}", "wb") as f:
        pickle.dump(state_arr, f)




def plot_density_plot(data, file_name):
    plt.figure(figsize=(10, 4))

    for t, vels in zip([x for x in range(101)], data):
        plt.scatter([t* 10] * len(vels), vels, color="blue", alpha=0.1, s=5)
    

    plt.xticks([x for x in range(0, 1001, 100)])
    plt.xlabel("Time")
    plt.ylabel("observation 6")
    plt.title("Environment Buffer")
    # Grid styling
    plt.grid(True, linestyle="--", alpha=0.6)

  
    plt.savefig(file_name)



def plot_d_env( data, file_name, dimension): 
    results = [[] for x in range(2000)]
    for (start, stop) in data['trajectory_indices']:
        span = np.arange(start, stop)
        each_time_step_data = data['next_obs'][span]

        for i, val in enumerate(each_time_step_data):
            results[i].append(val[6]) # observation 0

    
    
    plot_density_plot(results, file_name)





# TODO: also allow the dimension to be varied.
def plot_enviroment_buffer(work_dir, dimension=6):
    d_env_path = work_dir + '/replay_buffer.npz'
    d_mod_path = work_dir + '/d_mod.pickle'

    d_env = np.load(d_env_path)
    with open(d_mod_path, "rb") as f:
        d_mod = pickle.load(f)


    
    new_d_mod = [[] for x in range(2000)]

    for i, arr in enumerate(d_mod):
        for x in arr:
            new_d_mod[i].append(x[dimension])

    plot_density_plot(new_d_mod, f'{work_dir}/d_mod.png')
    plot_d_env(d_env, f'{work_dir}/d_env.png', dimension=dimension)


def create_graphs(work_dir): 
    plot_results(work_dir)
    plot_rollout_length(work_dir)
    # plot_critic_loss(work_dir)
    plot_enviroment_buffer( work_dir)



if __name__ == "__main__":
    create_graphs('exp/macura/experiment_name/gym___InvertedDoublePendulum-v4/2025.03.18/193502')
