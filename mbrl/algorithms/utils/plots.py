
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


def plot_enviroment_buffer(work_dir, graph_name='env_buffer'):
    pass




def create_graphs(work_dir): 
    plot_results(work_dir)
    plot_rollout_length(work_dir)
    plot_critic_loss(work_dir)
    