
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





"""
Read all the results in the folder, load them into dataframes, and average them together.
"""
def extract_reward_step_data_frame_from_paths(folder_paths_arr): 
    folder_paths_arr = [path + '/results.csv' for path in folder_paths_arr]
    
    dfs = []
    for path in folder_paths_arr: 
        df = pd.read_csv(path)
        df = df[['env_step', 'episode_reward']]
        dfs.append(df)

    # Perform a sequential merge on 'id'
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='env_step'), dfs)

    results_df = df_merged[['env_step']]    
    # Compute row-wise average, skipping the first column
    results_df['row_average'] = df_merged.iloc[:, 1:].mean(axis=1)
    results_df['row_min'] = df_merged.iloc[:, 1:].min(axis=1)  # Row-wise min
    results_df['row_max'] = df_merged.iloc[:, 1:].max(axis=1)  # Row-wise max
    results_df['row_std'] = df_merged.iloc[:, 1:].std(axis=1)  # Row-wise standard deviation

    return results_df



def plot_step_to_reward( paths):
    df = extract_reward_step_data_frame_from_paths(paths)
    
    plt.style.use("seaborn-darkgrid")  
    plt.figure(figsize=(8, 5))
    plt.plot(df['env_step'], df['row_average'], color='blue', label='Average Performance')
    plt.fill_between(df['env_step'], df['row_average'] - df['row_std'], df['row_average'] + df['row_std'], color='blue', alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.show()
    sns.savefig("visualizations/graphs/reward_to_step.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__": 
    paths = ['exp/macura/experiment_name/gym___InvertedPendulum-v4/2025.03.08/160511', 'exp/macura/experiment_name/gym___InvertedPendulum-v4/2025.03.08/160511']

    plot_step_to_reward(paths)