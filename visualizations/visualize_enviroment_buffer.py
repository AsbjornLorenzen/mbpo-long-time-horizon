import numpy as np
def visualize_enviroment_buffers(path, dimension): 
    
    d_env_path = path + 'replay_buffer.npz'
    d_mod_path = path + 'sac_buffer.npz'


    d_env = np.load(d_env_path)
    d_mod = np.load(d_mod_path)

    print(d_env.keys())
    print(d_env['obs'].shape)
    print(d_mod['obs'].shape)



if __name__ == "__main__": 
    folder_path = 'exp/macura/experiment_name/gym___InvertedDoublePendulum-v4/2025.03.10/163706/'
    dimension = 3
    visualize_enviroment_buffers(folder_path, dimension)