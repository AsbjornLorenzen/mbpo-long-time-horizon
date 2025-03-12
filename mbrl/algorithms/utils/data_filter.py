
"""
In this file define different methods of filtering the data.
Can probably filter data in some interesting way :))
"""


from mbrl.util.replay_buffer import ReplayBuffer, ReplayBufferDynamicLifeTime
import numpy as np


# https://github.com/nmslib/hnswlib

def filter_data(replay_buffer_real_env: ReplayBuffer, sac_buffer: ReplayBufferDynamicLifeTime, threshold=0.1): 
    
    print("FILTERING DATA!")
    count = 0
    for i in range(len(sac_buffer.obs)):
        state = sac_buffer.obs[i]
        smallest_distance = float('inf')
        for j in range(len(replay_buffer_real_env)):
            state_real = replay_buffer_real_env.obs[j]
            distance = np.mean(np.abs(np.subtract(state, state_real)))
            if distance < smallest_distance:
                smallest_distance = distance
        
        if smallest_distance > threshold:
            """sac_buffer.obs.pop(i)
            sac_buffer.next_obs.pop(i)
            sac_buffer.action.pop(i)
            sac_buffer.reward.pop(i)
            sac_buffer.lifetimeA.pop(i)
            sac_buffer.terminated.pop(i)
            sac_buffer.truncated.pop(i)
            """
            count += 1
    print(f"Filtered {count} data points.")
    print("done filtering data.")