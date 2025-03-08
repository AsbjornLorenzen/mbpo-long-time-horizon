"""
Script for automating the running of our experiments.
"""
import os




def top_k_experiment(n_k,n_runs=3, algorithm="macura_top_k", env_config="macura_inverted_pendulum"): 
    
    for k in range(2, n_k):
        for i in range(n_runs): 
            os.system(f"python -m mbrl.examples.main algorithm={algorithm} overrides={env_config} algorith.k={k} seed={i}")
            print(f"Run {i+1} completed.")




def run(n_runs=3, algorithm="macura", env_config="macura_inverted_pendulum"): 
    for i in range(n_runs): 
        os.system(f"python -m mbrl.examples.main algorithm={algorithm} overrides={env_config} seed={i}")
        print(f"Run {i+1} completed.")

if __name__ == "__main__": 
    run()