import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.macura as macura
import mbrl.algorithms.macura_new as macura_new
import mbrl.algorithms.m2ac as m2ac
import mbrl.algorithms.macura_modified_env as macura_modified_env
import mbrl.algorithms.macura_top_k as macura_top_k
import mbrl.algorithms.macura_importance_sampling as macura_importance_sampling
import mbrl.algorithms.macura_data_filter as macura_data_filter
import mbrl.algorithms.macura_wildcard as macura_wildcard
import os

"""
Use this to decide what GPU to run on!
"""
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"



import mbrl.util.env

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    # creates environment and the termination and reward function of environment
    # Therefore it uses cfg.overrides.env : gym___HalfCheetah-v2, where gym refers to the OpenAIGym and after ___ you
    # put the environment name

    print(f"Using the following algorithm: {cfg.algorithm.name}!")

    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg, test_env=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "mbpo":
        # test_env is used for evaluating the model after each training epoch but it is not clear why not env is used
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "m2ac":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return m2ac.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "macura":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
       
        return macura.train(env, test_env,test_env2 ,term_fn, cfg)
    
    if cfg.algorithm.name == "macura_new":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return macura_new.train(env, test_env,test_env2 ,term_fn, cfg)
    
    if cfg.algorithm.name == "macura_top_k":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return macura_top_k.train(env, test_env,test_env2 ,term_fn, cfg)

    if cfg.algorithm.name == "macura_importance_sampling":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return macura_importance_sampling.train(env, test_env,test_env2 ,term_fn, cfg)

    if cfg.algorithm.name == "macura_data_filter":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return macura_data_filter.train(env, test_env,test_env2 ,term_fn, cfg)        
    

    if cfg.algorithm.name == "macura_wildcard":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        return macura_wildcard.train(env, test_env,test_env2 ,term_fn, cfg)

    if cfg.algorithm.name == "macura_modified_env":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)

        # double gravity
        test_env.env.unwrapped.model.opt.gravity[2] = -19.64
        test_env2.env.unwrapped.model.opt.gravity[2] = -19.64

        # For modified cheetah: KeyError: "Invalid name 'cpole'. Valid names: ['bfoot', 'bshin', 'bthigh', 'ffoot', 'floor', 'fshin', 'fthigh', 'head', 'torso']"
        # # double length of pole
        # pole_geom_id = test_env.env.unwrapped.model.geom('cpole').id
        # # original_size = test_env.env.unwrapped.model.geom(pole_geom_id).size.copy()
        # test_env.env.unwrapped.model.geom(pole_geom_id).size[1] = 1.2

        # pole_geom_id = test_env2.env.unwrapped.model.geom('cpole').id
        # # original_size = test_env2.env.unwrapped.model.geom(pole_geom_id).size.copy()
        # test_env2.env.unwrapped.model.geom(pole_geom_id).size[1] = 1.2

        # model = test_env.env.unwrapped.model
        # model2 = test_env2.env.unwrapped.model
        # joint_id = model.joint("hinge").id  # Use the actual name from your env
        # model.dof_damping[joint_id] = 0.2  # Default is usually very small
        # joint_id = model2.joint("hinge").id  # Use the actual name from your env
        # model2.dof_damping[joint_id] = 0.2  # Default is usually very small

        # For modified lunarlander: 
        # maybe change gravity or add wind.
        # try the following (output from Claude):
        # Changing gravity (which is a single value in LunarLander)
        # env.unwrapped.world.gravity = (0, -5.0)  # Default is (0, -10.0)

        # Changing engine power
        # env.unwrapped.MAIN_ENGINE_POWER = 20.0  # Default is 13.0

        # Changing wind power
        # env.unwrapped.WIND_POWER = 2.0  # Default is 0.0 in no-wind version

        # Changing initial random position range
        # env.unwrapped.INITIAL_RANDOM = 0.5  # Default is 1.0

        # Changing leg spring constant
        # env.unwrapped.LEG_SPRING_TORQUE = 50.0  # Default is 40.0

        # Changing fuel usage
        # env.unwrapped.SIDE_ENGINE_FUEL_COST = 0.1  # Default is 0.03
        # env.unwrapped.MAIN_ENGINE_FUEL_COST = 0.3  # Default is 0.3

        # Changing leg away/down positions for the landing gear
        # env.unwrapped.LEG_AWAY = 20  # Default is 20
        # env.unwrapped.LEG_DOWN = 18  # Default is 18


        # reset to apply changes
        test_env.env.reset()
        test_env2.env.reset()

        return macura_modified_env.train(env, test_env,test_env2 ,term_fn, cfg)

if __name__ == "__main__":
        run()