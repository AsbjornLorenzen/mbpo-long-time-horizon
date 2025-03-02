import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.macura as macura
import mbrl.algorithms.m2ac as m2ac
import mbrl.algorithms.macura_modified_env as macura_modified_env

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
    if cfg.algorithm.name == "macura_modified_env":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)
        test_env2, *_ = mbrl.util.env.EnvHandler.make_env(cfg, test_env=True)

        # double gravity
        test_env.env.unwrapped.model.opt.gravity[2] = -19.64
        test_env2.env.unwrapped.model.opt.gravity[2] = -19.64

        # double length of pole
        pole_geom_id = test_env.env.unwrapped.model.geom('cpole').id
        # original_size = test_env.env.unwrapped.model.geom(pole_geom_id).size.copy()
        test_env.env.unwrapped.model.geom(pole_geom_id).size[1] = 1.2

        pole_geom_id = test_env2.env.unwrapped.model.geom('cpole').id
        # original_size = test_env2.env.unwrapped.model.geom(pole_geom_id).size.copy()
        test_env2.env.unwrapped.model.geom(pole_geom_id).size[1] = 1.2

        # reset to apply changes
        test_env.env.reset()
        test_env2.env.reset()

        return macura_modified_env.train(env, test_env,test_env2 ,term_fn, cfg)

if __name__ == "__main__":
        run()