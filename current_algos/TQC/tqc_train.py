import argparse
import copy
import pickle
import random
import sys
import time

import gym
import numpy as np
import pybulletgym
import torch
from configs.continuous_actions import __path__
from common.eval_plot import plot_from_progress
from current_algos.TQC.tqc_agent import *
from current_envs.envs import *
from current_envs.wrappers.gym_POMDP_wrapper import gym_POMDP_wrapper

def evaluate_policy(test_env, test_agent,c):
    test_agent.mode = "test"
    rets = []
    
    for _ in range(c["eval_episodes"]):
        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if c["input_norm"]:
            s = test_agent.inp_normalizer.normalize(s, mode="test")
        cur_ret = 0

        d = False
        eval_epi_steps = 0
        
        while not d:

            eval_epi_steps += 1

            # select action
            a = test_agent.select_action(s)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if c["input_norm"]:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c["env"]["max_episode_steps"]:
                break
        
        # compute average return and append it
        rets.append(cur_ret)
    
    return rets

def train(c, agent_name):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init envs
    env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])
    test_env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])

    # wrappers
    for wrapper in c["env"]["wrappers"]:
        wrapper_kwargs = c["env"]["wrapper_kwargs"][wrapper]
        env = eval(wrapper)(env, **wrapper_kwargs)
        test_env = eval(wrapper)(test_env, **wrapper_kwargs)

    # get state_shape
    if c["env"]["state_type"] == "image":
        raise NotImplementedError("Currently, image input is not available for continuous action spaces.")
    
    elif c["env"]["state_type"] == "feature":
        state_dim = env.observation_space.shape[0]

    # seeding
    env.seed(c["seed"])
    test_env.seed(c["seed"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])

    # init agent
    agent = TQC_Agent(mode            = "train",
                      action_dim      = env.action_space.shape[0], 
                      state_dim       = state_dim, 
                      action_high     = env.action_space.high[0],
                      action_low      = env.action_space.low[0], 
                      top_quantiles_to_drop=c["top_quantiles_to_drop"],
                      n_quantiles     = c["n_quantiles"],
                      n_critics       = c["n_critics"],
                      actor_weights   = c["actor_weights"],
                      critic_weights  = c["critic_weights"],
                      input_norm      = c["input_norm"],
                      input_norm_prior= c["input_norm_prior"],
                      gamma           = c["gamma"],
                      tau             = c["tau"],
                      lr_actor        = c["lr_actor"],
                      lr_critic       = c["lr_critic"],
                      buffer_length   = c["buffer_length"],
                      batch_size      = c["batch_size"],
                      device          = c["device"],                         
                      env_str         = c["env"]["name"],
                      info            = c["env"]["info"],
                      seed            = c["seed"])
    
    # get initial state and normalize it
    s = env.reset()
    if c["input_norm"]:
        s = agent.inp_normalizer.normalize(s, mode="train")

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0
    
    # main loop    
    for total_steps in range(c["timesteps"]):

        epi_steps += 1
        
        # select action
        if total_steps < c["act_start_step"]:
            a = np.random.uniform(low=agent.action_low, high=agent.action_high, size=agent.action_dim)
        else:
            a = agent.select_action(s)
        # perform step
        s2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == c["env"]["max_episode_steps"] else d

        # potentially normalize s2
        if c["input_norm"]:
            s2 = agent.inp_normalizer.normalize(s2, mode="train")

        # add epi ret
        epi_ret += r
        
        # memorize
        agent.memorize(s, a, r, s2, d)
        
        # train
        if (total_steps >= c["upd_start_step"]):
            agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == c["env"]["max_episode_steps"]):
            
            # reset to initial state and normalize it
            s = env.reset()
            if c["input_norm"]:
                s = agent.inp_normalizer.normalize(s, mode="train")
            
            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % c["epoch_length"] == 0 and (total_steps + 1) > c["upd_start_step"]:

            epoch = (total_steps + 1) // c["epoch_length"]

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent),c=c)
            for ret in eval_ret:
                agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Avg_Q_val", average_only=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            agent.logger.log_tabular("Critic_loss", average_only=True)
            agent.logger.log_tabular("Actor_loss", average_only=True)
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=c["env"]["name"], info=c["env"]["info"])

            # save weights
            torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
            torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")
    
            # save input normalizer values 
            if c["input_norm"]:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)
        
if __name__ == "__main__":

    # init and prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="persist.json")
    parser.add_argument("--agent_name", type=str, default="tqc")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
        # read config file
    with open(__path__[0] + "/" + args.config_file) as f:
        c = json.load(f)

    # potentially overwrite seed
    if args.seed is not None:
        c["seed"] = args.seed

    # convert certain keys in integers
    for key in ["seed", "timesteps", "epoch_length", "eval_episodes", "buffer_length", "act_start_step",\
         "upd_start_step", "upd_every", "batch_size"]:
        c[key] = int(c[key])

    # handle maximum episode steps
    if c["env"]["max_episode_steps"] == -1:
        c["env"]["max_episode_steps"] = np.inf
    else:
        c["env"]["max_episode_steps"] = int(c["env"]["max_episode_steps"])

    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    # torch.autograd.set_detect_anomaly(True)
    train(c, args.agent_name)

