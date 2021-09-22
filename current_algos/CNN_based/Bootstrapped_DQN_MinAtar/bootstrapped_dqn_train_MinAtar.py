import argparse
import copy
import pickle
import random
import time

import gym
import gym_minatar
import numpy as np

import torch
from current_algos.common.eval_plot import plot_from_progress
from current_algos.common.custom_envs import MountainCar
from current_algos.CNN_based.Bootstrapped_DQN_MinAtar.bootstrapped_dqn_agent_MinAtar import *

# training config
TIMESTEPS = 100_000     # overall number of training interaction steps
EPOCH_LENGTH = 5000     # number of time steps between evaluation/logging events
EVAL_EPISODES = 10      # number of episodes to average per evaluation


def evaluate_policy(test_env, test_agent):
    test_agent.mode = "test"
    rets = []
    
    for _ in range(EVAL_EPISODES):
        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if test_agent.input_norm:
            s = test_agent.inp_normalizer.normalize(s, mode="test")

        # change s to be of shape (in_channels, height, width) instead of (height, width, in_channels)
        s = np.moveaxis(s, -1, 0)

        cur_ret = 0
        d = False
        
        while not d:

            # select action
            a = test_agent.select_action(s, active_head=None)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if test_agent.input_norm:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # change s2 to be of shape (in_channels, height, width) instead of (height, width, in_channels)
            s2 = np.moveaxis(s2, -1, 0)

            # s becomes s2
            s = s2
            cur_ret += r

        # compute average return and append it
        rets.append(cur_ret)
    
    return rets

def train(env_str, double, our_estimator, our_alpha, dqn_weights=None, seed=0, device="cpu"):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init env
    if env_str == "MountainCar":
        env = MountainCar(rewardStd=0)
        test_env = MountainCar(rewardStd=0)
        max_episode_steps = env._max_episode_steps
    else:
        env = gym.make(env_str)
        test_env = gym.make(env_str)
        max_episode_steps = np.inf if "MinAtar" in env_str else env._max_episode_steps

    # seeding
    env.seed(seed)
    test_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
    state_shape = (env.observation_space.shape[2], *env.observation_space.shape[0:2])

    # init agent
    agent = CNN_Bootstrapped_DQN_Agent(mode          = "train",
                                       num_actions   = env.action_space.n, 
                                       state_shape   = state_shape,
                                       double        = double,
                                       our_estimator = our_estimator,
                                       our_alpha     = our_alpha,
                                       dqn_weights   = dqn_weights,
                                       device        = device)
    
    # init the active head for action selection
    active_head = np.random.choice(agent.K)

    # get initial state and normalize it
    s = env.reset()
    if agent.input_norm:
        s = agent.inp_normalizer.normalize(s, mode="train")

    # change s to be of shape (in_channels, height, width) instead of (height, width, in_channels)
    s = np.moveaxis(s, -1, 0)

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0
    
    # main loop    
    for total_steps in range(TIMESTEPS):

        epi_steps += 1
        
        # select action
        if total_steps < agent.act_start_step:
            a = np.random.randint(low=0, high=agent.num_actions, size=1, dtype=int).item()
        else:
            a = agent.select_action(s, active_head)
        
        # perform step
        s2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == max_episode_steps else d

        # potentially normalize s2
        if agent.input_norm:
            s2 = agent.inp_normalizer.normalize(s2, mode="train")

        # change s2 to be of shape (in_channels, height, width) instead of (height, width, in_channels)
        s2 = np.moveaxis(s2, -1, 0)

        # add epi ret
        epi_ret += r
        
        # memorize
        agent.memorize(s, a, r, s2, d)

        # train
        if (total_steps >= agent.upd_start_step) and (total_steps % agent.upd_every == 0):
            for _ in range(agent.upd_every):
                agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == max_episode_steps):
 
            # reset active head for action selection
            active_head = np.random.choice(agent.K)

            # reset to initial state and normalize it
            s = env.reset()
            if agent.input_norm:
                s = agent.inp_normalizer.normalize(s, mode="train")
            
            # change s to be of shape (in_channels, height, width) instead of (height, width, in_channels)
            s = np.moveaxis(s, -1, 0)

            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % EPOCH_LENGTH == 0 and (total_steps + 1) > agent.upd_start_step:

            epoch = (total_steps + 1) // EPOCH_LENGTH

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent))
            for ret in eval_ret:
                agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            agent.logger.log_tabular("Q_val", with_min_and_max=True)
            agent.logger.log_tabular("Loss", average_only=True)
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=env_str, info=None)

            # save weights
            torch.save(agent.DQN.state_dict(), f"{agent.logger.output_dir}/{agent.name}_DQN_weights.pth")
    
            # save input normalizer values 
            if agent.input_norm:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)
    
if __name__ == "__main__":
    
    # helper function for parser
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # init and prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_str", type=str, default="Breakout-MinAtar-v0")
    parser.add_argument("--double", type=str2bool, default=False)
    parser.add_argument("--our_estimator", type=str2bool, default=True)
    parser.add_argument("--our_alpha", type=float, default=0.05)
    args = parser.parse_args()

    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    train(env_str=args.env_str, double=args.double, our_estimator=args.our_estimator, our_alpha=args.our_alpha, dqn_weights=None, seed=1, device="cpu")
