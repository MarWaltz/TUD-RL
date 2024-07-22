import math
import pickle
import random
from copy import deepcopy

import gym
import matplotlib.patches as patches
import numpy as np
import torch
from gym import spaces
from matplotlib import pyplot as plt

from tud_rl.envs._envs.AIS_Helper import AIS_Ship, CPANet
from tud_rl.envs._envs.HHOS_Fnc import cte, get_init_two_wp
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.MMG_TargetShip import Path
from tud_rl.envs._envs.VesselFnc import (ED, angle_to_2pi, bng_rel, cpa, dtr,
                                         head_inter, rtd, xy_from_polar)
from tud_rl.envs._envs.VesselPlots import TrajPlotter, get_rect, plot_jet


class AIS_Env(gym.Env):
    """This environment contains an agent which gets thrown into an environment with vessels following real AIS trajectories.."""

    def __init__(self, 
                 AIS_path : str = None,
                 supervised_path : str = None,
                 N_TSs : int = 5, 
                 pdf_traj : bool = False,
                 cpa : bool = False,
                 continuous : bool = True,
                 include_tcpa : bool = False):
        super().__init__()

        # Simulation settings
        self.delta_t = 5.0    # simulation time interval (in s)
        self.N_max   = 12_000 # maximum N-coordinate (in m)
        self.E_max   = 8_500  # maximum E-coordinate (in m)

        self.N_TSs = N_TSs # maximum number of other vessels

        # Read AIS data
        with open(AIS_path,"rb") as f:
            self.AIS_data = pickle.load(f)

        # Supervised learning estimator for DCPA / TCPA
        self.cpa = cpa
        self.supervised_path = supervised_path
        self.include_tcpa = include_tcpa

        if supervised_path is not None:
            self.forecast_window = 200  # forecast window
            self.n_obs_est   = 75      # number of points used for iterative 'n_forecasts'-step ahead forecasts
            self.n_forecasts = 10       # number of points to predict per forward pass
            self.data_norm = 10.        # normalization factor of data during SL phase
            self.diff = True            # whether to forecast positions or differences in positions

            assert self.forecast_window % self.n_forecasts == 0, "The number of points you want to estimate is not a multiple of the single-step forecasts."

            self.CPA_net = CPANet(n_forecasts=self.n_forecasts)
            self.CPA_net.load_state_dict(torch.load(supervised_path))

            # Evaluation mode and no gradient comp
            self.CPA_net.eval()
            for p in self.CPA_net.parameters():
                p.requires_grad = False

        self.n_loops = 4
        self.num_obs_OS = 2                               # number of observations for the OS
        
        if cpa:
            if self.include_tcpa:
                self.num_obs_TS = 6                               # number of observations per TS
            else:
                self.num_obs_TS = 5
        else:
            self.num_obs_TS = 4

        # Env config
        self.coll_dist = 350 # m
        self.path_width_left  = 200 # m
        self.path_width_right = 2000 # m

        # Plotting
        self.pdf_traj = pdf_traj   # whether to plot trajectory after termination
        self.TrajPlotter = TrajPlotter(plot_every=5 * 60, delta_t=self.delta_t)
        self.state_names = ["OS_spd", "CTE", "ED", "bng", "head-inter", "TS_spd", "DCPA", "TCPA"]

        # Gym definitions
        obs_size = self.num_obs_OS + self.N_TSs * self.num_obs_TS

        self.continuous = continuous

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        if self.continuous:
            self.action_space = spaces.Box(low  = np.full(1, -1.0, dtype=np.float32), 
                                           high = np.full(1,  1.0, dtype=np.float32))
        else:
            self.action_space = spaces.Discrete(3)

        self.dhead = self.n_loops * dtr(0.6)

        # Custom inits
        self._max_episode_steps = 15
        self.r = 0

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        while True:
            try:
                # sample situation: 0 - TS goes curve, 1 - TS goes linear
                self.sit = random.getrandbits(1)

                if self.sit == 0:
                    self.ttpt = 240 + np.random.random() * 60 # time until TS is at turning point (time to turning point) # 240 initially
                    self.ttc  = 240 #+ np.random.random() * 60 # time to collision for OS spawning
                else:
                    self.ttpt = 1000 + np.random.random() * 60 # time passed since TS was at turning point (time to turning point) # 180
                    self.ttc  = 240 #+ np.random.random() * 60 # time to collision for OS spawning

                # init TSs
                self.TSs = [self._get_AIS_TS()]

                # init agent
                self.OS = KVLCC2(N_init   = 0, 
                                E_init   = 0, 
                                psi_init = 0.0,
                                u_init   = 0.0,
                                v_init   = 0.0,
                                r_init   = 0.0,
                                delta_t  = self.delta_t,
                                N_max    = self.N_max,
                                E_max    = self.E_max,
                                nps      = 1.940969) # 1.940969 gives 8 m/s, 1.21312 gives 5 m/s speed

                # set longitudinal speed to near-convergence
                # Note: if we don't do this, the TCPA calculation for spawning other vessels is heavily biased
                self.OS.nu[0] = self.OS._get_u_from_nps(self.OS.nps, psi=self.OS.eta[2])

                # ---- place OS in head-on mode to AIS TS ----
                TS = self.TSs[0]

                # 1. Estimate TS position linearly
                E_hit = TS.e + self.ttc * TS.ve
                N_hit = TS.n + self.ttc * TS.vn

                # 2. Backtrace position of OS
                OS_head = angle_to_2pi(TS.head + np.pi)
                ve_OS, vn_OS = xy_from_polar(r=self.OS.nu[0], angle=OS_head)
                E_OS = E_hit - self.ttc * ve_OS
                N_OS = N_hit - self.ttc * vn_OS

                # 3. Set values
                self.OS.eta = np.array([N_OS, E_OS, OS_head])

                # create path
                ve, vn = xy_from_polar(r=self.OS.nu[0], angle=self.OS.eta[2])
                path_n = self.OS.eta[0] + np.arange(1000) * self.delta_t * vn
                path_e = self.OS.eta[1] + np.arange(1000) * self.delta_t * ve

                self.path = Path(level="global", north=path_n, east=path_e)
                self.path_one = deepcopy(self.path)
                self.path_one.move(-self.path_width_left)

                self.path_two = deepcopy(self.path)
                self.path_two.move(self.path_width_right)

                # add some noise
                self.OS.eta[0] += np.random.randn() * 1.0
                self.OS.eta[1] += np.random.randn() * 1.0
                self.OS.eta[2] += dtr(np.random.uniform(-3, 0))

                # generate other distracting target ships
                for _ in range(self.N_TSs-1):
                    self.TSs.append(self._get_usual_TS())

                # update cte
                _, wp1_N, wp1_E, _, wp2_N, wp2_E = get_init_two_wp(n_array=self.path.north, e_array=self.path.east, 
                                                                a_n=self.OS.eta[0], a_e=self.OS.eta[1])
                self.cte = cte(N1=wp1_N, E1=wp1_E, N2=wp2_N, E2=wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1])

                # init state
                self._set_state()
                self.state_init = self.state

                # trajectory storing
                self.TrajPlotter.reset(OS=self.OS, TSs=self.TSs, N_TSs=self.N_TSs)

            except:
                continue
            else:
                break
        return self.state

    def _get_AIS_TS(self) -> AIS_Ship:
        """Samples a target ship from the data."""
        return AIS_Ship(data=self.AIS_data, sit=self.sit, ttpt=self.ttpt)

    def _get_usual_TS(self) -> KVLCC2:
        """Generates a linearly-moving target ship."""
        TS = deepcopy(self.OS)
        TS.nu[0] += np.random.normal(0., 0.2)
        case = random.randint(0,1)

        # TS close to OS
        if case == 0:
            d = np.random.uniform(500., 1000.)
            head = angle_to_2pi(self.OS.eta[2] + dtr(np.random.uniform(-30, 30)))
            e_add, n_add = xy_from_polar(r=d, angle=angle_to_2pi(head + np.pi))
 
            TS.eta[0] += n_add
            TS.eta[1] += e_add
            TS.eta[2] = head

        # Irrelevant TS
        else:
            while True:
                n    = TS.eta[0] + np.random.uniform(1500., 3000.) * random.choice([-1, 1])
                e    = TS.eta[1] + np.random.uniform(1500., 3000.) * random.choice([-1, 1])
                head = TS.eta[2] + dtr(np.random.uniform(-180, 180))
                
                dcpa, _ = cpa(NOS=self.OS.eta[0], EOS=self.OS.eta[1], NTS=n, ETS=e, chiOS=self.OS._get_course(), chiTS=head,
                                 VOS=self.OS._get_V(), VTS=TS._get_V())
                if dcpa >= 800.:
                    TS.eta[0] = n
                    TS.eta[1] = e
                    TS.eta[2] = head
                    break              
        return TS

    def _iterative_forecasts(self, n_for_est : np.ndarray, e_for_est : np.ndarray) -> tuple:
        """Performs iterative 'n_forecasts'-step ahead forecasts using the given input points.
        Returns:
            np.ndarray(self.forecast_windows,) : predicted n_points
            np.ndarray(self.forecast_windows,) : predicted e_points
        """
        assert (len(n_for_est) == self.n_obs_est) and (len(e_for_est) == self.n_obs_est), "Size mismatch for estimation data."
        
        # Rotate estimation data
        #rot_angle = 0.0 - bng_abs(N0=n_for_est[0], E0=e_for_est[0], N1=n_for_est[1], E1=e_for_est[1])
        #e_for_est, n_for_est = rotate_point(x=e_for_est, y=n_for_est, cx=e_for_est[0], cy=n_for_est[0], angle=-rot_angle)

        # Output setup
        n_steps = int(self.forecast_window / self.n_forecasts)
        n_predicted = np.array([])
        e_predicted = np.array([])

        for _ in range(n_steps):        

            if self.diff:
                # Take last 'n_obs_est' elements for prediction
                n_diff = torch.tensor(np.diff(n_for_est[-self.n_obs_est:]), dtype=torch.float32) / self.data_norm
                e_diff = torch.tensor(np.diff(e_for_est[-self.n_obs_est:]), dtype=torch.float32) / self.data_norm

                n_diff = torch.reshape(n_diff, (1, self.n_obs_est-1, 1))
                e_diff = torch.reshape(e_diff, (1, self.n_obs_est-1, 1))

                # Add absolute position
                pos = torch.reshape(torch.tensor([e_for_est[-1], n_for_est[-1]], dtype=torch.float32) / (100. * self.data_norm), shape=(1, 2))

                # Forward pass
                est = self.CPA_net(e_diff, n_diff, pos)
                e_add = est[0][:self.n_forecasts].detach().numpy() * self.data_norm
                n_add = est[0][self.n_forecasts:].detach().numpy() * self.data_norm
                
                new_e = e_for_est[-1] + np.cumsum(e_add)
                new_n = n_for_est[-1] + np.cumsum(n_add)

            else:
                # Take last 'n_obs_est' elements for prediction
                n_in = torch.tensor(n_for_est[-self.n_obs_est:], dtype=torch.float32)
                e_in = torch.tensor(e_for_est[-self.n_obs_est:], dtype=torch.float32)

                n_in = torch.reshape(n_in, (1, self.n_obs_est, 1))
                e_in = torch.reshape(e_in, (1, self.n_obs_est, 1))

                # Forward pass
                est = self.CPA_net(e_in, n_in)
                new_e = est[0][:self.n_forecasts].detach().numpy() * self.data_norm
                new_n = est[0][self.n_forecasts:].detach().numpy() * self.data_norm

            # Append estimates
            n_predicted = np.concatenate((n_predicted, new_n))
            e_predicted = np.concatenate((e_predicted, new_e))

            # Update estimation data
            n_for_est = np.concatenate((n_for_est, new_n))
            e_for_est = np.concatenate((e_for_est, new_e))

        # Rotate back
        #e_predicted, n_predicted = rotate_point(x=np.array(e_predicted), y=np.array(n_predicted), cx=e_for_est[0], cy=n_for_est[0], #angle=rot_angle)
        return n_predicted, e_predicted

    def _get_dcpa_tcpa_supervised(self, TS: AIS_Ship):
        """Returns DCPA and TCPA using the corresponding neural network, 
        which has been trained separately."""
        # Only every 10th step for computation time
        if not hasattr(TS, "tcpa") or self.step_cnt % 2 == 0:

            ptr = TS.ptr

            # Take last 'n_obs_est' observations, including the current one
            n_base = TS.n_traj[ptr-self.n_obs_est+1:ptr+1]
            e_base = TS.e_traj[ptr-self.n_obs_est+1:ptr+1]

            # Predict future
            n_future, e_future = self._iterative_forecasts(n_for_est=n_base.copy(), e_for_est=e_base.copy())

            # Stack it together
            n_TS = np.concatenate((n_base, n_future))
            e_TS = np.concatenate((e_base, e_future))

            TS.n_traj_pred = n_TS
            TS.e_traj_pred = e_TS

            # Agent behavior
            n_OS, e_OS, head_OS = self.OS.eta
            vOS_e, vOS_n = xy_from_polar(r=self.OS._get_V(), angle=head_OS)

            n_OS_full = n_OS + vOS_n * np.arange(-self.n_obs_est + 1, self.forecast_window+1) * self.delta_t
            e_OS_full = e_OS + vOS_e * np.arange(-self.n_obs_est + 1, self.forecast_window+1) * self.delta_t

            # Compute dcpa and tcpa
            dist = np.sqrt((n_OS_full - n_TS)**2 + (e_OS_full - e_TS)**2)

            dcpa = max([0, np.min(dist)])
            tcpa = self.delta_t * (np.argmin(dist) - self.forecast_window - self.n_obs_est)
        else:
            dcpa = TS.dcpa
            tcpa = TS.tcpa - self.delta_t
        return dcpa, tcpa

    def _update_dcpa_tcpa_real_traj(self, TS : AIS_Ship) -> None:
        """Updates the DCPA and TCPA by accessing the true future trajectory."""
        #raise NotImplementedError()

        # Only every 10th step for computation time
        if not hasattr(TS, "tcpa_real") or self.step_cnt % 1 == 0:
            l = len(TS.e_traj)
            #future_pts = l - (TS.ptr + 1)
            #future = np.arange(1, future_pts + 1)
            #past = np.arange(-TS.ptr, 1)

            # Agent behavior
            n_OS, e_OS, head_OS = self.OS.eta
            vOS_e, vOS_n = xy_from_polar(r=self.OS._get_V(), angle=head_OS)

            n_OS_full = n_OS + vOS_n * np.arange(-TS.ptr, l - TS.ptr) * self.delta_t
            e_OS_full = e_OS + vOS_e * np.arange(-TS.ptr, l - TS.ptr) * self.delta_t

            # Compute dcpa and tcpa
            dist = np.sqrt((n_OS_full - TS.n_traj)**2 + (e_OS_full - TS.e_traj)**2)

            TS.dcpa_real = max([0, np.min(dist)])
            TS.tcpa_real = self.delta_t * (np.argmin(dist) - TS.ptr)
        else: 
            TS.tcpa_real -= self.delta_t

    def _set_state(self):
        """State consists of (all from agent's perspective): 
        
        OS:
            speed      
        Dynamic obstacle:
            ED_TS
            relative bearing
            heading intersection angle C_T
            speed (V)
        """
        # quick access for OS
        N0, E0, head0 = self.OS.eta

        # OS related
        state_OS = np.array([self.OS._get_V() / 5.0, self.cte / self.path_width_left])

        # Dynamic obstacle related
        state_TSs = []

        for TS_idx, TS in enumerate(self.TSs):
            
            if TS_idx == 0:
                N = TS.n
                E = TS.e
                headTS = TS.head
            else:
                N, E, headTS = TS.eta

            # Distance
            ED_OS_TS_norm = (ED(N0=N0, E0=E0, N1=N, E1=E, sqrt=True)-self.coll_dist) / 3000.

            # Relative bearing
            bng_rel_TS = bng_rel(N0=N0, E0=E0, N1=N, E1=E, head0=head0, to_2pi=False) / (math.pi)

            # Heading intersection angle
            C_TS = head_inter(head_OS=head0, head_TS=headTS, to_2pi=False) / (math.pi)

            # Speed
            if TS_idx == 0:
                V_TS = TS.v / 5.0
            else:
                V_TS = TS._get_V() / 5.0

            # Interface to supervised learning module
            if self.cpa:
                if TS_idx == 0:
                    if self.supervised_path is not None:
                        try:
                            dcpa, tcpa = self._get_dcpa_tcpa_supervised(TS)
                        except:
                            dcpa = TS.dcpa
                            tcpa = TS.tcpa

                        TS.dcpa = dcpa
                        TS.tcpa = tcpa
                    else:
                        self._update_dcpa_tcpa_real_traj(TS)

                else:
                    dcpa, tcpa = cpa(NOS=N0, EOS=E0, NTS=N, ETS=E, chiOS=self.OS._get_course(), chiTS=TS._get_course(),
                                     VOS=self.OS._get_V(), VTS=TS._get_V())
                    TS.dcpa = dcpa
                    TS.tcpa = tcpa

                # Normalize
                dcpa = (dcpa - self.coll_dist) / 100.0

                # Store it
                if self.include_tcpa:
                    tcpa = tcpa / 100.0
                    state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS, V_TS, tcpa, dcpa])
                else:
                    state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS, V_TS, dcpa])
            else:
                # store it
                state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS, V_TS])

        # sort according to tcpa
        if self.cpa:
            # for debugging:
            self.first_TS_idx = np.argmin([el[-1] for el in state_TSs])

            # usual
            state_TSs = sorted(state_TSs, key=lambda x: x[-1], reverse=False)

            # mask dcpa
            state_TSs_masked = []
            for el in state_TSs:
                el[-1] = 1. if el[-1] > 0. else -1.
                state_TSs_masked.append(el)
            state_TSs = state_TSs_masked

        # or according to Euclidean distance
        else:
            state_TSs = sorted(state_TSs, key=lambda x: x[0], reverse=False)
        
        state_TSs = np.array(state_TSs).flatten()

        # combine state
        self.state = np.concatenate([state_OS, state_TSs], dtype=np.float32)

        # add some noise
        #self.state[0:6] += np.random.normal(loc=0.0, scale=0.05, size=len(self.state[0:6]))

    def _heading_control(self, a : float):
        """Controls the heading of the vessel."""
        if self.continuous:
            assert (-1 <= a) and (a <= 1), "Unknown action."
            self.OS.eta[2] = angle_to_2pi(self.OS.eta[2] + a * self.dhead)
        else:
            assert a in [0, 1, 2], "Unknown action."

            if a == 0:
                pass
            elif a == 1:
                self.OS.eta[2] = angle_to_2pi(self.OS.eta[2] + self.dhead)
            else:
                self.OS.eta[2] = angle_to_2pi(self.OS.eta[2] - self.dhead)

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        a = float(a)
        self.a = a
        self._heading_control(a)

        # update agent dynamics
        for _ in range(self.n_loops):
            self.OS._upd_dynamics()

        # update cte
        _, wp1_N, wp1_E, _, wp2_N, wp2_E = get_init_two_wp(n_array=self.path.north, e_array=self.path.east, 
                                                           a_n=self.OS.eta[0], a_e=self.OS.eta[1])
        self.cte = cte(N1=wp1_N, E1=wp1_E, N2=wp2_N, E2=wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1])

        # update environmental dynamics, e.g., other vessels
        for _ in range(self.n_loops):
            for TS in self.TSs:
                TS._upd_dynamics()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += (self.delta_t * self.n_loops)

        # trajectory plotting
        self.TrajPlotter.step(OS=self.OS, TSs=self.TSs, respawn_flags=[False] * self.N_TSs, step_cnt=self.step_cnt)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()
        return self.state, self.r, d, {}

    def _calculate_reward(self, a):
        """Returns reward of the current state."""
        N0, E0, _ = self.OS.eta

        self.r = 0.0

        # Path leaving
        if -self.cte >= self.path_width_left:
            self.r -= 1.0

        elif self.cte >= self.path_width_right:
            self.r -= 1.0

        # Collision
        for TS in self.TSs:
            if isinstance(TS, AIS_Ship):
                n = TS.n
                e = TS.e
            else:
                n = TS.eta[0]
                e = TS.eta[1]

            if ED(N0=N0, E0=E0, N1=n, E1=e) <= self.coll_dist:
                self.r -= 1.0

    def _done(self):
        """Returns boolean flag whether episode is over."""

        d = False

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            d = True

        # create pdf trajectory
        if d and self.pdf_traj:
            self.TrajPlotter.plot_traj_fnc(E_max=self.E_max, N_max=self.N_max, goal=self.goal,\
                goal_reach_dist=self.goal_reach_dist, Lpp=self.OS.Lpp, step_cnt=self.step_cnt)
        return d

    def __str__(self) -> str:
        N0, E0, head0 = self.OS.eta
        u, v, r = np.round(self.OS.nu,3)

        ste = f"Step: {self.step_cnt}"
        pos = f"N: {N0:.2f}, E: {E0:.2f}, " + r"$\psi$: " + f"{rtd(head0):.2f}°"
        vel = f"u: {u:.4f}, v: {v:.4f}, r: {r:.4f}"
        cte = f"CTE: {self.cte:.2f}"
        return ste + "\n" + pos + "\n" + vel + "\n" + cte

    def render(self, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # plot every nth timestep (except we only want trajectory)
        if not self.pdf_traj:

            if self.step_cnt % 1 == 0: 

                # check whether figure has been initialized
                if len(plt.get_fignums()) == 0:
                    #self.fig = plt.figure(figsize=(11, 6))
                    #self.gs  = self.fig.add_gridspec(2, 2)
                    #self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                    #self.ax1 = self.fig.add_subplot(self.gs[0, 1]) # reward
                    #self.ax2 = self.fig.add_subplot(self.gs[1, 0]) # state
                    #self.ax3 = self.fig.add_subplot(self.gs[1, 1]) # action

                    self.fig2 = plt.figure(figsize=(10,7))
                    self.fig2_ax = self.fig2.add_subplot(111)

                    plt.ion()
                    plt.show()
                
                # ------------------------------ ship movement --------------------------------
                for ax in [self.fig2_ax]: # , self.ax0
                    # clear prior axes, set limits and add labels and title
                    ax.clear()
                    ax.set_xlim(0, self.E_max + 1000)
                    ax.set_ylim(-1000, self.N_max)

                    # Axis
                    ax.set_xlabel("East [m]", fontsize=8)
                    ax.set_ylabel("North [m]", fontsize=8)

                    # access
                    N0, E0, head0 = self.OS.eta
                    chiOS = self.OS._get_course()
                    VOS = self.OS._get_V()

                    # set OS
                    rect = get_rect(E = E0, N = N0, width = int(self.coll_dist/2), length = int(self.coll_dist), heading = head0,
                                    linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)

                    # path
                    ax.plot(self.path.east, self.path.north)
                    ax.plot(self.path_one.east, self.path_one.north, alpha=0.75)
                    ax.plot(self.path_two.east, self.path_two.north, alpha=0.75)
                   
                    # step information
                    ax.text(0.05 * self.E_max, 0.9 * self.N_max, self.__str__(), fontsize=8)

                    # add jets according to COLREGS
                    for COLREG_deg in [5, 355]:
                        ax = plot_jet(axis = ax, E=E0, N=N0, l = 1000, 
                                      angle = head0 + dtr(COLREG_deg), color='black', alpha=0.3)


                    # set other vessels
                    for TS_idx, TS in enumerate(self.TSs):

                        # access
                        if isinstance(TS, AIS_Ship):
                            N, E, headTS = TS.n, TS.e, TS.head
                            chiTS = TS.head
                            VTS = TS.v
                        else:
                            N, E, headTS = TS.eta
                            chiTS = TS._get_course()
                            VTS = TS._get_V()

                        if not hasattr(self, "first_TS_idx"):
                            col = "blue"
                        else:
                            if TS_idx == self.first_TS_idx:
                                col = "red"
                            else:
                                col = "blue"

                        # place TS
                        rect = get_rect(E = E, N = N, width = int(self.coll_dist/2), length = int(self.coll_dist), heading = headTS,
                                        linewidth=1, edgecolor=col, facecolor='none')
                        ax.add_patch(rect)

                        # domain
                        circle = patches.Circle((E, N), radius=self.coll_dist, edgecolor=col, facecolor="none")
                        ax.add_patch(circle)

                        # add two jets according to COLREGS
                        #for COLREG_deg in [5, 355]:
                        #    ax= plot_jet(axis = ax, E=E, N=N, l = self.sight, 
                        #                 angle = headTS + dtr(COLREG_deg), color=col, alpha=0.3)

                        # compute CPA measures
                        DCPA_TS, TCPA_TS, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N, ETS=E, \
                            chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS, get_positions=True)

                        #ax.text(E + 100, N , f"TCPA: {np.round(TCPA_TS, 2)}", fontsize=7,
                        #            horizontalalignment='center', verticalalignment='center', color=col)
                        #ax.text(E + 100, N-100, f"DCPA: {np.round(DCPA_TS, 2)}", fontsize=7,
                        #            horizontalalignment='center', verticalalignment='center', color=col)

                        try:
                            #ax.text(E + 10, N - 100, f"TCPA_real: {np.round(TS.tcpa_real, 2)}", fontsize=7,
                            #        horizontalalignment='center', verticalalignment='center', color=col)
                            ax.text(E + 10, N - 300, f"DCPA_real: {np.round(TS.dcpa_real - self.coll_dist, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        except:
                            pass

                        try:
                            #ax.text(E + 10, N - 500, f"TCPA_est: {np.round(TS.tcpa, 2)}", fontsize=7,
                            #        horizontalalignment='center', verticalalignment='center', color=col)
                            ax.text(E + 10, N - 700, f"DCPA_est: {np.round(TS.dcpa - self.coll_dist, 2)} | TS-idx: {TS_idx}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        except:
                            pass

                        # Plot true trajectory
                        if isinstance(TS, AIS_Ship):
                            ax.scatter(TS.e_traj, TS.n_traj, s=4, color="black", alpha=0.05)

                        # Plot estimated trajectory
                        if hasattr(TS, "n_traj_pred"):
                            ax.scatter(TS.e_traj_pred, TS.n_traj_pred, s=4, color="red")

                # ------------------------------ reward plot --------------------------------
                if False:
                    if self.step_cnt == 0:
                        self.ax1.clear()
                        self.ax1.old_r = 0
                        self.ax1.old_time = 0

                    self.ax1.set_xlim(0, self._max_episode_steps)
                    #self.ax1.set_ylim(-1.25, 0.1)
                    self.ax1.set_xlabel("Timestep in episode")
                    self.ax1.set_ylabel("Reward")

                    self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r, self.r], color = "black")
                    self.ax1.old_r = self.r
                    self.ax1.old_time = self.step_cnt

                # ------------------------------ state plot --------------------------------
                if False:
                    if self.step_cnt == 0:
                        self.ax2.clear()
                        self.ax2.old_time = 0
                        self.ax2.old_state = [0] * (self.num_obs_OS + self.num_obs_TS)

                    self.ax2.set_xlim(0, self._max_episode_steps)
                    self.ax2.set_xlabel("Timestep in episode")
                    self.ax2.set_ylabel("State information")

                    try:
                        for i, obs in enumerate(self.state):
                            if i >= self.num_obs_OS:
                                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_state[i], obs], 
                                               color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i], 
                                               label=self.state_names[i])

                        self.ax2.old_time = self.step_cnt
                        self.ax2.old_state = self.state
                    except:
                        pass

                    if self.step_cnt == 0:
                        self.ax2.legend()

                # ------------------------------ action plot --------------------------------
                if False:
                    if self.step_cnt == 0:
                        self.ax3.clear()
                        self.ax3.old_time = 0
                        self.ax3.old_action = 0

                    self.ax3.set_xlim(0, self._max_episode_steps)
                    self.ax3.set_ylim(-1.1, 2.1)
                    self.ax3.set_xlabel("Timestep in episode")
                    self.ax3.set_ylabel("Action")

                    try:
                        self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_action, self.a], color="black", alpha=0.5)
                        self.ax3.old_time = self.step_cnt
                        self.ax3.old_action = self.a
                    except:
                        pass

                plt.pause(0.001)
