import gym
import matplotlib.patches as patches
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from tud_rl.envs.FossenCS2 import CyberShipII
from tud_rl.envs.FossenFnc import (COLREG_COLORS, COLREG_NAMES, ED,
                                   angle_to_2pi, angle_to_pi, bng_rel, dtr,
                                   head_inter, rtd, tcpa)


class FossenEnv(gym.Env):
    """This environment contains an agent steering a CyberShip II."""

    def __init__(self, cnt_approach="tau"):
        super().__init__()

        # simulation settings
        self.delta_t      = 0.5              # simulation time interval (in s)
        self.N_max        = 50               # maximum N-coordinate (in m)
        self.E_max        = 50               # maximum E-coordinate (in m)
        self.N_TSs        = 5                # number of other vessels
        self.safety_dist  = 3                # minimum distance, if less then collision (in m)
        self.jet_length   = 15               # size of the jets for plotting (in m)
        self.cnt_approach = cnt_approach     # whether to control actuator forces or rudder angle and rps directly

        # gym definitions
        obs_size = 8 + self.N_TSs * 6
        self.observation_space  = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                             high = np.full(obs_size,  np.inf, dtype=np.float32))
        
        if cnt_approach in ["tau", "rps_angle"]:
            self.action_space = spaces.Discrete(3)

        elif cnt_approach == "f123":
            self.action_space = spaces.Discrete(9)

        # custom inits
        self._max_episode_steps = 1e3
        self.r = 0
        self.r_head = 0
        self.r_dist = 0
        self.r_coll = 0
        self.r_coll_sigma = 5
        self.state_names = ["u", "v", "r", "N_rel", "E_rel", r"$\Psi$", r"$\Psi_e$", "ED"]


    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # init goal
        self.goal = {"N" : np.random.uniform(self.N_max - 25, self.N_max),
                     "E" : np.random.uniform(self.E_max - 25, self.E_max)}

        # init other vessels
        self.TSs = [CyberShipII(N_init       = np.random.uniform(15, self.N_max), 
                                E_init       = np.random.uniform(15, self.E_max), 
                                psi_init     = np.random.uniform(0, np.pi),
                                u_init       = np.random.uniform(0, 1),
                                v_init       = 0.0,
                                r_init       = 0.0,
                                delta_t      = self.delta_t,
                                N_max        = self.N_max,
                                E_max        = self.E_max,
                                cnt_approach = self.cnt_approach,
                                tau_u        = np.random.uniform(0, 5)) for _ in range(self.N_TSs)]

        # init agent (OS for 'Own Ship') and calculate initial distance to goal
        self.OS = CyberShipII(N_init       = 10.0, 
                              E_init       = 10.0, 
                              psi_init     = np.random.uniform(0, np.pi),
                              u_init       = np.random.uniform(0, 1),
                              v_init       = 0.0,
                              r_init       = 0.0,
                              delta_t      = self.delta_t,
                              N_max        = self.N_max,
                              E_max        = self.E_max,
                              cnt_approach = self.cnt_approach,
                              tau_u        = 3.0)

        self.OS_goal_ED_init = ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=self.goal["N"], E1=self.goal["E"])
        
        # init state
        self._set_state()
        self.state_init = self.state

        return self.state


    def _set_state(self):
        """State consists of (all from agent's perspective): 
        
        OS:
            u, v, r, 
            N_rel, E_rel, heading

        Goal:
            relative bearing
            ED_goal
        
        Dynamic obstacle (for each, sorted by TCPA):
            relative bearing
            heading intersection angle C_T
            u_TS
            ED_TS
            COLREG mode TS (sigma_TS)
            TCPA
        
        Note: Everything is normalized. If the TCPA is < 0s or > 60s for a TS, everything for this TS is set 0.
        """

        N0, E0, head0 = self.OS.eta             # N, E, heading
        chiOS = head0 + self.OS._get_beta()     # course angle (heading + sideslip)
        VOS = self.OS._get_V()                  # aggregated velocity

        #-------------------------------- OS related ---------------------------------
        state_OS = np.concatenate([self.OS.nu, np.array([N0 / self.N_max, E0 / self.E_max, head0 / (2*np.pi)])])


        #------------------------------ goal related ---------------------------------
        OS_goal_ED = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])

        state_goal = np.array([bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=head0) / (2*np.pi), 
                               OS_goal_ED / self.OS_goal_ED_init])


        #--------------------------- dynamic obstacle related -------------------------
        state_TSs = []

        for TS in self.TSs:

            N, E, headTS = TS.eta               # N, E, heading
            chiTS = headTS + TS._get_beta()     # course angle (heading + sideslip)
            VTS = TS._get_V()                   # aggregated velocity

            # relative bearing
            bng_rel_TS = bng_rel(N0=N0, E0=E0, N1=N, E1=E, head0=head0) / (2*np.pi)

            # heading intersection angle
            C_TS = head_inter(head_OS=head0, head_TS=headTS) / (2*np.pi)

            # longitudinal speed
            u_TS = TS.nu[0]

            # euclidean distance
            ED_TS = ED(N0=N0, E0=E0, N1=N, E1=E, sqrt=True) / self.E_max

            # COLREG mode
            sigma_TS = self._get_COLREG_situation(OS=self.OS, TS=TS)

            # TCPA
            TCPA_TS = tcpa(NOS=N0, EOS=E0, NTS=N, ETS=E, chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS) / 60

            # store it
            if 0 <= TCPA_TS <= 1:
                state_TSs.append([bng_rel_TS, C_TS, u_TS, ED_TS, sigma_TS, TCPA_TS])
        
        # create zero state if no TS is close according to TCPA
        if len(state_TSs) == 0:
            state_TSs = np.array([0.0] * 6 * self.N_TSs)

        # otherwise sort according to descending TCPA_TS
        else:
            state_TSs = np.array(sorted(state_TSs, key=lambda x: x[-1], reverse=True))
            state_TSs = state_TSs.flatten(order="C")

            # padd zeroes at the left side to guarantee state size is always identical
            state_TSs = np.pad(state_TSs, (self.N_TSs * 6 - len(state_TSs), 0), 'constant').astype(np.float32)


        #------------------------------- combine state ------------------------------
        self.state = np.concatenate([state_OS, state_goal, state_TSs])


    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done, {}."""

        # perform control action
        self.OS._control(a)

        # update resulting tau
        self.OS._set_tau()

        # update agent dynamics
        self.OS._upd_dynamics(mirrow=False)

        # update environmental dynamics, e.g., other vessels
        [TS._upd_dynamics(mirrow=True) for TS in self.TSs]

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t
        
        return self.state, self.r, d, {}


    def _calculate_reward(self, w_dist=1, w_head=0.5, w_coll=1, w_map=1):
        """Returns reward of the current state."""

        N0, E0, psi = self.OS.eta

        # --------------------------- Path planning reward (Xu et al. 2022) -----------------------------

        # 1. Distance reward
        OS_goal_ED = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])
        r_dist = - OS_goal_ED / self.OS_goal_ED_init

        # 2. Heading reward
        r_head = -angle_to_pi(bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=psi)) / np.pi


        # ----------------------------------- Collision reward ------------------------------------------
        r_coll = 0

        for TS in self.TSs:

            EDsq_TS = ED(N0=N0, E0=E0, N1=TS.eta[0], E1=TS.eta[1], sqrt=False)

            # Basic Gaussian reward
            r_coll -= np.exp(-0.5 * EDsq_TS / self.r_coll_sigma**2)

            # Explicit collision penalty
            r_coll -= 10 if EDsq_TS < self.safety_dist**2 else 0


        # ----------------------------------- Leave-the-map reward --------------------------------------
        r_map = -10 if self.OS._is_off_map() else 0


        # -------------------------------------- Overall reward -----------------------------------------
        self.r_dist = r_dist
        self.r_head = r_head
        self.r_coll = r_coll
        self.r_map  = r_map
        self.r = w_dist * r_dist + w_head * r_head + w_coll * r_coll + w_map * r_map


    def _done(self):
        """Returns boolean flag whether episode is over."""

        # goal reached
        OS_goal_ED = ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=self.goal["N"], E1=self.goal["E"])
        if OS_goal_ED <= 2.5:
            return True

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True

        return False


    def _get_COLREG_situation(self, OS, TS, distance=np.inf):
        """Determines the COLREG situation from the perspective of the OS. Follows Xu et al. (2020, Ocean Engineering).

        Args:
            OS (CyberShip):    own vessel with attributes eta, nu
            TS (CyberShip):    target vessel with attributes eta, nu
            distance (float):  in m, minimum distance to constitute non-zero situation

        Returns:
            0  -  no conflict situation
            1  -  head-on
            2  -  starboard crossing
            3  -  portside crossing
            4  -  overtaking
        """

        # quick access
        NOS, EOS, psi_OS = OS.eta
        NTS, ETS, psi_TS = TS.eta

        # check whether TS is too far away
        if ED(N0=NOS, E0=EOS, N1=NTS, E1=ETS) > distance:
            return 0

        # relative bearing from OS to TS
        bng_OS = bng_rel(N0=NOS, E0=EOS, N1=NTS, E1=ETS, head0=psi_OS)

        # relative bearing from TS to OS
        bng_TS = bng_rel(N0=NTS, E0=ETS, N1=NOS, E1=EOS, head0=psi_TS)

        # get overall speeds and sideslip angles
        V_OS = OS._get_V()
        V_TS = TS._get_V()

        side_OS = OS._get_beta()
        side_TS = TS._get_beta()

        # intersection angle
        C_T = head_inter(head_OS=psi_OS, head_TS=psi_TS)

        # intersection angle under consideration of sideslip
        C_T_side = head_inter(head_OS = psi_OS + side_OS, head_TS = psi_TS + side_TS)

        #-------------------------------------------------------------------------------------------------------
        # Note: For Head-on, starboard crossing, and portside crossing, we do not care about the sideslip angle.
        #       The latter comes only into play for checking the overall speed of USVs in overtaking.
        #-------------------------------------------------------------------------------------------------------

        # COLREG 1: Head-on
        if np.abs(rtd(angle_to_pi(bng_OS))) <= 5 and np.abs(rtd(angle_to_pi(bng_TS))) <= 5:
            return 1
        
        # COLREG 2: Starboard crossing
        if 5 <= rtd(bng_OS) <= 112.5 and 185 <= rtd(C_T) <= 292.5:
            return 2

        # COLREG 3: Portside crossing
        if 247.5 <= rtd(bng_OS) <= 355 and 67.5 <= rtd(C_T) <= 175:
            return 3

        # COLREG 4: Overtaking
        if 112.5 <= rtd(bng_TS) <= 247.5 and V_OS * np.cos(C_T_side) > V_TS:
            return 4

        # COLREG 0: nothing
        return 0


    def __str__(self) -> str:
        ste = f"Step: {self.step_cnt}"
        pos = f"N: {np.round(self.OS.eta[0], 3)}, E: {np.round(self.OS.eta[1], 3)}, " + r"$\psi$: " + f"{np.round(rtd(self.OS.eta[2]), 3)}°"
        vel = f"u: {np.round(self.OS.nu[0], 3)}, v: {np.round(self.OS.nu[1], 3)}, r: {np.round(self.OS.nu[2], 3)}"
        return ste + "\n" + pos + "\n" + vel


    def _get_rect(self, E, N, width, length, heading, **kwargs):
        """Returns a patches.rectangle object. heading in rad."""

        # quick access
        x = E - width/2
        y = N - length/2, 
        cx = E
        cy = N
        heading = -heading   # negate since our heading is defined clockwise, contrary to plt rotations

        # translate point to origin
        tempX = x - cx
        tempY = y - cy

        # apply rotation
        rotatedX = tempX * np.cos(heading) - tempY * np.sin(heading)
        rotatedY = tempX * np.sin(heading) + tempY * np.cos(heading)

        # translate back
        E0 = rotatedX + cx
        N0 = rotatedY + cy

        # create rect
        return patches.Rectangle((E0, N0), width, length, rtd(heading), **kwargs)


    def _plot_jet(self, axis, E, N, l, angle, **kwargs):
        """Adds a line to an axis (plt-object) originating at (E,N), having a given length l, 
           and following the angle (in rad)."""

        # transform angle in [0, 2pi)
        angle = angle_to_2pi(angle)

        # 1. Quadrant
        if angle <= np.pi/2:
            E1 = E + np.sin(angle) * l
            N1 = N + np.cos(angle) * l
        
        # 2. Quadrant
        elif 3/2 *np.pi < angle <= 2*np.pi:
            angle = 2*np.pi - angle

            E1 = E - np.sin(angle) * l
            N1 = N + np.cos(angle) * l

        # 3. Quadrant
        elif np.pi < angle <= 3/2*np.pi:
            angle -= np.pi

            E1 = E - np.sin(angle) * l
            N1 = N - np.cos(angle) * l

        # 4. Quadrant
        elif np.pi/2 < angle <= np.pi:
            angle = np.pi - angle

            E1 = E + np.sin(angle) * l
            N1 = N - np.cos(angle) * l
        
        # draw on axis
        axis.plot([E, E1], [N, N1], **kwargs)
        return axis


    def render(self):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 1 == 0: 

            # check whether figure has been initialized
            if len(plt.get_fignums()) == 0:
                self.fig = plt.figure(figsize=(10, 7))
                self.gs  = self.fig.add_gridspec(2, 2)
                self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                self.ax1 = self.fig.add_subplot(self.gs[0, 1]) # reward
                self.ax2 = self.fig.add_subplot(self.gs[1, 0]) # state
                self.ax3 = self.fig.add_subplot(self.gs[1, 1]) # action
                plt.ion()
                plt.show()
            
            # ------------------------------ ship movement --------------------------------
            # clear prior axes, set limits and add labels and title
            self.ax0.clear()
            self.ax0.set_xlim(-5, self.E_max + 5)
            self.ax0.set_ylim(-5, self.N_max + 5)
            self.ax0.set_xlabel("East")
            self.ax0.set_ylabel("North")

            # set OS
            N0, E0, head0 = self.OS.eta          # N, E, heading
            chiOS = head0 + self.OS._get_beta()  # course angle (heading + sideslip)
            VOS = self.OS._get_V()               # aggregated velocity
            
            self.ax0.text(-2, self.N_max - 7, self.__str__(), fontsize=8)
            
            rect = self._get_rect(E = E0, N = N0, width = self.OS.width, length = self.OS.length, heading = head0,
                                  linewidth=1, edgecolor='red', facecolor='none')
            self.ax0.add_patch(rect)

            # add jets according to COLREGS
            for COLREG_deg in [5, 112.5, 247.5, 355]:
                self.ax0 = self._plot_jet(axis = self.ax0, E=E0, N=N0, l = self.jet_length, 
                                          angle = head0 + dtr(COLREG_deg), color='red', alpha=0.3)

            for COLREG_deg in [67.5, 175, 185, 292.5]:
                self.ax0 = self._plot_jet(axis = self.ax0, E=E0, N=N0, l = self.jet_length, 
                                          angle = head0 + dtr(COLREG_deg), color='gray', alpha=0.3)

            # set simplified ship domain
            circ = patches.Circle((E0, N0), radius=self.jet_length, edgecolor='red', facecolor='none', alpha=0.3)
            self.ax0.add_patch(circ)

            # set goal (stored as NE)
            self.ax0.scatter(self.goal["E"], self.goal["N"], color="blue")
            self.ax0.text(self.goal["E"], self.goal["N"] + 2,
                          r"$\psi_g$" + f": {np.round(rtd(bng_rel(N0=N0, E0=E0, N1=self.goal['N'], E1=self.goal['E'], head0=head0)),3)}°",
                          horizontalalignment='center', verticalalignment='center', color='blue')

            # set other vessels
            for TS in self.TSs:
                N, E, headTS = TS.eta               # N, E, heading
                chiTS = headTS + TS._get_beta()     # course angle (heading + sideslip)
                VTS = TS._get_V()                   # aggregated velocity

                # determine color according to COLREG scenario
                COLREG = self._get_COLREG_situation(OS=self.OS, TS=TS, distance=10000)
                col = COLREG_COLORS[COLREG]

                # vessel
                rect = self._get_rect(E = E, N = N, width = TS.width, length = TS.length, heading = headTS,
                                      linewidth=1, edgecolor=col, facecolor='none', label=COLREG_NAMES[COLREG])
                self.ax0.add_patch(rect)

                # add two jets according to COLREGS
                for COLREG_deg in [5, 355]:
                    self.ax0 = self._plot_jet(axis = self.ax0, E=E, N=N, l = self.jet_length, 
                                              angle = headTS + dtr(COLREG_deg), color=col, alpha=0.75)

                # TCPA
                TCPA_TS = tcpa(NOS=N0, EOS=E0, NTS=N, ETS=E, chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)
                self.ax0.text(E, N + 2, f"TCPA: {np.round(TCPA_TS, 2)}",
                              horizontalalignment='center', verticalalignment='center', color=col)

            # set legend for COLREGS
            self.ax0.legend(handles=[patches.Patch(color=COLREG_COLORS[i], label=COLREG_NAMES[i]) for i in range(5)], fontsize=8,
                            loc='lower center', bbox_to_anchor=(0.75, 1.0), fancybox=False, shadow=False, ncol=5).get_frame().set_linewidth(0.0)


            # ------------------------------ reward plot --------------------------------
            if self.step_cnt == 0:
                self.ax1.clear()
                self.ax1.old_time = 0
                self.ax1.old_r_head = 0
                self.ax1.old_r_dist = 0
                self.ax1.old_r_coll = 0

            self.ax1.set_xlim(0, self._max_episode_steps)
            #self.ax1.set_ylim(-1.25, 0.1)
            self.ax1.set_xlabel("Timestep in episode")
            self.ax1.set_ylabel("Reward")

            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_head, self.r_head], color = "blue", label="Heading")
            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_dist, self.r_dist], color = "black", label="Distance")
            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_coll, self.r_coll], color = "green", label="Collision")
            
            if self.step_cnt == 0:
                self.ax1.legend()

            self.ax1.old_time = self.step_cnt
            self.ax1.old_r_head = self.r_head
            self.ax1.old_r_dist = self.r_dist
            self.ax1.old_r_coll = self.r_coll


            # ------------------------------ state plot --------------------------------
            if self.step_cnt == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_state = self.state_init

            self.ax2.set_xlim(0, self._max_episode_steps)
            #self.ax2.set_ylim(-1, 1.1)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("State information")

            for i in range(8):
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_state[i], self.state[i]], 
                               color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i], 
                               label=self.state_names[i])          
            if self.step_cnt == 0:
                self.ax2.legend()

            self.ax2.old_time = self.step_cnt
            self.ax2.old_state = self.state


            # ------------------------------ action plot --------------------------------
            if self.step_cnt == 0:
                self.ax3.clear()
                self.ax3_twin = self.ax3.twinx()
                self.ax3.old_time = 0
                self.ax3.old_action = 0
                self.ax3.old_rud_angle = 0
                self.ax3.old_tau_cnt_r = 0

            self.ax3.set_xlim(0, self._max_episode_steps)
            self.ax3.set_ylim(-0.1, self.action_space.n - 1 + 0.1)
            self.ax3.set_yticks(range(self.action_space.n))
            self.ax3.set_yticklabels(range(self.action_space.n))
            self.ax3.set_xlabel("Timestep in episode")
            self.ax3.set_ylabel("Action (discrete)")

            self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_action, self.OS.action], color="black", alpha=0.5)

            # add rudder angle plot
            if self.cnt_approach == "rps_angle":
                self.ax3_twin.plot([self.ax3.old_time, self.step_cnt], [rtd(self.ax3.old_rud_angle), rtd(self.OS.rud_angle)], color="blue")
                self.ax3_twin.set_ylim(-rtd(self.OS.rud_angle_max) - 5, rtd(self.OS.rud_angle_max) + 5)
                self.ax3_twin.set_yticks(range(-int(rtd(self.OS.rud_angle_max)), int(rtd(self.OS.rud_angle_max)) + 5, 5))
                self.ax3_twin.set_yticklabels(range(-int(rtd(self.OS.rud_angle_max)), int(rtd(self.OS.rud_angle_max)) + 5, 5))
                self.ax3_twin.set_ylabel("Rudder angle (in °, blue)")
                self.ax3.old_rud_angle = self.OS.rud_angle

            elif self.cnt_approach == "tau":
                self.ax3_twin.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_tau_cnt_r, self.OS.tau_cnt_r], color="blue")
                self.ax3_twin.set_ylim(-self.OS.tau_cnt_r_max - 0.1, self.OS.tau_cnt_r_max + 0.1)
                self.ax3_twin.set_yticks(np.linspace(-100 * self.OS.tau_cnt_r_max, 100 * self.OS.tau_cnt_r_max, 9)/100)
                self.ax3_twin.set_yticklabels(np.linspace(-100 * self.OS.tau_cnt_r_max, 100 * self.OS.tau_cnt_r_max, 9)/100)
                self.ax3_twin.set_ylabel("Tau_r (in Nm, blue)")
                self.ax3.old_tau_cnt_r = self.OS.tau_cnt_r

            self.ax3.old_time = self.step_cnt
            self.ax3.old_action = self.OS.action

            plt.pause(0.001)
