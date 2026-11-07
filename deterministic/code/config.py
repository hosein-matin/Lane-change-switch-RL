"""Single source of truth for all model, filter, reward and learning parameters.

Units: metres, seconds. Dimensional urban-arterial regime (see paper Sec. V-A).
"""
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Cfg:
    # --- time ---
    dt: float = 0.1            # Delta: sample / stay-option length [s]
    N: int = 10                # change-option length in steps -> T_m = N*dt = 1 s
    Tf: float = 15.0           # episode length [s] -> M = Tf/dt = 150 steps
    tau_d: float = 2.0         # dwell time [s]
    n_sub: int = 10            # physics substeps per sample (neighbour integration, clearance monitoring)

    # --- ego ---
    L_veh: float = 5.0         # vehicle length (all vehicles) [m]
    d0: float = 2.0            # required bumper clearance [m]
    u_lo: float = -6.0         # ego acceleration bounds [m/s^2]
    u_hi: float = 2.5
    vmax: float = 20.0         # ego speed cap [m/s]

    # --- HOCBF (sample-time anticipative interval I_q) ---
    lam_f: float = 0.6
    kap_f: float = 1.0
    lam_r: float = 0.6
    kap_r: float = 1.0
    kap_v: float = 2.0

    # --- human followers: OVM (Bando, dimensional) with saturation ---
    ovm_alpha: float = 0.8     # 1/s
    ovm_dc: float = 15.0       # m
    ovm_ell: float = 8.0       # m
    f_amax: float = 2.5        # follower accel saturation
    f_bmax: float = 7.0        # follower braking saturation (magnitude)
    # declared follower acceleration bound used by the filter:
    #  'ovm' = per-sample bound derived from the declared OVM (requires the human model)
    #  'max' = model-free worst case a_f <= f_amax (only the saturation is assumed)
    follower_bound: str = os.environ.get('FOLLOWER_BOUND', 'ovm')
    # critic ablation: hide the leaders' disturbance parameters (mean, amplitude, frequency, phase)
    critic_nodist: bool = os.environ.get('CRITIC_NODIST', '0') == '1'
    # two-lane commitment: during a change option the ego occupies both lanes (input must satisfy both
    # lanes' sampled intervals; the old follower keeps reacting to the ego)
    n_lanes: int = int(os.environ.get('N_LANES', '2'))   # number of lanes L (2 or 3)
    dual_lane: bool = os.environ.get('DUAL_LANE', '0') == '1'
    # 'intersect': per-sample intersection of both lanes' intervals (earlier variant, may need a fallback);
    # 'hold': one input, certified over the whole commitment T_m for both lanes, is held (no fallback needed)
    dual_mode: str = os.environ.get('DUAL_MODE', 'intersect')
    # leader acceleration bound used by the filter: 0 = declared per-leader bound -A|w| (generator family);
    # > 0 = generic bound -LEADER_BOUND m/s^2 for every leader (no knowledge of the scenario generator)
    leader_generic: float = float(os.environ.get('LEADER_BOUND', '0'))

    # --- reward ---
    w_d: float = 1.0           # per metre
    w_u: float = 1.0           # per (m/s^2)^2 * s
    c0: float = 2.0            # physical switching cost per initiation
    C_fail: float = 200.0      # failure charge

    # --- learning scales (frozen) ---
    rew_scale: float = 0.1     # rewards multiplied by this inside learners only

    @property
    def M(self):
        return int(round(self.Tf / self.dt))

    @property
    def Tm(self):
        return self.N * self.dt

    def as_dict(self):
        d = asdict(self)
        d.update(M=self.M, Tm=self.Tm)
        return d
