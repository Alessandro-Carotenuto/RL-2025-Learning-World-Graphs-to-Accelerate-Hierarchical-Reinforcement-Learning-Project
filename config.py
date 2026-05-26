import math
import torch
from wrappers.minigrid_wrapper import EnvSizes

# ========================
# USER CONFIGURATION
# ========================

_max_steps = 2000

externalconfig = {
    # --- env ---
    'maze_size':             EnvSizes.MEDIUM,
    'num_balls':             5,
    'max_steps_per_episode': _max_steps,
    'device':                'cuda' if torch.cuda.is_available() else 'cpu',

    # --- phase 1 ---
    'phase1_iterations':     3,
    'goal_policy_lr':        5e-3,
    'vae_mu0':               9.0,
    'pivotal_spread_alpha':  0.02,
    'explore_top_fraction':  0.20,
    'diversity_walk_number': 30,
    'walk_length':           400,
    'walk_bias':             0.70,
    'walk_episodes':         10,
    'graph_walk_length':     50,
    'graph_num_attempts':    150,
    'convergence_threshold': 0.01,

    # --- phase 3 (integration training) ---
    'phase3_episodes':          50,
    'manager_horizon':          10,
    'neighborhood_size':        math.ceil(EnvSizes.MEDIUM.value / 8),
    'manager_lr':               5e-4,
    'worker_lr':                1e-4,
    'goal_timeout':             200,  # max steps on a goal before forcing replanning
    'traversal_shaping_weight': 2.0,
    'narrow_goal_timeout':      50,   # steps in NARROW_GOAL without progress → back to FINDING

    # --- manager pretrain ---
    'manager_wide_horizons_per_episode':   20,
    'manager_wide_ppo_epochs':             4,
    'manager_narrow_pretrain_episodes':    5000,  # 0 = skip
    'manager_narrow_horizons_per_episode': 20,
    'manager_narrow_ppo_epochs':           1,
    'manager_narrow_use_per':              True,
    'manager_narrow_per_warmup':           250,
    'manager_narrow_per_replay_freq':      50,
    'manager_narrow_per_buffer_size':      5000,
    'manager_narrow_per_batch_size':       32,
    'manager_narrow_per_alpha':            0.6,
    'manager_narrow_per_beta_start':       0.4,
    'manager_narrow_per_lr_factor':        1,
    'manager_narrow_per_entropy_coef':     0.05,
    'manager_narrow_entropy_start':        0.3,   # entropy_coef at ep 0 (prevents early collapse)
    'manager_narrow_entropy_end':          0.001, # entropy_coef at final ep

    # curriculum_manager_pretrain=True  → use phases below
    # curriculum_manager_pretrain=False → single flat phase with manager_wide_pretrain_episodes
    'curriculum_manager_pretrain':       False,
    'manager_wide_pretrain_episodes':    1,  # used only when curriculum_manager_pretrain=False
    'manager_wide_pretrain_r_offset':    2,  # r_offset for non-curriculum mode

    # curriculum phases (r_offset added to neighborhood_size)
    # each phase: {episodes, r_offset, lr_start_factor, lr_end_factor,
    #              entropy_coef, entropy_warmup_eps, entropy_warmup_coef}
    'manager_wide_pretrain_phases': [
        {'episodes': 30000, 'r_offset': 2, 'lr_start_factor': 1.0, 'lr_end_factor': 0.1,
         'entropy_coef': 0.001, 'entropy_warmup_eps': 0,   'entropy_warmup_coef': 0.001},
        {'episodes': 20000, 'r_offset': 1, 'lr_start_factor': 1.0, 'lr_end_factor': 0.1,
         'entropy_coef': 0.001, 'entropy_warmup_eps': 500, 'entropy_warmup_coef': 0.005},
        {'episodes': 10000, 'r_offset': 0, 'lr_start_factor': 1.2, 'lr_end_factor': 0.1,
         'entropy_coef': 0.001, 'entropy_warmup_eps': 500, 'entropy_warmup_coef': 0.005},
    ],

    # --- PPO ---
    'ppo_epochs':   4,
    'ppo_clip_eps': 0.2,
    'gae_lambda':   0.95,

    # --- diagnostics ---
    'diagnostic_interval':   100000,
    'diagnostic_checkstart': False,

    # --- worker pretrain curriculum ---
    'threshold_r1': 0.95,  'threshold_r2': 0.85,  'threshold_r3': 0.85,
    'repeat_r1':    3,      'repeat_r2':    2,      'repeat_r3':    2,
    'i_steps_r1':   30,     'i_steps_r2':   80,     'i_steps_r3':   200,
    'dropby_r1':    5,      'dropby_r2':    20,     'dropby_r3':    40,
    'f_steps_r1':   5,      'f_steps_r2':   10,     'f_steps_r3':   20,
    'substage_cap_r1': 5000, 'substage_cap_r2': 1000, 'substage_cap_r3': 1000,

    # --- post-pretrain Worker edge refining ---
    'worker_edge_refine':      True,
    'worker_graph_attempts':   50,   # attempts per pivot
    'worker_graph_max_steps':  100,  # steps per attempt

    # --- worker pretrain PER ---
    'worker_per_use':            False,
    'worker_per_buffer_size':    500,
    'worker_per_warmup':         250,
    'worker_per_replay_freq':    250,
    'worker_per_batch_episodes': 50,
    'worker_per_alpha':          0.6,
    'worker_per_beta_start':     0.4,
    'worker_per_lr_factor':      0.5,
}
