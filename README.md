# Learning World Graphs to Accelerate Hierarchical Reinforcement Learning

<p align="center">
  <img src="readme assets/reinforce.gif" alt="HRL agent navigating a 24×24 maze to collect all five balls" width="75%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+" height="28" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" height="28" />
  <img src="https://img.shields.io/badge/MiniGrid-Gymnasium-0081A7?logo=openai&logoColor=white" alt="MiniGrid" height="28" />
  <img src="https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white" alt="CUDA optional" height="28" />
  <img src="https://img.shields.io/badge/Trained_on-Kaggle-20BEFF?logo=kaggle&logoColor=white" alt="Trained on Kaggle" height="28" />
  <img src="https://img.shields.io/badge/Architecture-HRL%20%2B%20World%20Graph-8A2BE2" alt="HRL + World Graph" height="28" />
</p>

**Hierarchical reinforcement learning on a 24×24 maze: a VAE discovers structural bottleneck states and builds a World Graph; a Manager + Worker system uses that graph to efficiently collect all five balls per episode.**

*Alessandro Carotenuto*

Reimagined and rebuilt from scratch, inspired by:

> Wenling Shang, Alex Trott, Stephan Zheng, Caiming Xiong, Richard Socher — *Learning World Graphs to Accelerate Hierarchical Reinforcement Learning*, arXiv:1907.00664 (2019)

The original paper proposes a general two-stage HRL framework tested across multiple maze tasks. This project reconstructs and substantially extends the core ideas for the **MultiGoal** task, with an original Wide-then-Narrow goal-selection scheme, a three-state Worker FSM, and a full pretraining pipeline developed from scratch.

---

## Table of Contents

- [How it works](#how-it-works)
- [Typical Run](#typical-run)
- [Architecture](#architecture)
- [Training pipeline](#training-pipeline)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Reference](#reference)

---

## How it works

The system operates in three phases:

**Phase 1 — World Graph Discovery.** A recurrent VAE with Hard-Kumaraswamy binary latent variables trains jointly with a curiosity-driven Goal-Conditioned Policy (GCP). As both explore the maze, the VAE's prior means collapse onto structural bottleneck cells — *pivotal states* — that represent the skeleton of the maze. Random walks between discovered pivotal states build the directed weighted *World Graph*.

**Phase 2 — Pretraining.** Each component is pretrained in isolation before integration: the Worker and GCP on short-to-long navigation curricula, the Manager wide head on pivot-to-ball proximity rewards, the Manager narrow head on precise ball placement with an oracle wide goal, and both heads jointly.

**Phase 3 — Integration Training.** The Manager selects a (wide, narrow) goal pair and holds it for up to `goal_timeout` steps. The Worker's three-state FSM executes the plan: navigating to a nearby pivot, following the graph to the wide goal, then using MLP navigation for the narrow goal. Goals are replanned on success, ball collection, timeout, or spinning detection.

The World Graph for the MEDIUM maze (checkpoint `phase1_checkpoint_MEDIUM.pt`) contains **58 pivotal states** and **121 directed edges**.

---

## Typical Run

The plots below show typical training dynamics across the full pipeline. Individual runs vary in convergence speed due to the stochasticity of Phase 1 graph discovery and random weight initialisation, but the overall shape and trends are consistent across runs.

### Phase 1 — World Graph Discovery

<!-- TODO: add Phase 1 diagnostics plot -->

<p align="center">
  <img src="readme assets/WG EXAMPLE.png" alt="World Graph overlaid on the 24×24 maze: 58 pivotal states and 121 directed edges" width="70%" />
</p>

### Pretrain — Worker

<p align="center">
  <img src="readme assets/pretrain_worker_diagnostics.png" alt="Worker pretraining diagnostics" width="70%" />
</p>

### Pretrain — Manager Wide

<p align="center">
  <img src="readme assets/pretrain_manager_wide_diagnostics.png" alt="Manager wide head pretraining diagnostics" width="70%" />
</p>

### Pretrain — Manager Narrow

<p align="center">
  <img src="readme assets/pretrain_manager_narrow_diagnostics.png" alt="Manager narrow head pretraining diagnostics" width="70%" />
</p>

### Pretrain — Manager Joint

<p align="center">
  <img src="readme assets/pretrain_manager_joint_diagnostics.png" alt="Manager joint pretraining diagnostics" width="70%" />
</p>

### Phase 3 — Integration Training

<p align="center">
  <img src="readme assets/diagnostics_sizeMEDIUM_h10_n3_ep5000.png" alt="Phase 3 integration training diagnostics" width="70%" />
</p>

---

## Architecture

### Manager (`HierarchicalManager`)

- A2C-LSTM, input: `[state_x, state_y, prev_wide_goal_x, prev_wide_goal_y]`
- **Wide head** — Categorical over N pivotal states; selects a long-range waypoint
- **Narrow head** — Categorical over `2r(r+1)` cells in a Manhattan diamond of radius `r` around the wide goal; refines to a precise target cell
- Combined log-probability: `wide_log_prob + narrow_log_prob`
- Updated once per episode (end-of-episode A2C) to avoid gradient version conflicts
- Hyperparameters: lr = 5×10⁻⁴, entropy_coef = 1×10⁻⁵, value_coef = 0.05

With `r = 3` (MEDIUM maze): 2×3×4 = **24 narrow cells** per wide goal.

### Worker (`HierarchicalWorker`)

- A2C-MLP, **17 inputs**: 15 from an asymmetric agent-relative wall patch + 2 goal coordinates in the agent frame
- **Wall patch**: 3×5 slice of the pre-padded `wall_mask`, rotated agent-relative, rows in front only (no backward action)
- **Goal encoding**: `(goal_fwd, goal_rgt)` in agent frame via CW rotation by `dir × 90°`
- 3 actions: turn left, turn right, move forward
- Hyperparameters: lr = 1×10⁻⁴

#### Worker FSM

```
FINDING ──(on pivot, path exists)──► TRAVERSAL ──(complete)──► NARROW_GOAL
   └──(already at wide_goal)────────────────────────────────► NARROW_GOAL

TRAVERSAL ──(desync / missing edge)──► FINDING
```

| State | Behaviour |
|-------|-----------|
| **FINDING** | MLP navigates to nearest non-blacklisted pivot; pivot-local timeout `finding_local_timeout = 30` steps; on timeout the pivot is blacklisted and the next nearest is tried |
| **TRAVERSAL** | Deterministic edge-following from the World Graph (Dijkstra path via `generate_actions_from_path`); no MLP |
| **NARROW_GOAL** | MLP navigates to the narrow goal; capped indirectly by Manager's `goal_timeout` |

Spinning detection (`report_step`) counts consecutive steps with no position change across all FSM states; after `spinning_timeout` steps the Manager replans.

### World Graph (`GraphManager`)

- Directed weighted graph over pivotal states
- Edges built via GCP-guided random walks between pivot pairs
- Shortest paths: Dijkstra, BFS fallback for disconnected nodes
- Refined post-pretrain by the Worker (`refine_edges_with_worker`)

---

## Training pipeline

### Phase 1 — World Graph Discovery

Alternating loop between:

1. **VAE training** on collected trajectories — Hard-Kumaraswamy latents identify pivotal states via prior-mean collapse
2. **GCP training** with curiosity reward from VAE reconstruction error
3. **Random walks** from current pivotal states for broad coverage
4. **Edge building** via GCP-guided walks between pivot pairs

Key hyperparameter: `vae_mu0` controls the L0 target (i.e. the density of pivotal states). Higher `mu0` → more pivotal states → denser graph.

### Phase 2 — Pretraining

| Step | Function | Notes |
|------|----------|-------|
| Worker + GCP pretrain | `run_worker_pretrain` | Curriculum: distance r = 1 → 3 |
| Edge refinement | `refine_edges_with_worker` | Worker replaces VAE-generated edge paths |
| Manager wide pretrain | `run_manager_wide_pretrain` | Pivot-to-ball proximity; `pretrain_r = r + 1` for higher hit rate |
| Manager narrow pretrain | `run_manager_narrow_pretrain` | Oracle wide goal; PER replay buffer |
| Manager joint pretrain | `run_manager_joint_pretrain` | Both heads together |

### Phase 3 — Integration Training

Full HRL loop: Manager selects (wide, narrow) goal → Worker FSM executes → Manager updates once per episode. Goal persistence keeps the same goal for up to `goal_timeout` steps; the `narrow_blacklist` prevents re-selecting already-reached narrow goals within the episode.

---

## Project structure

```
.
├── main.py                          # All pipeline entry points and standalone functions
├── config.py                        # Centralised configuration (externalconfig dict)
├── requirements.txt
│
├── local_networks/
│   ├── hierarchical_system.py       # HierarchicalManager, HierarchicalWorker, HierarchicalTrainer
│   ├── policy_networks.py           # GoalConditionedPolicy (GCP)
│   └── vaesystem.py                 # VAE with Hard-Kumaraswamy latents
│
├── local_distributions/
│   └── hardkuma.py                  # HardKumaraswamy distribution (rsample, stretch-and-clamp)
│
├── wrappers/
│   └── minigrid_wrapper.py          # MiniGrid environment wrapper (MULTIGOAL mode)
│
├── utils/
│   ├── graph_manager.py             # GraphManager: Dijkstra, BFS, adjacency list
│   ├── visualization.py             # Episode rendering (MP4), pretrain diagnostic plots
│   ├── checkpoint.py                # Phase 1 checkpoint save/load, maze restoration
│   ├── statistics_buffer.py         # StatBuffer for Phase 1 metrics
│   └── misc.py                      # manhattan_distance, resolve_device, walk utilities
│
└── bufferclasses/
    └── replay_buffers.py            # PER buffers: NarrowReplayBuffer, WorkerEpisodeReplayBuffer
```

---

## Configuration

All hyperparameters live in `config.py` (`externalconfig` dict) and can be overridden at any call site via `config_overrides`:

```python
run_phase3_standalone(
    checkpoint_path='phase1_checkpoint_MEDIUM.pt',
    config_overrides={'phase3_episodes': 300, 'goal_timeout': 200},
)
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maze_size` | `MEDIUM` (24×24) | Environment grid size |
| `num_balls` | `5` | Balls to collect per episode |
| `phase1_iterations` | `3` | VAE + GCP alternation rounds |
| `vae_mu0` | `9.0` | L0 target — controls pivotal state density |
| `neighborhood_size` | `3` | Manhattan diamond radius `r` for narrow goal |
| `manager_horizon` | `10` | Worker steps per Manager decision |
| `goal_timeout` | `200` | Max steps on a goal before forced replanning |
| `phase3_episodes` | `50` | Integration training episodes |
| `worker_edge_refine` | `True` | Refine graph edges with trained Worker |
| `curriculum_manager_pretrain` | `False` | Enable curriculum in Manager pretraining |

---

## Usage

All entry points are in `main.py`. Edit the active call at the bottom and run:

```bash
python main.py
```

### Full pipeline (Phase 1 → 2 → 3)

```python
train_full_phase1_to_phase3()
```

### Phase 3 only (requires Phase 1 checkpoint)

```python
run_phase3_standalone(
    checkpoint_path='phase1_checkpoint_MEDIUM.pt',
    config_overrides=externalconfig,
    fixed_balls=True,
    phase3_animation=False,
)
```

### Worker pretraining only

```python
run_worker_pretrain_standalone(
    use_checkpoint=True,
    checkpoint_path='phase1_checkpoint_MEDIUM.pt',
    config_overrides=externalconfig,
)
```

### Manager pretraining only

```python
run_manager_wide_narrow_pretrain_standalone(
    checkpoint_path='phase1_checkpoint_MEDIUM.pt',
    config_overrides=externalconfig,
    save_path='manager_pretrained.pt',
)
```

### Evaluation (no training)

```python
# Load trained checkpoint + session and run N greedy episodes
testing_grounds(
    checkpoint_path='phase1_checkpoint_MEDIUM.pt',
    num_episodes=500,
    temperature=0.4,
    max_steps=10_000,
)
```

Reports success rate (all 5 balls collected within `max_steps`) and mean steps per episode, and saves a steps-per-episode plot with moving average.

### Render an episode as MP4

```python
render_phase3_episode_gif(
    checkpoint_path='phase1_checkpoint_MEDIUM.pt',
    filename='episode.mp4',
    fps=15,
    max_steps=500,
)
```

---

## Reference

```bibtex
@article{shang2019learning,
  title   = {Learning World Graphs to Accelerate Hierarchical Reinforcement Learning},
  author  = {Shang, Wenling and Trott, Alex and Zheng, Stephan and Xiong, Caiming and Socher, Richard},
  journal = {arXiv preprint arXiv:1907.00664},
  year    = {2019}
}
```
