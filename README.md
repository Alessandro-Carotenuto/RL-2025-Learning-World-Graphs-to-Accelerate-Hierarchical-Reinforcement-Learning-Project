# Learning World Graphs to Accelerate Hierarchical Reinforcement Learning

A reimagination and ground-up rebuild of the ideas from:

> **Learning World Graphs to Accelerate Hierarchical Reinforcement Learning**  
> Wenling Shang, Alex Trott, Stephan Zheng, Caiming Xiong, Richard Socher  
> Salesforce Research — arXiv:1907.00664 (2019)

The original paper proposes a general two-stage HRL framework tested across multiple maze tasks. This project rebuilds and extends those ideas specifically for the **MultiGoal** task on a 24×24 maze, with substantial architectural and training modifications developed from scratch.

---

## Overview

**Stage 1 — World Graph Discovery**: A recurrent VAE with Hard-Kumaraswamy binary latent variables jointly trains with a curiosity-driven Goal-Conditioned Policy (GCP) to explore the maze and identify *pivotal states* — structural bottlenecks of the environment. These states become the nodes of a directed weighted *World Graph*, whose edges encode actionable transitions between them.

**Stage 2 — Hierarchical RL**: A hierarchical Manager + Worker system leverages the World Graph to navigate efficiently and collect balls. The Manager uses a **Wide-then-Narrow (WN)** instruction scheme: it first selects a pivotal state as a wide goal, then refines to a cell within a Manhattan diamond neighborhood as the narrow goal. The Worker uses graph traversal to reach the wide goal, then MLP navigation for the narrow goal.

---

## Architecture

### Manager (`HierarchicalManager`)
- PPO-LSTM with 7-dimensional input: `[state_x, state_y, prev_wide_goal_x, prev_wide_goal_y, nearest_ball_x, nearest_ball_y, n_remaining]`
- **Wide head**: Categorical over N pivotal states
- **Narrow head**: Categorical over `2r(r+1)` cells in a Manhattan diamond of radius `r` around the wide goal
- Input to the manager is always the **closest pivotal state to the agent** (pivot-space consistency between pretraining and inference)
- Updated once per episode via PPO

### Worker (`HierarchicalWorker`)
- PPO-MLP with 17 inputs: 15 from an asymmetric agent-relative wall patch (5×3) + 2 goal coordinates in agent frame
- 3 actions: turn left, turn right, move forward
- **3-state FSM**: `FINDING → TRAVERSAL → NARROW_GOAL`
  - **FINDING**: MLP navigates to the nearest non-blacklisted pivot
  - **TRAVERSAL**: deterministic graph-edge following (Dijkstra path)
  - **NARROW_GOAL**: MLP navigates to the narrow goal

### World Graph (`GraphManager`)
- Directed weighted graph over pivotal states
- Edges built via random walks and GCP refinement
- Shortest paths via Dijkstra with BFS fallback

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, PyTorch, MiniGrid, Gymnasium, NumPy, Matplotlib, Pillow, imageio.

CUDA is auto-detected; the system falls back to CPU if unavailable.

---

## Usage

All entry points are in `main.py`. The main pipeline:

```bash
python main.py
```

This runs the full Phase 1 → Phase 2 → Phase 3 pipeline as configured in `config.py`.

### Standalone entry points

```python
# Phase 3 only (requires a Phase 1 checkpoint)
run_phase3_standalone(checkpoint_path="phase1_checkpoint_MEDIUM.pt", config_overrides={...})

# Worker pretraining only
run_worker_pretrain_standalone(checkpoint_path=..., config_overrides={...})

# Manager wide+narrow+joint pretraining only
run_manager_wide_narrow_pretrain_standalone(checkpoint_path=..., save_path=..., config_overrides={...})
```

### Quick test (no Phase 1/2 required)

```bash
python _mini_test.py
```

Runs 5 Phase 3 episodes with a mock graph on a small environment, useful for verifying crash-free training and visualization.

---

## Configuration

All hyperparameters are in `config.py` and can be overridden at call sites via `config_overrides` dicts.

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `maze_size` | Environment size (default: MEDIUM, 24×24) |
| `num_balls` | Balls to collect per episode |
| `phase1_iterations` | VAE + GCP alternation rounds |
| `vae_mu0` | Target L0 norm (controls pivotal state density) |
| `neighborhood_size` | Manhattan diamond radius `r` for narrow goal |
| `manager_horizon` | Worker steps per manager decision |
| `goal_timeout` | Max steps on a goal before replanning |
| `manager_wide_pretrain_episodes` | Wide head pretraining episodes |
| `manager_narrow_pretrain_episodes` | Narrow head pretraining episodes |
| `manager_joint_pretrain_episodes` | Joint wide+narrow pretraining episodes |
| `phase3_episodes` | Integration training episodes |

---

## Project Structure

```
.
├── main.py                          # All pipeline entry points
├── config.py                        # Centralized configuration
├── _mini_test.py                    # Quick smoke test (Phase 3, mock graph)
├── requirements.txt
│
├── local_networks/
│   ├── hierarchical_system.py       # Manager, Worker, HierarchicalTrainer
│   ├── policy_networks.py           # GoalConditionedPolicy (GCP)
│   └── vaesystem.py                 # VAE with Hard-Kumaraswamy latents
│
├── local_distributions/
│   └── hardkuma.py                  # HardKumaraswamy distribution
│
├── wrappers/
│   └── minigrid_wrapper.py          # MiniGrid environment wrapper
│
├── utils/
│   ├── graph_manager.py             # GraphManager (Dijkstra, BFS)
│   ├── visualization.py             # Episode rendering, diagnostic plots
│   └── checkpoint.py               # Save/load Phase 1 checkpoints
│
└── bufferclasses/
    └── replay_buffers.py            # PER replay buffers (worker + narrow manager)
```

---

## Training Pipeline

### Phase 1 — World Graph Discovery
Alternating loop between:
1. **VAE training** on collected trajectories → identifies pivotal states via prior means
2. **GCP training** with curiosity reward from VAE reconstruction error
3. **Random walks** from current pivotal states for broad coverage
4. **Edge building** via GCP-guided walks between pivot pairs

### Phase 2 — Pretraining
Sequential pretraining of each component:
1. **Worker pretrain**: curriculum from short to long distances (r=1 → 3)
2. **Edge refinement**: worker refines graph edge action sequences
3. **Manager wide pretrain**: wide head learns to select pivots near balls
4. **Manager narrow pretrain**: narrow head learns to place goal on ball (oracle wide goal)
5. **Manager joint pretrain**: both heads trained together in pivot-space with composite reward

### Phase 3 — Integration Training
Full HRL loop: Manager selects (wide, narrow) goals → Worker executes FSM → Manager updates once per episode.

---

## Key Implementation Notes

- **Pivot-space state**: The manager always receives the closest pivotal state to the agent (not raw position), keeping the LSTM in-distribution with pretraining.
- **Goal persistence**: Manager keeps the same (wide, narrow) goal for up to `goal_timeout` steps; replanning triggers only on goal reached, ball collected, timeout, or spinning detection.
- **Narrow blacklist**: Reached narrow goals are blacklisted episode-wide, except cells containing an active ball.
- **Spinning detection**: Worker reports consecutive stationary steps across all FSM states; triggers manager replanning after `spinning_timeout` steps.

---

## Reference

```bibtex
@article{shang2019learning,
  title={Learning World Graphs to Accelerate Hierarchical Reinforcement Learning},
  author={Shang, Wenling and Trott, Alex and Zheng, Stephan and Xiong, Caiming and Socher, Richard},
  journal={arXiv preprint arXiv:1907.00664},
  year={2019}
}
```
