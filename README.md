# Learning World Graphs to Accelerate Hierarchical Reinforcement Learning (original implementation)

**Independent implementation** of the paper by Shang et al. (2019)

> **Paper:** [arXiv:1907.00664](https://arxiv.org/abs/1907.00664)  
> **Note:** Unofficial implementation for educational purposes

## What This Does

Two-stage framework for hierarchical RL:
- **Phase 1:** Learn world graph (pivotal waypoints + connections) unsupervised
- **Phase 2:** Use graph for hierarchical planning (Manager + Worker)

## Installation

```bash
pip install torch gymnasium minigrid numpy matplotlib
```

Requirements: Python 3.8+, PyTorch 2.0+

## Usage

### Phase 1: World Graph Discovery
```python
from main import alternating_training_loop
from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
from local_networks.vaesystem import VAESystem
from local_networks.policy_networks import GoalConditionedPolicy
from utils.statistics_buffer import StatBuffer

env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
env.phase = 1

policy = GoalConditionedPolicy(lr=5e-3)
vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=3.0)
buffer = StatBuffer()

pivotal_states, world_graph = alternating_training_loop(
    env, policy, vae_system, buffer, max_iterations=8
)
```

### Phase 2: Hierarchical Training
```python
from local_networks.hierarchical_system import (
    HierarchicalManager, HierarchicalWorker, HierarchicalTrainer
)

manager = HierarchicalManager(pivotal_states, horizon=15)
worker = HierarchicalWorker(world_graph, pivotal_states)

manager.initialize_from_goal_policy(policy)
worker.initialize_from_goal_policy(policy)

env.phase = 2
trainer = HierarchicalTrainer(manager, worker, env, horizon=15)

for episode in range(100):
    stats = trainer.train_episode(max_steps=200)
    print(f"Episode {episode}: reward={stats['episode_reward']:.2f}")
```

### Run Tests
```bash
python main.py
```

## Structure

```
├── main.py                      # Training orchestration
├── wrappers/
│   └── minigrid_wrapper.py     # Custom MiniGrid environment
├── local_networks/
│   ├── vaesystem.py            # VAE with HardKumaraswamy
│   ├── policy_networks.py     # Goal-conditioned policy
│   └── hierarchical_system.py # Manager + Worker
├── local_distributions/
│   └── hardkuma.py             # HardKumaraswamy distribution
└── utils/
    ├── graph_manager.py        # World graph structure
    └── statistics_buffer.py    # Trajectory storage
```

## Key Components

**Phase 1:**
- Recurrent VAE with binary latents discovers pivotal states
- Goal-conditioned policy (πg) learns navigation
- Graph built from random walks + πg refinement

**Phase 2:**
- Manager: Wide-then-Narrow goal selection from graph
- Worker: Executes using graph traversal + policy
- Transfer learning from Phase 1

## Citation

```bibtex
@article{shang2019learning,
  title={Learning World Graphs to Accelerate Hierarchical Reinforcement Learning},
  author={Shang, Wenling and Trott, Alex and Zheng, Stephan and Xiong, Caiming and Socher, Richard},
  journal={arXiv preprint arXiv:1907.00664},
  year={2019}
}
```

## Acknowledgments

Original paper by Shang et al. (Salesforce Research). This is an independent implementation for educational purposes.
