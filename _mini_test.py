import sys, math, torch
sys.path.insert(0, 'C:/Users/alex1/Documents/Cloned Repositories/RL-2025-Learning-World-Graphs-to-Accelerate-Hierarchical-Reinforcement-Learning-Project')

from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker, HierarchicalTrainer
from utils.graph_manager import GraphManager
from wrappers.minigrid_wrapper import MinigridWrapper, EnvModes, EnvSizes
from local_networks.policy_networks import GoalConditionedPolicy

env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL, max_steps=200)
env.reset()
env.phase = 2

valid_cells = {(x,y) for x in range(env.width) for y in range(env.height) if env._is_traversable(env.grid.get(x,y))}

pivots_list = sorted(valid_cells)
pivotal_states = pivots_list[::max(1, len(pivots_list)//6)][:6]
print(f'Pivots: {pivotal_states}')

world_graph = GraphManager()
for ps in pivotal_states:
    world_graph.add_node(ps)
for i in range(len(pivotal_states)-1):
    a, b = pivotal_states[i], pivotal_states[i+1]
    world_graph.add_edge(a, b, weight=1.0, path=[a, b])
    world_graph.add_edge(b, a, weight=1.0, path=[b, a])
print(f'Graph: {len(world_graph.edges)} edges')

manager = HierarchicalManager(pivotal_states, neighborhood_size=2, lr=5e-4, horizon=5, device='cpu')
goal_policy = GoalConditionedPolicy(maze_size=EnvSizes.SMALL.value, device='cpu')
worker = HierarchicalWorker(world_graph, pivotal_states, lr=1e-4, goal_policy=goal_policy,
                            maze_size=EnvSizes.SMALL.value, neighborhood_size=2, device='cpu')

worker.valid_cells = valid_cells
worker.build_wall_mask(env.width, env.height)

trainer = HierarchicalTrainer(
    env=env, manager=manager, worker=worker,
    horizon=5, goal_timeout=20,
    workershaping=False, managershaping=False,
    instant_traversal=False,
    traversal_shaping_weight=0.0
)

print('Running 5 Phase 3 episodes...')
for ep in range(5):
    stats = trainer.train_episode(max_steps=150, full_breakdown_every=999)
    r = stats['episode_reward']
    b = stats['balls_collected']
    s = stats['episode_steps']
    print(f'  Ep {ep+1}: reward={r:.2f}, balls={b}, steps={s}')

print('Phase 3 training: OK')
