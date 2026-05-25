import math
import torch
from minigrid.core.world_object import Wall
from local_networks.vaesystem import VAESystem
from local_networks.policy_networks import GoalConditionedPolicy


def save_phase1_checkpoint(path, pivotal_states, world_graph, policy, vae_system, config, grid_state):
    """Save all Phase 1 outputs to a single file."""
    checkpoint = {
        'pivotal_states': pivotal_states,
        'world_graph': world_graph,
        'policy_state_dict': policy.state_dict(),
        'vae_state_dict': vae_system.state_dict(),
        'vae_kwargs': {
            'state_dim': vae_system.state_dim,
            'action_vocab_size': vae_system.action_vocab_size,
            'mu0': vae_system.mu0,
            'grid_size': vae_system.grid_size,
        },
        'config': config,
        'grid_state': grid_state,
    }
    torch.save(checkpoint, path)
    print(f"Phase 1 checkpoint saved to '{path}'")


def load_phase1_checkpoint(path):
    """Load Phase 1 checkpoint. Returns (pivotal_states, world_graph, policy, vae_system, config, grid_state)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    config.setdefault('max_steps_per_episode', 2000)
    config.setdefault('neighborhood_size', math.ceil(config['maze_size'].value / 4))
    config.setdefault('manager_horizon', config['max_steps_per_episode'] // 120)
    config.setdefault('manager_lr', 5e-4)
    config.setdefault('worker_lr', 1e-4)
    config.setdefault('goal_policy_lr', 5e-3)
    config.setdefault('diagnostic_interval', 10000)
    config.setdefault('diagnostic_checkstart', False)

    vae_kw = checkpoint['vae_kwargs']
    vae_system = VAESystem(
        state_dim=vae_kw['state_dim'],
        action_vocab_size=vae_kw['action_vocab_size'],
        mu0=vae_kw['mu0'],
        grid_size=vae_kw['grid_size'],
    )
    vae_system.load_state_dict(checkpoint['vae_state_dict'])
    vae_system.to(config['device'])

    policy = GoalConditionedPolicy(lr=config['goal_policy_lr'], maze_size=config['maze_size'].value, device=config['device'])
    result = policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  [checkpoint] GCP architecture mismatch (checkpoint={list(checkpoint['policy_state_dict'].keys())[:2]}…) — loaded with strict=False")

    print(f"Phase 1 checkpoint loaded from '{path}'")
    print(f"  Pivotal states: {len(checkpoint['pivotal_states'])}")
    print(f"  Graph edges:    {len(checkpoint['world_graph'].edges)}")

    return (
        checkpoint['pivotal_states'],
        checkpoint['world_graph'],
        policy,
        vae_system,
        config,
        checkpoint['grid_state'],
    )


def save_phase2_checkpoint(path, trainer, agent_start, ball_positions, phase1_checkpoint_path):
    """Save Phase 2 training state for resumption."""
    torch.save({
        'manager_state_dict': trainer.manager.state_dict(),
        'worker_state_dict': trainer.worker.state_dict(),
        'goal_policy_state_dict': trainer.worker.goal_policy.state_dict() if trainer.worker.goal_policy is not None else None,
        'global_episode_counter': trainer.global_episode_counter,
        'global_step_counter': trainer.global_step_counter,
        'diagnostic_history': trainer.diagnostic_history,
        'agent_start': agent_start,
        'ball_positions': ball_positions,
        'phase1_checkpoint_path': str(phase1_checkpoint_path),
    }, path)
    print(f"Phase 2 checkpoint saved to '{path}' (episode {trainer.global_episode_counter})")


def load_phase2_checkpoint(path, trainer):
    """Load Phase 2 checkpoint into an already-constructed trainer. Returns (agent_start, ball_positions, phase1_checkpoint_path)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    trainer.manager.load_state_dict(ckpt['manager_state_dict'])
    trainer.worker.load_state_dict(ckpt['worker_state_dict'])
    if ckpt['goal_policy_state_dict'] is not None and trainer.worker.goal_policy is not None:
        trainer.worker.goal_policy.load_state_dict(ckpt['goal_policy_state_dict'])
    trainer.global_episode_counter = ckpt['global_episode_counter']
    trainer.global_step_counter = ckpt['global_step_counter']
    trainer.diagnostic_history = ckpt['diagnostic_history']
    print(f"Phase 2 checkpoint loaded from '{path}' (resuming from episode {trainer.global_episode_counter})")
    return ckpt['agent_start'], ckpt['ball_positions'], ckpt['phase1_checkpoint_path']


def restore_maze_from_grid_state(env, grid_state):
    """Overwrite env.grid with the Phase 1 maze walls from the ASCII grid_state."""
    h = len(grid_state)
    w = len(grid_state[0]) if h > 0 else 0
    for y in range(h):
        for x in range(w):
            char = grid_state[y][x]
            if char == '#':
                env.grid.set(x, y, Wall())
                env.placeable_grid[x][y] = False
            else:
                env.grid.set(x, y, None)
                env.placeable_grid[x][y] = True

    # Agent start position is never placeable
    env.placeable_grid[env.agent_start_pos[0]][env.agent_start_pos[1]] = False
