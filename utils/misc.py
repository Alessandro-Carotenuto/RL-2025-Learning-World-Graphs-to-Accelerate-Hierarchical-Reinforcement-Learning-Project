import random
import torch

#----------------------------------------------------------------------------#
#                                UTILITIES                                   #
#----------------------------------------------------------------------------#
def resolve_device(config=None):
    """Validate and print the active device. If config is given, mutates config['device'] in place.
    Always returns the resolved device string."""
    if config is None:
        config = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    if config['device'] == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            config['device'] = 'cpu'
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return config['device']

#-----------------------------------------------------------------------------
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

#-----------------------------------------------------------------------------

def sample_goal_position(env, start_pos, max_distance=20):
    """Sample a reachable goal within max_distance (Manhattan) using BFS."""
    try:
        reachable_positions = env.BFS_all_reachable(start_pos)
    except Exception as e:
        print(f"    BFS failed: {e}, using fallback goal")
        return (start_pos[0] + 1, start_pos[1])

    candidates = [
        pos for pos in reachable_positions
        if pos != start_pos and manhattan_distance(start_pos, pos) <= max_distance
    ]

    if not candidates:
        fallback = [pos for pos in reachable_positions if pos != start_pos]
        return random.choice(fallback) if fallback else (start_pos[0] + 1, start_pos[1])

    return random.choice(candidates)

#-----------------------------------------------------------------------------

def _walk_away_from_spawn(env, spawn: tuple, walk_length: int = 400, bias: float = 0.7) -> tuple:
    """
    Random walk biased toward moving away from spawn.
    At each step: if move_forward increases manhattan distance from spawn,
    take it with probability `bias`; otherwise pick a random action.
    Returns the position reached. No map knowledge required — only env.step().
    """
    env.reset()
    current_pos = tuple(env.agent_pos)
    dir_delta = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    for _ in range(walk_length):
        dx, dy = dir_delta[env.agent_dir]
        fwd = (current_pos[0] + dx, current_pos[1] + dy)
        if (manhattan_distance(fwd, spawn) > manhattan_distance(current_pos, spawn)
                and random.random() < bias):
            action = 2  # move_forward (away from spawn)
        else:
            action = random.choice([0, 1, 2])
        try:
            _, _, term, trunc, _ = env.step(action)
            current_pos = tuple(env.agent_pos)
            if term or trunc:
                break
        except Exception:
            break
    return current_pos