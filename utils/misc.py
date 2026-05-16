import random
import torch

#-----------------------------------------------------------------------------

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