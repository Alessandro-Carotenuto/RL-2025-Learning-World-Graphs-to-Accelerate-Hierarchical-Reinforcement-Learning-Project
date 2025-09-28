import random

from typing import List, Tuple, Dict, Optional

#-----------------------------------------------------------------------------
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

#-----------------------------------------------------------------------------


def sample_goal_position(env, agent_pos: Tuple[int, int], max_distance: int = 6) -> Tuple[int, int]:
    """Sample a reachable goal position within max_distance from agent."""
    
    try:
        reachable_positions = env.BFS_all_reachable(agent_pos)
    except Exception as e:
        print(f"    BFS failed: {e}, using fallback goal")
        # Simple fallback without boundary validation
        return (agent_pos[0] + 1, agent_pos[1])
    
    # Filter by maximum distance AND exclude agent's current position
    nearby_goals = []
    for pos in reachable_positions:
        if (manhattan_distance(agent_pos, pos) <= max_distance and 
            pos != agent_pos):
            nearby_goals.append(pos)
    
    if not nearby_goals:
        # Fallback: get any reachable position != agent_pos
        other_positions = [pos for pos in reachable_positions if pos != agent_pos]
        if other_positions:
            return random.choice(other_positions)
        else:
            # Last resort: return a nearby position (let environment handle if invalid)
            return (agent_pos[0] + 1, agent_pos[1])
    
    return random.choice(nearby_goals)
#-----------------------------------------------------------------------------