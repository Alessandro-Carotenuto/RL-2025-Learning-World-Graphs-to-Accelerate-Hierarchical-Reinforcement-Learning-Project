import numpy as np
from typing import List, Tuple, Optional
from itertools import permutations

def compute_optimal_reward_for_episode(env, debug=False) -> Tuple[float, int]:
    """
    Compute the theoretical optimal reward for the current episode configuration.
    Uses BFS to find actual shortest paths through the maze (accounting for walls).
    NO MANHATTAN FALLBACK - only real pathfinding.
    
    Args:
        env: MinigridWrapper environment with balls spawned
        debug: Print debug information
        
    Returns:
        tuple: (optimal_reward, optimal_steps)
    """
    if debug:
        print(f"\n[OPTIMAL DEBUG] Starting computation")
        print(f"  env.mode: {env.mode}")
        print(f"  env.mode.value: {env.mode.value}")
        print(f"  env.phase: {env.phase}")
    
    # Only works for MULTIGOAL Phase 2
    if env.mode.value != 2 or env.phase != 2:  # EnvModes.MULTIGOAL = 2
        if debug:
            print(f"  EARLY EXIT: Wrong mode or phase")
        return 0.0, 0
    
    # Get current state
    agent_start = tuple(env.agent_pos)
    ball_positions = list(env.active_balls)
    total_balls = len(ball_positions)
    
    if debug:
        print(f"  Agent at: {agent_start}")
        print(f"  Active balls: {total_balls}")
        print(f"  Ball positions: {ball_positions}")
        print(f"  env.max_steps: {env.max_steps}")
    
    if total_balls == 0:
        if debug:
            print(f"  EARLY EXIT: No balls")
        return 0.0, 0
    
    # Greedy nearest-neighbor algorithm using ACTUAL BFS DISTANCES
    current_pos = agent_start
    unvisited_balls = ball_positions.copy()
    total_steps = 0
    collection_order = []
    
    if debug:
        print(f"\n  [GREEDY] Starting nearest-neighbor search (BFS distances ONLY)")
        print(f"  Agent position: {agent_start}")
    
    while unvisited_balls:
        # Find nearest ball using ACTUAL BFS pathfinding
        nearest_ball = None
        min_distance = float('inf')
        
        if debug:
            print(f"    Current pos: {current_pos}, Unvisited: {len(unvisited_balls)}")
        
        for ball_pos in unvisited_balls:
            try:
                # Use BFS to find actual path through maze
                # Pass agent_start as dummy player position (won't block paths)
                solvable, path = env.BFS(current_pos, ball_pos, agent_start)
                
                if solvable and path and len(path) > 1:
                    # Actual distance is path length minus 1 (path includes start and end)
                    actual_distance = len(path) - 1
                    
                    if debug:
                        print(f"      {ball_pos}: BFS distance = {actual_distance} (path length = {len(path)})")
                    
                    if actual_distance < min_distance:
                        min_distance = actual_distance
                        nearest_ball = ball_pos
                else:
                    # BFS failed - this ball is unreachable
                    if debug:
                        print(f"      {ball_pos}: UNREACHABLE (BFS returned empty/invalid path)")
                    
            except Exception as e:
                if debug:
                    print(f"      ERROR computing distance to {ball_pos}: {e}")
                # Skip this ball if BFS throws an error
                continue
        
        if nearest_ball is None:
            if debug:
                print(f"    ERROR: No reachable balls remaining!")
            # Can't reach any more balls - compute partial reward
            break
        
        # Move to nearest ball
        if debug:
            print(f"    -> Selected: {nearest_ball}, BFS distance: {min_distance}")
        total_steps += min_distance
        current_pos = nearest_ball
        unvisited_balls.remove(nearest_ball)
        collection_order.append(nearest_ball)
    
    # Calculate optimal reward based on actual reward structure
    balls_collected = len(collection_order)
    
    if balls_collected == 0:
        if debug:
            print(f"\n  [REWARD CALCULATION] No balls reachable!")
        return 0.0, 0
    
    # Step penalty
    step_penalty = total_steps * (-0.01)
    
    # Ball collection rewards
    ball_rewards = balls_collected * 1
    
    # Terminal bonus (only if all balls collected)
    terminal_bonus = 0.0
    if balls_collected == total_balls:
        time_penalty = 0.45 * (total_steps / env.max_steps)
        terminal_bonus = 0.5 - time_penalty
    
    optimal_reward = step_penalty + ball_rewards + terminal_bonus
    optimal_reward = max(0,optimal_reward)  # Ensure non-negative reward

    if debug:
        print(f"\n  [REWARD CALCULATION]")
        print(f"    Collection order: {collection_order}")
        print(f"    Balls collected: {balls_collected}/{total_balls}")
        print(f"    Total BFS steps: {total_steps}")
        print(f"    Step penalty: {step_penalty:.3f} ({total_steps} * -0.01)")
        print(f"    Ball rewards: {ball_rewards:.3f} ({balls_collected} * 0.05)")
        if balls_collected == total_balls:
            time_penalty = 0.45 * (total_steps / env.max_steps)
            print(f"    Terminal bonus: {terminal_bonus:.3f} (0.5 - {time_penalty:.3f})")
        else:
            print(f"    Terminal bonus: 0.0 (only {balls_collected}/{total_balls} balls reachable)")
        print(f"    OPTIMAL REWARD: {optimal_reward:.3f}")
        print(f"    OPTIMAL STEPS: {total_steps}")
    
    return optimal_reward, total_steps


def compute_optimal_reward_bruteforce_small(env, max_balls_for_bruteforce: int = 6) -> Tuple[float, int]:
    """
    Compute truly optimal reward using brute-force for small numbers of balls.
    Falls back to greedy for larger problems. Uses BFS for actual maze distances.
    NO MANHATTAN FALLBACK.
    
    Args:
        env: MinigridWrapper environment
        max_balls_for_bruteforce: Max balls to try all permutations (factorial complexity!)
        
    Returns:
        tuple: (optimal_reward, optimal_steps)
    """
    if env.mode.value != 2 or env.phase != 2:
        return 0.0, 0
    
    agent_start = tuple(env.agent_pos)
    ball_positions = list(env.active_balls)
    total_balls = len(ball_positions)
    
    if total_balls == 0:
        return 0.0, 0
    
    # Use greedy for large problems (factorial complexity)
    if total_balls > max_balls_for_bruteforce:
        return compute_optimal_reward_for_episode(env, debug=False)
    
    # Build distance matrix using BFS (actual pathfinding) - NO FALLBACK
    all_positions = [agent_start] + ball_positions
    n = len(all_positions)
    distance_matrix = np.full((n, n), float('inf'))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0
                continue
            
            try:
                # Use BFS to find actual path through maze
                solvable, path = env.BFS(all_positions[i], all_positions[j], agent_start)
                if solvable and path and len(path) > 1:
                    distance_matrix[i][j] = len(path) - 1
                # else: remains inf (unreachable)
            except:
                # If BFS fails, leave as inf (unreachable)
                pass
    
    # Try all permutations of ball collection order
    best_steps = float('inf')
    ball_indices = list(range(1, n))  # Exclude agent start (index 0)
    
    for perm in permutations(ball_indices):
        # Calculate total steps for this order
        current_idx = 0  # Start from agent position
        steps = 0
        valid_path = True
        
        for next_idx in perm:
            if distance_matrix[current_idx][next_idx] == float('inf'):
                # This path is impossible
                valid_path = False
                break
            steps += distance_matrix[current_idx][next_idx]
            current_idx = next_idx
        
        if valid_path and steps < best_steps:
            best_steps = steps
    
    if best_steps == float('inf'):
        # No valid path exists
        return 0.0, 0
    
    total_steps = int(best_steps)
    
    # Calculate reward
    step_penalty = total_steps * (-0.01)
    ball_rewards = total_balls * 0.05
    time_penalty = 0.45 * (total_steps / env.max_steps)
    terminal_bonus = 0.5 - time_penalty
    
    optimal_reward = step_penalty + ball_rewards + terminal_bonus
    
    return optimal_reward, total_steps


# Utility function to test optimal computation
def test_optimal_computation():
    """Test the optimal reward computation on a simple scenario."""
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 2
    env.randomgen = True
    
    obs = env.reset()
    
    print(f"Agent at: {env.agent_pos}")
    print(f"Balls at: {list(env.active_balls)}")
    print(f"Total balls: {env.total_balls}")
    
    # Compute optimal reward
    optimal_reward, optimal_steps = compute_optimal_reward_for_episode(env)
    
    print(f"\nOptimal (greedy) solution:")
    print(f"  Steps: {optimal_steps}")
    print(f"  Reward: {optimal_reward:.3f}")
    
    # Try brute force for comparison
    if env.total_balls <= 5:
        bf_reward, bf_steps = compute_optimal_reward_bruteforce_small(env)
        print(f"\nOptimal (brute-force) solution:")
        print(f"  Steps: {bf_steps}")
        print(f"  Reward: {bf_reward:.3f}")


if __name__ == "__main__":
    test_optimal_computation()