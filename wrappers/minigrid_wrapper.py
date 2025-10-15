from enum import Enum

import minigrid
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

import random

import math

import torch
import torch.nn as nn

from collections import deque

import numpy as np

#----------------------------------------------------------------------------#
#                            ENV MODES                                       #
#----------------------------------------------------------------------------#

class EnvModes(Enum):
    DOORKEY=1
    MULTIGOAL=2
    
class EnvSizes(Enum):
    EXTRASMALL=6
    SMALL=10
    MEDIUM=24
    LARGE=36
    EXTRALARGE=52


#----------------------------------------------------------------------------#
#                            CUSTOM MINIGRID WRAPPER                         #
#----------------------------------------------------------------------------#
# Allows Custon Environent Creeation


class MinigridWrapper(MiniGridEnv):

    def __init__(  # CONSTRUCTOR AND SET-UP
        self,
        size=EnvSizes.MEDIUM,
        #XS = 6
        #SMALL = 10
        #MEDIUM = 24
        #LARGE = 36
        #XL    = 52
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        mode=EnvModes.DOORKEY,
        phase_one_eps=5,
        **kwargs
    ):

        # SIZE VARIABLE
        self.size = int(size.value)

        self.mode = mode

        self.randomgen=True
        self.firstgen = True  # IS THE FIRST GENERATION
        self.firstgen_phase2 = True

        # PHASES ------------------------------------------------

        self.phase = 1  # phase can be 1 or 2
        self.phase1_reset_num = phase_one_eps

        # BALL POSITIONS FOR MULTIGOAL MODE ---------------------

        self.active_balls = set()  # Track ball positions
        self.total_balls = 0
        self.balls_collected = 0

        # ------------------------------------------------------

        # SETUP INITIAL POSITION-DIRECTION
        if self.randomgen:
            xstart = random.randint(1, self.size-2)
            ystart = random.randint(1, self.size-2)
            self.agent_start_pos = (xstart, ystart)
            self.agent_start_dir = random.randint(0, 3)
            print(self.agent_start_pos)

        # GENERATE MISSION SPACE (EXPLAINED IN THE GEN-MISSION FUNCTION)
        # --- THIS GETS PASSED TO THE PARENT CLASS TO GET SET AS AN ATTRIBUTE
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # ARBITRARLY SET MAX_STEP LIMIT IF NOT PROVIDED
        if max_steps is None:
            max_steps = 4 * self.size**2

        # SET-UP BASED ON PARENT CLASS
        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,

            # OBVIOUS FASTER TRAINING IF TRUE:
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return 'XL ENVIRONMENT'

#----------------------------------------------------------------------------#
#                    BFSs                                                    #
#----------------------------------------------------------------------------#

    def BFS(self, startpos, endpos, playerpos):
        """
        Uses Breadth-First Search (BFS) to check if there's a path 
        between two positions in the minigrid environment.

        Args:
            startpos (tuple): Starting position as (x, y)
            endpos (tuple): Target position as (x, y)

        Returns:
            tuple: (solvability: bool, path: list)
                   - solvability: True if path exists, False otherwise  
                   - path: List of (x, y) coordinates from start to end, 
                          empty list if no path found
        """

        # Validate input positions
        if not startpos or not endpos:
            return False, []

        # BFS data structures
        queue = deque([startpos])           # Queue for positions to explore
        visited = set()                     # Track explored positions
        visited.add(startpos)
        # Track parent of each position for path reconstruction
        parent = {}

        # BFS main loop - explore positions level by level
        while queue:
            # Get next position to explore (FIFO - breadth-first)
            current_pos = queue.popleft()
            x, y = current_pos

            # Check if we've reached the goal
            if current_pos == endpos:
                # Goal found! Reconstruct the path by backtracking through parents
                path = []
                current = current_pos

                # Follow parent chain from goal back to start
                while current in parent:
                    path.append(current)
                    current = parent[current]

                # Add start position (it has no parent)
                path.append(startpos)

                # Reverse path to get start->goal order instead of goal->start
                path.reverse()

                return True, path

            # Explore all 4 adjacent cells (up, down, left, right)
            # Movement directions: up(-1,0), down(+1,0), left(0,-1), right(0,+1)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # Calculate new position
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)

                # Add these checks BEFORE self.grid.get(nx, ny)
                if not (1 <= nx < self.size - 1 and 1 <= ny < self.size - 1):
                    continue
                if new_pos in visited:
                    continue

                # Get the object at this grid position
                cell_obj = self.grid.get(nx, ny)

                px = playerpos[0]
                py = playerpos[1]
                if nx == px and ny == py:
                    playertile = True
                else:
                    playertile = False

                # Check if this position is valid to move to
                if (
                    # Within grid bounds (excluding border walls)
                    1 <= nx < self.size - 1 and
                    1 <= ny < self.size - 1 and
                    # Haven't visited this position yet
                    new_pos not in visited and
                    # Position is traversable (empty, goal, or key)
                    self._is_traversable(cell_obj) and (
                        not playertile or new_pos == endpos)

                ):
                    # Valid position found - add to exploration queue

                    queue.append(new_pos)
                    visited.add(new_pos)

                    # Record parent relationship for path reconstruction
                    parent[new_pos] = current_pos

        # Queue exhausted without finding goal - no path exists
        return False, []
    
    def BFS_all_reachable(self, startpos):
        """
        Uses Breadth-First Search (BFS) to find all positions reachable 
        from a starting position in the minigrid environment.
        
        Args:
            startpos (tuple): Starting position as (x, y)
            
        Returns:
            list: List of all reachable (x, y) coordinates from startpos
        """
        # Validate input position
        if not startpos:
            return []
        
        # BFS data structures
        queue = deque([startpos])           # Queue for positions to explore
        visited = set()                     # Track explored positions
        visited.add(startpos)
        
        # BFS main loop - explore all reachable positions
        while queue:
            # Get next position to explore (FIFO - breadth-first)
            current_pos = queue.popleft()
            x, y = current_pos
            
            # Explore all 4 adjacent cells (up, down, left, right)
            # Movement directions: up(-1,0), down(+1,0), left(0,-1), right(0,+1)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # Calculate new position
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)
                
                # Skip if already visited
                if new_pos in visited:
                    continue
                    
                # Skip if out of bounds
                if not (1 <= nx < self.size - 1 and 1 <= ny < self.size - 1):
                    continue
                
                # Get the object at this grid position
                cell_obj = self.grid.get(nx, ny)
                
                # Check if this position is valid to move to
                if (
                    # Within grid bounds (excluding border walls)
                    1 <= nx < self.size - 1 and
                    1 <= ny < self.size - 1 and
                    # Position is traversable (empty, goal, or key)
                    self._is_traversable(cell_obj)
                ):
                    # Valid position found - add to exploration queue
                    queue.append(new_pos)
                    visited.add(new_pos)
        
        # Convert visited set to list and return all reachable positions
        return list(visited)

    def _is_traversable(self, cell_obj):
        
        """
        Helper method to determine if a cell can be traversed.

        Args:
            cell_obj: The object at the grid position (from self.grid.get())

        Returns:
            bool: True if the cell can be traversed, False otherwise
        """

        # Empty cell (None) is always traversable
        if cell_obj is None:
            return True

        # Goal is always traversable
        if isinstance(cell_obj, Goal):
            # print("BFS: Goal")
            return True

        # Key is traversable (agent can walk on it to pick it up)
        if isinstance(cell_obj, Key):
            # print("BFS: Key")
            return True

        # Walls are never traversable
        if isinstance(cell_obj, Wall):
            return False

        # Doors - only traversable if unlocked
        if isinstance(cell_obj, Door):
            # print("BFS: Door")
            return True

        
            # Ball is traversable (agent can walk on it to collect it)
        if isinstance(cell_obj, Ball):
            return True


        # For any other object type, default to not traversable
        return False

#----------------------------------------------------------------------------#
#                    RANDOMIZATIONS                                          #
#----------------------------------------------------------------------------#

    # PLACES A RANDOM KEY AND RESERVES THE TILE
    def PlaceRandomKey(self, color=0):

        keyrow = 0
        keycol = 0

        while (self.placeable_grid[keycol][keyrow] == False):
            # key position
            keypos = random.randint(1, self.size**2-1)
            keyrow = math.floor(keypos/self.size)
            keycol = keypos-keyrow*self.size

        self.grid.set(keycol, keyrow, Key(COLOR_NAMES[color]))
        print("key placed at", keypos, keycol, keyrow)
        self.placeable_grid[keycol][keyrow] = False
        print("key tile reserved")

        return (keycol, keyrow)

    # PLACES A RANDOM DOOR AND RESERVES THE TILE
    def PlaceRandomDoor(self, color=0):  # ON A WALL
        doorrow = 0
        doorcol = 0

        cellobj = self.grid.get(doorcol, doorrow)
        wflag = isinstance(cellobj, Wall)

        while (wflag == False or (doorrow == 0 and doorcol == 0)):
            # place door
            doorpos = random.randint(1, self.size**2-1)
            doorrow = math.floor(doorpos/self.size)
            doorcol = doorpos-doorrow*self.size

            cellobj = self.grid.get(doorcol, doorrow)
            wflag = isinstance(cellobj, Wall)

        self.grid.set(doorcol, doorrow, Door(
            COLOR_NAMES[color], is_locked=True))
        print("door placed at", doorpos, doorcol, doorrow)
        self.placeable_grid[doorcol][doorrow] = False
        print("door tile reserved")

        return (doorcol, doorrow)

    # PLACES A RANDOM DOOR ON A PATH TO GOAL, ON A WALL IF PRESENT
    def DoorOnPathToGoal(self, path, color=0):
        doorrow = 0
        doorcol = 0

        isthereawall = False

        print("LUNGHEZZA", len(path))
        if len(path) <= 2:
            return (doorcol, doorrow)

        for i in range(0, len(path)):
            cellobj = self.grid.get(path[i][0], path[i][1])
            isthereawall = isinstance(cellobj, Wall)
            if (isthereawall):
                isthereawall = True  # useless
                break
            else:
                pass

        cellobj = self.grid.get(doorcol, doorrow)
        wflag = False

        if (isthereawall):
            while (wflag == False):
                # place door
                pathpoint = random.randint(1, len(path)-1)
                point = path[pathpoint]

                doorrow = point[1]
                doorcol = point[0]

                cellobj = self.grid.get(doorcol, doorrow)
                wflag = isinstance(cellobj, Wall)
        else:
            print("No Wall")
            while (self.placeable_grid[doorcol][doorrow] == False):
                pathpoint = random.randint(1, len(path)-1)
                point = path[pathpoint]
    
                doorrow = point[1]
                doorcol = point[0]

        self.grid.set(doorcol, doorrow, Door(
            COLOR_NAMES[color], is_locked=True))
        print("door placed at", doorcol, doorrow)
        self.placeable_grid[doorcol][doorrow] = False
        print("door tile reserved")

        return (doorcol, doorrow)

    # PLACES A RANDOM GOAL AND RESERVES THE TILE
    def PlaceRandomGoal(self):
        goalrow = 0
        goalcol = 0

        while (self.placeable_grid[goalcol][goalrow] == False):
            # place goal
            goalpos = random.randint(1, self.size**2-1)
            goalrow = math.floor(goalpos/self.size)
            goalcol = goalpos-goalrow*self.size

        self.put_obj(Goal(), goalcol, goalrow)
        print("goal placed at", goalpos, goalcol, goalrow)
        self.placeable_grid[goalcol][goalrow] = False
        print("goal tile reserved")

        return (goalcol, goalrow)

    # REMOVES WALL FROM A PATH, EXCEPT FOR A SINGLE POINT
    def ClearPath_exceptionpoint(self, path, point):
        i = 0
        for pos in path:
            x = pos[0]
            y = pos[1]
            cellobj = self.grid.get(x, y)
            if isinstance(cellobj, Wall):
                if (not (x == point[0] and y == point[1])):
                    self.grid.set(x, y, None)
                    i += 1

        print("Path Cleared! Path length:", len(path), "Cleared:", i)
        return

    # RESERVES A PATH
    def ReservePath(self, path):
        print("LEN", len(path))
        for pos in path:
            x = pos[0]
            y = pos[1]
            self.placeable_grid[x][y] = False
        return

    # NOISE FILLS THE WALLS
    def NoiseFiller(self, threshold=0.5, noise_scale=0.1):
        for x in range(1, self.size-1):
            for y in range(1, self.size-1):

                # Only place walls on unreserved tiles
                if self.placeable_grid[x][y] == True:
                    noise_value = random.random()

                    # Alternative: pseudo-random based on position
                    # noise_value = (math.sin(x * noise_scale) * math.cos(y * noise_scale) + 1) / 2

                    # Place wall if noise exceeds threshold
                    if noise_value > threshold:
                        self.grid.set(x, y, Wall())
                        self.placeable_grid[x][y] = False

    # RESET PLACEABLE GRID
    def placeablegrid_reset(self):
        # Create a matrix (numpy) full of "1"
        #   basically when i wanna mark a place on the grid in which you cannot
        #   place a wall or something, the corrispective point in the grid will
        #   be marked as "zero"

        self.placeable_grid = np.ones((self.size, self.size), dtype=bool)

        self.placeable_grid[0, :] = False
        self.placeable_grid[self.size-1, :] = False
        self.placeable_grid[:, self.size-1] = False
        self.placeable_grid[:, 0] = False

        # Reserve the player tile
        self.placeable_grid[self.agent_start_pos[0]
                            ][self.agent_start_pos[1]] = False

        print("Placeable Grid and base rectangle generated")

        return

    # RESET GRID AND PLACEABLE GRID
    def resetgrid(self):
        self.grid = Grid(self.w, self.h)
        self.grid.wall_rect(0, 0, self.w, self.h)
        self.placeablegrid_reset()

    def removeitems(self):
        for x in range(self.w):
            for y in range(self.h):
                tile = self.grid.get(x, y)
                if (not(isinstance(tile, Wall))):
                    self.grid.set(x, y, None)
                    self.placeable_grid[x][y] = True

    # RANDOM GENERATES A SOLVABLE MAZE
    def EasyDoorKeyMap(self):  

        playerpos = (self.agent_start_pos[0], self.agent_start_pos[1])
        solve1 = False  # PLAYER KEY
        solve2 = False  # KEY DOOR
        # solve3=False #PLAYER GOAL DIRECT

        while ((not (solve1 and solve2))):
            self.resetgrid()
            keypos = self.PlaceRandomKey()  # places a key and reserves the tile
            goalpos = self.PlaceRandomGoal()  # places a goal and reserves the tile

            self.NoiseFiller(0.70)  # noise and reserves

            solve1, keypath = self.BFS(
                keypos, (self.agent_start_pos[0], self.agent_start_pos[1]), self.agent_start_pos)
            if (not solve1):
                print("NO AGENT-KEY PATH\n")
                continue
            self.ReservePath(keypath)

            solve2, keytogoal = self.BFS(goalpos, keypos, self.agent_start_pos)
            if (not solve2):
                print("NO KEY-GOAL PATH\n")
                continue

            self.NoiseFiller(0.9)

            doorpos = self.DoorOnPathToGoal(keytogoal)
            if (doorpos == (0, 0)):
                print("DOOR-GOAL TOO CLOSE")
                solve2 = False
            self.ClearPath_exceptionpoint(keytogoal, doorpos)
            self.ReservePath(keytogoal)

            if (not (solve1 and solve2)):
                print("--- RESET ---\n")

        self.removeitems()
        return goalpos, playerpos, doorpos, keypos

    def Phase2_ResetDoorKey(self, color=0):
        if self.firstgen_phase2:
            # PLACE PLAYER
            self.agent_pos = (self.p2playerpos[0],self.p2playerpos[1])

            # PLACE KEY
            key_x = self.p2keypos[0]
            key_y = self.p2keypos[1]
            
            self.grid.set(key_x, key_y, Key(COLOR_NAMES[color]))
            self.placeable_grid[key_x][key_y] = False

            # PLACE DOOR
            door_x = self.p2doorpos[0]
            door_y = self.p2doorpos[1]
            
            self.grid.set(door_x, door_y, Door(COLOR_NAMES[color], is_locked=True))
            self.placeable_grid[door_x][door_y] = False

            # PLACE GOAL
            goal_x = self.p2goalpos[0]
            goal_y = self.p2goalpos[1]
            
            self.put_obj(Goal(), goal_x, goal_y)
            self.placeable_grid[goal_x][goal_y] = False
            
            self.firstgen_phase2=False

        else:
            
            solve1 = False  # PLAYER KEY
            solve2 = False  # KEY DOOR
            
            while ((not (solve1 and solve2))):
                self.removeitems()
                
                keypos = self.PlaceRandomKey()  # places a key and reserves the tile
                goalpos = self.PlaceRandomGoal()  # places a goal and reserves the tile
                
                solve1,keypath=self.BFS(keypos, self.agent_pos, self.agent_pos)
                if (not solve1):
                    print("PHASE 2: No Agent-Key Path")
                solve2,keytogoal=self.BFS(keypos,goalpos, self.agent_pos)
                if (not solve2):
                    print("PHASE 2: NO Key-Goal Path")
                
                doorpos= self.DoorOnPathToGoal(keytogoal)
                if (doorpos == (0, 0)):
                    print("DOOR-GOAL TOO CLOSE")
                    solve2 = False
                if (not (solve1 and solve2)):
                        print("--- RESET (phase 2) ---\n")
    
    def ResetMultiGoals(self, playerpos, goals=5):
        self.removeitems()
        reachables = self.BFS_all_reachable(playerpos)
        
        ballpos = []
        for i in range(goals):
            px = playerpos[0]
            py = playerpos[1]
            while (px == playerpos[0] and py == playerpos[1]):
                point = random.randint(0, len(reachables)-1)
                px = reachables[point][0]
                py = reachables[point][1]
                
            self.grid.set(px, py, Ball(COLOR_NAMES[i]))
            #print("BALL PLACED", px, py)
            self.placeable_grid[px][py] = False
            ballpos.append((px, py))
        
        # Track active balls
        self.active_balls = set(ballpos)
        self.total_balls = len(ballpos)
        self.balls_collected = 0
        
        return ballpos

    def EasyGeneralPurposeMap(self):
        
       self.resetgrid()
       playerpos = (self.agent_start_pos[0], self.agent_start_pos[1])
       self.placeable_grid[playerpos[0]][playerpos[1]]=False
       self.NoiseFiller(0.65)  # noise and reserves
       return

    def ascii_encode_gridelement(self,obj):
        if obj is None:
            return '-'
        elif isinstance(obj, Wall):
            return '#'
        elif isinstance(obj, Goal):
            return 'G'
        elif isinstance(obj, Key):
            return 'K'
        elif isinstance(obj, Door):
            return 'D'
        elif isinstance(obj, Ball):
            return 'B'
        else:
            return '.'

    def getGridState(self,view=False):
        ASCIIGRID=[]
        for y in range(self.h):
            row = []
            for x in range(self.w):
                EL=self.ascii_encode_gridelement(self.grid.get(x,y))
                if view:
                    print(EL, end='')
                row.append(EL)
            ASCIIGRID.append(row)
            print()
        return ASCIIGRID
    
    # STARTS GRID GENERATION
    def _gen_grid(self, width, height):

        # # Generate vertical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())

        if self.agent_start_pos is not None:
            # self.agent_pos = np.array(self.agent_start_pos)
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            # Place the agent
            self.place_agent()

        # Place the door and key
        if self.randomgen == True:
            if self.firstgen:

                # Create an empty grid
                self.grid = Grid(width, height)
                self.w = width
                self.h = height
                # Generate the surrounding walls
                self.grid.wall_rect(0, 0, width, height)

                # Create a matrix (numpy) full of "1"
                #   basically when i wanna mark a place on the grid in which you cannot
                #   place a wall or something, the corrispective point in the grid will
                #   be marked as "zero"

                self.placeablegrid_reset()
                
                if self.mode==EnvModes.DOORKEY:
                    self.p2goalpos, self.p2playerpos, self.p2doorpos, self.p2keypos = self.EasyDoorKeyMap()
                elif self.mode==EnvModes.MULTIGOAL:
                    self.EasyGeneralPurposeMap()
                
                self.firstgen = False
                self.phasetestcounter=0
            else:
                if self.phase == 1:
                    self.place_agent()
                    self.phasetestcounter+=1
                    if self.phasetestcounter>=self.phase1_reset_num:
                        self.phase+=1
                elif self.phase == 2:
                    if self.mode==EnvModes.DOORKEY:
                        self.Phase2_ResetDoorKey()
                    elif self.mode==EnvModes.MULTIGOAL:
                        playerpos = (self.agent_start_pos[0], self.agent_start_pos[1])
                        self.ResetMultiGoals(playerpos)  # This now sets up tracking
                
        # else:
        #     self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        #     self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        self.mission = self._gen_mission()

    # def step(self, action):
    #     #ACTUAL STEP
    #     obs, reward, terminated, truncated, info = self.env.step(action)

    #     #LAUNCHES PROCESSING OF OBSERVATIONS AND REWARDS
    #     processed_obs = self._process_obs(obs)
    #     normalized_reward = self._normalize_reward(reward)
    #     return processed_obs, normalized_reward, terminated, truncated, info

    #OVERRIDING STEP TO CUSTOMIZE REWARDS
    def step(self, action):
        if self.mode == EnvModes.MULTIGOAL and self.phase == 2:
            ball_collected = False
            
            if action == self.actions.forward:
                fwd_pos = self.front_pos
                
                # Boundary check before accessing grid
                if (0 <= fwd_pos[0] < self.width and 
                    0 <= fwd_pos[1] < self.height):
                    fwd_cell = self.grid.get(*fwd_pos)
                    
                    if isinstance(fwd_cell, Ball) and tuple(fwd_pos) in self.active_balls:
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)
                        self.active_balls.remove(tuple(fwd_pos))
                        self.balls_collected += 1
                        ball_collected = True
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.mode == EnvModes.MULTIGOAL and self.phase == 2 and ball_collected:
            reward += 1  # Per ball reward
            
            if len(self.active_balls) == 0:
                terminated = True
                # Terminal bonus: +0.5 max, reduced by time penalty
                time_penalty = 0.45 * (self.step_count / self.max_steps)
                reward += (0.6 - time_penalty)

        
        return obs, reward, terminated, truncated, info

    def _process_obs(self, obs):
        # EXTRACT AND PROCESS THE OBSERVATIONS
        return obs
