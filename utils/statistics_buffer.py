from typing import List
import random

#----------------------------------------------------------------------------#
#                                STATISTICS BUFFER                           #
#----------------------------------------------------------------------------#


class StatBuffer:
    def __init__(self, capacity=100000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.capacity = capacity

        # THIS KEEPS TRACK OF THE STEPS OF THE CURRENT EPISODE, IT IS NEEDED
        # TO COMPUTE STATS BASED ON IT, AND RESET FROM EPISODES

        self.current_episode_steps = 0
        self.placeholder_previous_episode = 0
        self.episodes_in_buffer = 0
        self.total_steps_in_buffer = 0

        # THIS IS FOR MULTIPLE EPISODES
        self.finalrewards = []
        # LIST OF TUPLES (beginstep, endstep, length)
        self.episodes_extremes = []

    # SAME AS ADD BUT WITHOUT COUNTING A STEP, TO USE WHEN RESETTING EPISODE
    def add_base(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        self.current_episode_steps += 1
        self.total_steps_in_buffer += 1

        # MAINTAIN CAPACITY LIMIT
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)

    def end_episode(self):
        # computes the total reward for one episode
        self.finalrewards.append(sum(
            self.rewards[self.placeholder_previous_episode:self.placeholder_previous_episode+self.current_episode_steps]))
        self.episodes_extremes.append(
            (self.placeholder_previous_episode, self.placeholder_previous_episode+self.current_episode_steps, self.current_episode_steps))
        # updates episodes in buffer list
        self.episodes_in_buffer += 1
        # resets the counters of the STEPS STREAM to know when an episode end
        self.placeholder_previous_episode += self.current_episode_steps
        self.current_episode_steps = 0
    
    def add_episodes(self, episodes_data: List) -> None:
        """
        Add multiple episodes to the buffer from alternating training.
        
        Args:
            episodes_data: List of episode data dictionaries
        """
        for episode_data in episodes_data:
            states = episode_data['states']
            actions = episode_data['actions'] 
            rewards = episode_data['rewards']
            
            # Add each step to buffer
            for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
                next_state = states[i+1] if i+1 < len(states) else state
                done = (i == len(states) - 1)
                self.add(state, action, reward, next_state, done)
            
            # End episode
            self.end_episode()
    
    def get_STATEtraj_from_ep(self, epnumber=0):
        return self.states[self.episodes_extremes[epnumber]
                                           [0]:self.episodes_extremes[epnumber][1]]
    
    def get_STATEtraj_stream(self):
        return self.states
    
    def get_ACTtraj_from_ep(self, epnumber=0):
        return self.actions[self.episodes_extremes[epnumber]
                                           [0]:self.episodes_extremes[epnumber][1]]
    
    def get_ACTtraj_stream(self):
        return self.actions
    
    def get_STATE_ACT_traj_from_ep(self,epnumber=0):
        states=self.states[self.episodes_extremes[epnumber]
                                           [0]:self.episodes_extremes[epnumber][1]]
        actions=self.actions[self.episodes_extremes[epnumber]
                                           [0]:self.episodes_extremes[epnumber][1]]
        return list(zip(states, actions))
    
    def get_STATE_ACT_traj_stream(self):
        return list(zip(self.states,self.actions))
    
    def get_STATE_ACT_traj_stream_byep(self):
        return [self.get_STATE_ACT_traj_from_ep(i) for i in range(self.episodes_in_buffer)]
        


#----------------------------------------------------------------------------#
#                    STATISTICS BUFFER SUBCLASS: TESTBUFFER                  #
#----------------------------------------------------------------------------#

# THIS IS JUST A TESTBUFFER, A SAMPLE BUFFER RANDOMIZED TO TEST THE VISUALIZER

class TestBuffer(StatBuffer):
    def __init__(self, num_episodes=50, capacity=100000):
        # Initialize the parent class
        super().__init__(capacity)

        # Generate random episode data
        for ep in range(num_episodes):
            episode_length = random.randint(10, 30)  # Shorter for testing

            # Generate a random episode
            for step in range(episode_length):
                # Generate random state, action, reward, next_state
                state = f"state_{ep}_{step}"
                action = random.randint(0, 3)
                reward = random.uniform(-1, 1)
                next_state = f"state_{ep}_{step+1}"

                # Last step in episode
                is_last = (step == episode_length - 1)

                # Add to buffer
                self.add(state, action, reward, next_state, is_last)

            # End the episode to update tracking variables
            self.end_episode()

        # Verify the buffer has the expected number of episodes
        assert self.episodes_in_buffer == num_episodes
        assert len(self.finalrewards) == num_episodes