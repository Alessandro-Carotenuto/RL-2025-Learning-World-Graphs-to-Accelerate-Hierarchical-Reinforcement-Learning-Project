import matplotlib.pyplot as plt
from utils.statistics_buffer import StatBuffer

#----------------------------------------------------------------------------#
#                                VISUALIZER CLASS                            #
#----------------------------------------------------------------------------#

# NEEDS A STATBUFFER OBJECT TO WORK
# SEPARATED TO KEEP VISUALIZATION AND DATA COLLECTION SEPARATED


class Visualizer:
    def __init__(self, buffer, height=10, width=20):
        self.buffer = buffer
        # THIS RAISES AN ERROR IF BUFFER IS NOT A STATBUFFER OBJECT
        if buffer is not None and not isinstance(buffer, StatBuffer):
            raise TypeError("buffer must be a StatBuffer object")

        self.height = height
        self.width = width

    # THIS PLOTS THE TOTAL REWARD STREAM OF THE WHOLE TRAINING, REW PER STREAM

    def total_reward_stream(self):
        self.steps = range(0, self.buffer.total_steps_in_buffer)
        self.rewards = self.buffer.rewards

        plt.figure(figsize=(self.width, self.height))
        plt.plot(self.steps, self.rewards, 'r-')
        plt.xlabel('Steps', fontsize=26)
        plt.ylabel('Reward per Step', fontsize=26)
        plt.title('Total Step-Reward Stream', fontsize=26)
        plt.grid(True, alpha=0.3)

    # THIS PLOTS THE TOTAL REWARD STREAM OF THE WHOLE TRAINING
    # EP BY EP CUMULATIVE, THAT IS PROBABLY THE MOST USEFUL ONE

    def total_reward_by_ep_stream(self):

        self.episodes = range(0, self.buffer.episodes_in_buffer)
        self.rewards = self.buffer.finalrewards

        plt.figure(figsize=(self.width, self.height))
        plt.plot(self.episodes, self.rewards, 'b-')
        plt.xlabel('Episodes', fontsize=26)
        plt.ylabel('Total Reward', fontsize=26)
        plt.title('Rewards by episode', fontsize=26)
        plt.grid(True, alpha=0.3)

    # THIS PLOTS THE REWARD PROGRESSION OF A SPECIFIC EPISODE

    def reward_stream_for_ep(self, epnumber):

        self.steps = range(0, self.buffer.episodes_extremes[epnumber][2])
        self.rewards = self.buffer.rewards[self.buffer.episodes_extremes[epnumber]
                                           [0]:self.buffer.episodes_extremes[epnumber][1]]
        
        plt.figure(figsize=(self.width, self.height))
        plt.plot(self.steps, self.rewards, 'g-')
        plt.xlabel('Steps', fontsize=26)
        plt.ylabel('Reward per Step', fontsize=26)
        plt.title('Total Step-Reward Stream for episode ' +
                  str(epnumber), fontsize=26)
        plt.grid(True, alpha=0.3)