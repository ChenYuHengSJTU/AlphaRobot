import numpy as np
import os

from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
BASE_PATH = '/tmp/colab_dope_run'
GAME = 'Asterix'
LOG_PATH = os.path.join(BASE_PATH, 'random_dqn', GAME)
class MyRandomDQNAgent(dqn_agent.DQNAgent):
    def __init__(self, sess, num_actions):
        """This maintains all the DQN default argument values."""
        super(MyRandomDQNAgent, self).__init__(sess, num_actions)
    def step(self, reward, observation):

        _ = super(MyRandomDQNAgent, self).step(reward, observation)
        return np.random.randint(self.num_actions)
    
    def create_random_dqn_agent(sess, environment):

        return MyRandomDQNAgent(sess, num_actions=environment.action_space.n)
# Create the runner class with this agent. We use very small numbers of
#    steps
# to terminate quickly, as this is mostly meant for demonstrating how one
#     can
# use the framework. We also explicitly terminate after 110 iterations (
#    instead
# of the standard 200) to demonstrate the plotting of partial runs.
random_dqn_runner = run_experiment.Runner(LOG_PATH,
                                          create_random_dqn_agent,
                                          game_name=GAME,
                                          num_iterations=200,
                                          training_steps=10,
                                          evaluation_steps=10,
                                          max_steps_per_episode=100)
# print(’Will train agent, please be patient, may be a while...’)
random_dqn_runner.run_experiment()
# print(’Done training!’)
