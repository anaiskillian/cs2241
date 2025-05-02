# tests/test_agents.py
import unittest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.server import Server
from src.environment.request import Request, RequestType
from src.environment.cluster import ServerCluster
from src.agents.round_robin import RoundRobinAgent
from src.agents.random_agent import RandomAgent
from src.agents.least_loaded import LeastLoadedAgent
from src.agents.mab_agent import MultiArmedBanditAgent, BanditStrategy

class TestRoundRobinAgent(unittest.TestCase):
    def test_round_robin_selection(self):
        agent = RoundRobinAgent(num_servers=3)
        
        # Create a dummy observation
        observation = {
            'server_utils': np.array([5, 8, 3]),
            'latency_history': np.zeros(5),
            'decision_history': np.zeros(5),
            'request_type': np.array([1, 0, 0, 0, 0])
        }
        
        # Agent should cycle through servers
        self.assertEqual(agent.select_action(observation), 0)
        self.assertEqual(agent.select_action(observation), 1)
        self.assertEqual(agent.select_action(observation), 2)
        self.assertEqual(agent.select_action(observation), 0)  # Back to first

class TestLeastLoadedAgent(unittest.TestCase):
    def test_least_loaded_selection(self):
        agent = LeastLoadedAgent(num_servers=3)
        
        # Create a dummy observation with server 2 being least loaded
        observation = {
            'server_utils': np.array([5, 8, 3]),
            'latency_history': np.zeros(5),
            'decision_history': np.zeros(5),
            'request_type': np.array([1, 0, 0, 0, 0])
        }
        
        # Agent should select server 2 (index starts at 0)
        self.assertEqual(agent.select_action(observation), 2)

class TestMABAgent(unittest.TestCase):
    def test_mab_initialization(self):
        agent = MultiArmedBanditAgent(
            num_servers=3,
            num_request_types=5,
            strategy=BanditStrategy.EPSILON_GREEDY,
            epsilon=0.1
        )
        
        self.assertEqual(agent.num_servers, 3)
        self.assertEqual(agent.q_values.shape, (5, 3))
        
    def test_epsilon_greedy_strategy(self):
        # Using fixed seed for reproducibility
        np.random.seed(42)
        
        agent = MultiArmedBanditAgent(
            num_servers=3,
            num_request_types=5,
            strategy=BanditStrategy.EPSILON_GREEDY,
            epsilon=0.0  # Always exploit for testing
        )
        
        # Initialize q-values for testing
        agent.q_values = np.array([
            [0.1, 0.2, 0.3],  # Request type 0: server 2 is best
            [0.3, 0.2, 0.1],  # Request type 1: server 0 is best
            [0.2, 0.3, 0.1],  # Request type 2: server 1 is best
            [0.1, 0.3, 0.2],  # Request type 3: server 1 is best
            [0.3, 0.1, 0.2]   # Request type 4: server 0 is best
        ])
        
        # Test REQUEST_TYPE.SELECT (0)
        observation = {
            'server_utils': np.array([5, 8, 3]),
            'latency_history': np.zeros(5),
            'decision_history': np.zeros(5),
            'request_type': np.array([1, 0, 0, 0, 0])  # One-hot for type 0
        }
        
        # Should select server 2 as it has highest q-value for type 0
        self.assertEqual(agent.select_action(observation), 2)

if __name__ == '__main__':
    unittest.main()
