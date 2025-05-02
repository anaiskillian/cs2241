import numpy as np
from typing import List, Dict, Tuple, Optional
import gym
from gym import spaces
import collections

from .server import Server
from .request import Request, RequestType

class ServerCluster(gym.Env):
    """
    Environment simulating a cluster of servers processing requests.
    Implements the OpenAI Gym interface for reinforcement learning.
    """
    def __init__(
        self,
        num_servers: int = 4,
        server_configs: Optional[List[Dict]] = None,
        history_length: int = 5,
        time_step: float = 0.1,
        max_steps: int = 1000
    ):
        super().__init__()

        self.time_step = time_step
        self.current_time = 0
        self.max_steps = max_steps
        self.steps_taken = 0
        self.delayed_rewards = []
        self.history_length = history_length

        # Initialize servers
        self.servers = []
        if server_configs:
            for i, config in enumerate(server_configs):
                self.servers.append(Server(server_id=i, **config))
        else:
            # Create default homogeneous servers
            for i in range(num_servers):
                self.servers.append(Server(server_id=i))

        self.num_servers = len(self.servers)

        # Request tracking
        self.pending_requests = []  # Requests waiting to be assigned
        self.active_requests = {}   # request_id -> (server_id, start_time)
        self.completed_requests = []  # Stores completed request info
        self.rejected_requests = []  # Stores rejected request info

        # History tracking for RL state
        self.latency_history = collections.deque(maxlen=history_length)
        self.decision_history = collections.deque(maxlen=history_length)

        # Fill history with default values
        for _ in range(history_length):
            self.latency_history.append(0.0)
            self.decision_history.append(0)

        # Performance metrics
        self.total_latency = 0.0
        self.request_count = 0
        self.completed_count = 0
        self.rejected_count = 0

        # Action and observation spaces for RL
        self.action_space = spaces.Discrete(self.num_servers)

        # Observation space: only latency history, decision history, request type
        self.observation_space = spaces.Dict(
            {
                "latency_history": spaces.Box(
                    low=0, high=float("inf"), shape=(history_length,), dtype=np.float32
                ),
                "decision_history": spaces.Box(
                    low=0,
                    high=self.num_servers - 1,
                    shape=(history_length,),
                    dtype=np.int32,
                ),
                "request_type": spaces.Box(
                    low=0, high=1, shape=(len(RequestType),), dtype=np.int32
                ),
            }
        )

    def reset(self):
        """Reset the environment to initial state."""
        for server in self.servers:
            server.reset()

        self.current_time = 0
        self.steps_taken = 0
        self.pending_requests = []
        self.active_requests = {}
        self.completed_requests = []
        self.rejected_requests = []

        # Reset history
        self.latency_history = collections.deque(maxlen=self.history_length)
        self.decision_history = collections.deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.latency_history.append(0.0)
            self.decision_history.append(0)

        # Reset metrics
        self.total_latency = 0.0
        self.request_count = 0
        self.completed_count = 0
        self.rejected_count = 0

        # Add initial request to get started
        self._add_new_request()

        return self._get_observation()

    def step(self, action):
        """
        Real-mode only: assign request, run real async, only process completed real requests.
        """
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        # Get current request to be routed
        if not self.pending_requests:
            self._add_new_request()

        current_request = self.pending_requests.pop(0)
        server_id = action

        # Try to assign request to the selected server
        success = False
        if 0 <= server_id < self.num_servers:
            server = self.servers[server_id]
            success = server.add_request(
                current_request.request_id,
                current_request,
                current_request.base_processing_time
            )

        # Track the request
        if success:
            self.active_requests[current_request.request_id] = (
                server_id,
                self.current_time,
            )
            self.decision_history.append(server_id)
        else:
            self.rejected_requests.append((current_request, self.current_time))
            self.rejected_count += 1
            self.decision_history.append(self.num_servers)

        # Advance time (for bookkeeping)
        self.current_time += self.time_step

        # Only process completed real requests
        completed_requests = []
        for server in self.servers:
            server_completed = server.step(self.time_step)
            for result in server_completed:
                # We expect (request_id, latency) or similar, but original code expects (req_id, req_obj, completion_time)
                # Map to (req_id, req_obj, completion_time=latency)
                if isinstance(result, tuple) and len(result) == 2:
                    req_id, latency = result
                    # Look up request_obj and start_time
                    if req_id in self.active_requests:
                        s_id, start_time = self.active_requests.pop(req_id)
                        # For real mode, we treat latency as completion_time
                        completed_requests.append(
                            (None, start_time, self.current_time, latency)
                        )
                        self.completed_requests.append(
                            (None, start_time, self.current_time, latency)
                        )
                        self.total_latency += latency
                        self.completed_count += 1
                        self.latency_history.append(latency)
                elif isinstance(result, tuple) and len(result) == 3:
                    req_id, req_obj, latency = result
                    if req_id in self.active_requests:
                        s_id, start_time = self.active_requests.pop(req_id)
                        completed_requests.append(
                            (req_obj, start_time, self.current_time, latency)
                        )
                        self.completed_requests.append(
                            (req_obj, start_time, self.current_time, latency)
                        )
                        self.total_latency += latency
                        self.completed_count += 1
                        self.latency_history.append(latency)

        # Add new request for next step
        if len(self.pending_requests) == 0:
            self._add_new_request()

        # Reward: negative of average latency for this step if any completed, else small reward
        if completed_requests:
            avg_latency = sum(r[3] for r in completed_requests) / len(completed_requests)
            reward = 1.0 / (1.0 + avg_latency)
        else:
            reward = 0.01
        if not success:
            reward -= 0.5

        observation = self._get_observation()
        info = {
            "latency": self.get_average_latency(),
            "throughput": self.get_throughput(),
            "completed": self.completed_count,
            "rejected": self.rejected_count,
            "success_rate": self.get_success_rate(),
        }
        return observation, reward, done, info

    def _add_new_request(self):
        """Add a new request to the pending queue."""
        # Randomly select a request type
        req_type = np.random.choice(list(RequestType))

        # Create the request
        request = Request.create(
            request_type=req_type,
            arrival_time=self.current_time,
            size=np.random.uniform(0.8, 1.2)  # Random size variation
        )

        self.pending_requests.append(request)
        self.request_count += 1

        return request

    def _get_observation(self):
        """Construct the observation for the RL agent (real mode: no server_utils)."""
        request_type_onehot = np.zeros(len(RequestType))
        if self.pending_requests:
            req_type_idx = self.pending_requests[0].request_type.value - 1
            request_type_onehot[req_type_idx] = 1
        return {
            "latency_history": np.array(self.latency_history),
            "decision_history": np.array(self.decision_history),
            "request_type": request_type_onehot,
        }

    def get_average_latency(self):
        """Calculate average latency for completed requests."""
        if self.completed_count == 0:
            return 0
        return self.total_latency / self.completed_count

    def get_throughput(self):
        """Calculate throughput as requests per time unit."""
        if self.current_time == 0:
            return 0
        return self.completed_count / self.current_time

    def get_success_rate(self):
        """Calculate the percentage of requests that were not rejected."""
        if self.request_count == 0:
            return 1.0
        return (self.request_count - self.rejected_count) / self.request_count

    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"Time: {self.current_time:.2f}, Step: {self.steps_taken}")
            print(f"Requests: {self.request_count} total, {self.completed_count} completed, {self.rejected_count} rejected")
            print(f"Avg Latency: {self.get_average_latency():.4f}, Throughput: {self.get_throughput():.4f} req/s")

            print("\nServer Status:")
            for i, server in enumerate(self.servers):
                print(f"Server {i}: CPU {server.cpu_utilization*100:.1f}%, {len(server.active_requests)} active requests")

        return None
