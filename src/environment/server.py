# src/environment/server.py
import numpy as np
from typing import Dict, List, Tuple

class Server:
    """
    Simulates a server with configurable hardware specifications and
    tracks current workload and utilization metrics.
    """
    def __init__(
        self,
        server_id: int,
        cpu_speed: float = 1.0,    # Relative CPU speed (higher is faster)
        ram_size: int = 16,        # RAM in GB
        processing_capacity: int = 10  # How many requests it can process simultaneously
    ):
        self.server_id = server_id
        self.cpu_speed = cpu_speed
        self.ram_size = ram_size
        self.processing_capacity = processing_capacity
        
        # Current state
        self.cpu_utilization = 0.0  # 0-1 range
        self.ram_utilization = 0.0  # 0-1 range
        
        # Current workload tracking
        self.active_requests = {}   # request_id -> (request_obj, time_remaining)
        
        # Performance tracking
        self.completed_requests = 0
        self.total_processing_time = 0
        self.request_type_distribution = {}  # type -> count
        
    def quantized_cpu_util(self, bins: int = 10) -> int:
        """Returns CPU utilization quantized to specified number of bins (1-10)."""
        return min(bins, max(1, int(self.cpu_utilization * bins) + 1))
    
    def can_accept_request(self) -> bool:
        """Check if server can accept a new request based on current capacity."""
        return len(self.active_requests) < self.processing_capacity
    
    def add_request(self, request_id: str, request_obj, processing_time: float) -> bool:
        """
        Attempt to add a request to this server.
        Returns True if accepted, False if rejected.
        """
        if not self.can_accept_request():
            return False
            
        # Adjust processing time based on CPU speed
        adjusted_time = processing_time / self.cpu_speed
        
        # Add to active requests
        self.active_requests[request_id] = (request_obj, adjusted_time)
        
        # Update utilization
        self._update_utilization()
        
        # Track request type distribution
        req_type = request_obj.request_type
        if req_type not in self.request_type_distribution:
            self.request_type_distribution[req_type] = 0
        self.request_type_distribution[req_type] += 1
            
        return True
    
    def step(self, time_delta: float) -> List[Tuple]:
        """
        Simulate server processing for time_delta duration.
        Returns list of completed requests.
        """
        completed = []
        
        # Process each active request
        requests_to_remove = []
        for req_id, (req_obj, time_remaining) in self.active_requests.items():
            # Update time remaining
            new_time = time_remaining - time_delta
            
            if new_time <= 0:
                # Request completed
                completion_time = time_delta + new_time  # Actual time spent in this step
                self.total_processing_time += time_remaining
                completed.append((req_id, req_obj, completion_time))
                requests_to_remove.append(req_id)
                self.completed_requests += 1
            else:
                # Request still processing
                self.active_requests[req_id] = (req_obj, new_time)
        
        # Remove completed requests
        for req_id in requests_to_remove:
            del self.active_requests[req_id]
            
        # Update utilization after processing
        self._update_utilization()
            
        return completed
    
    def _update_utilization(self):
        """Update CPU and RAM utilization based on current workload."""
        # Simple model: utilization proportional to number of active requests
        self.cpu_utilization = len(self.active_requests) / self.processing_capacity if self.processing_capacity > 0 else 0
        
        # For RAM, we could implement a more detailed model based on request types
        # For now, using a simple approximation
        self.ram_utilization = self.cpu_utilization * 0.8  # Simplification
    
    def get_state(self) -> Dict:
        """Return the current state of the server for the agent's observation."""
        return {
            "server_id": self.server_id,
            "cpu_utilization": self.cpu_utilization,
            "quantized_cpu_util": self.quantized_cpu_util(),
            "ram_utilization": self.ram_utilization,
            "active_requests": len(self.active_requests),
            "request_type_distribution": self.request_type_distribution.copy()
        }
    
    def reset(self):
        """Reset server to initial state."""
        self.cpu_utilization = 0.0
        self.ram_utilization = 0.0
        self.active_requests = {}
        self.completed_requests = 0
        self.total_processing_time = 0
        self.request_type_distribution = {}


# src/environment/request.py
import uuid
from enum import Enum, auto
from dataclasses import dataclass

class RequestType(Enum):
    """Enum representing different types of requests with varying resource requirements."""
    SELECT = auto()        # Simple query
    JOIN = auto()          # More complex join operation
    AGGREGATE = auto()     # Aggregate operations (SUM, AVG, etc)
    UPDATE = auto()        # Data modification
    COMPLEX_QUERY = auto() # Complex analytics query
    
    @classmethod
    def get_processing_time(cls, req_type):
        """Return base processing time for each request type."""
        processing_times = {
            cls.SELECT: 1.0,
            cls.JOIN: 2.5,
            cls.AGGREGATE: 2.0,
            cls.UPDATE: 1.5,
            cls.COMPLEX_QUERY: 4.0
        }
        return processing_times.get(req_type, 1.0)
    
    @classmethod
    def get_ram_requirement(cls, req_type):
        """Return RAM requirement (as a percentage of total) for each request type."""
        ram_requirements = {
            cls.SELECT: 0.05,
            cls.JOIN: 0.15,
            cls.AGGREGATE: 0.1,
            cls.UPDATE: 0.05,
            cls.COMPLEX_QUERY: 0.25
        }
        return ram_requirements.get(req_type, 0.05)

@dataclass
class Request:
    """Class representing a request with its properties."""
    request_id: str
    request_type: RequestType
    arrival_time: float
    size: float = 1.0  # Size multiplier affecting processing time
    
    @classmethod
    def create(cls, request_type, arrival_time, size=1.0):
        """Factory method to create a new request with a unique ID."""
        return cls(
            request_id=str(uuid.uuid4()),
            request_type=request_type,
            arrival_time=arrival_time,
            size=size
        )
    
    @property
    def base_processing_time(self):
        """Get the base processing time for this request type."""
        return RequestType.get_processing_time(self.request_type) * self.size
    
    @property
    def ram_requirement(self):
        """Get the RAM requirement for this request type."""
        return RequestType.get_ram_requirement(self.request_type) * self.size


# src/environment/cluster.py
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
        
        # Observation space includes:
        # - Server utilization (quantized 1-10) for each server
        # - Last k latency values
        # - Last k decisions
        # - Request type (one-hot encoded for 5 types)
        self.observation_space = spaces.Dict({
            'server_utils': spaces.Box(low=1, high=10, shape=(self.num_servers,), dtype=np.int32),
            'latency_history': spaces.Box(low=0, high=float('inf'), shape=(history_length,), dtype=np.float32),
            'decision_history': spaces.Box(low=0, high=self.num_servers-1, shape=(history_length,), dtype=np.int32),
            'request_type': spaces.Box(low=0, high=1, shape=(len(RequestType),), dtype=np.int32)
        })
    
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
        Process one step in the environment:
        1. Assign pending request to selected server
        2. Simulate servers processing for time_step duration
        3. Process completed requests
        4. Add new requests
        
        Args:
            action: Index of the server to route the current request to
            
        Returns:
            observation, reward, done, info
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
            # Add to active requests
            self.active_requests[current_request.request_id] = (server_id, self.current_time)
            # Record the decision
            self.decision_history.append(server_id)
        else:
            # Request rejected
            self.rejected_requests.append((current_request, self.current_time))
            self.rejected_count += 1
            # Record rejection (using num_servers as a special code)
            self.decision_history.append(self.num_servers)
        
        # Advance time and process all servers
        self.current_time += self.time_step
        
        completed_requests = []
        for server in self.servers:
            server_completed = server.step(self.time_step)
            for req_id, req_obj, completion_time in server_completed:
                # Look up when the request was assigned
                server_id, start_time = self.active_requests.pop(req_id)
                latency = self.current_time - start_time
                
                completed_requests.append((req_obj, start_time, self.current_time, latency))
                self.completed_requests.append((req_obj, start_time, self.current_time, latency))
                
                # Update metrics
                self.total_latency += latency
                self.completed_count += 1
                
                # Record the latency
                self.latency_history.append(latency)
        
        # Add new request for next step
        if len(self.pending_requests) == 0:
            self._add_new_request()
        
        # Calculate reward based on completed requests in this step
        reward = 0
        if completed_requests:
            # Reward is negative of average latency
            avg_latency = sum(r[3] for r in completed_requests) / len(completed_requests)
            reward = 1.0 / (1.0 + avg_latency)  # Inverse of latency
        else:
            reward = 0.01  # Small positive reward for processing
        
        # Penalty for rejected requests
        if not success:
            reward -= 0.5
            
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'latency': self.get_average_latency(),
            'throughput': self.get_throughput(),
            'completed': self.completed_count,
            'rejected': self.rejected_count,
            'success_rate': self.get_success_rate()
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
        """Construct the observation for the RL agent."""
        # Get server utilizations
        server_utils = np.array([server.quantized_cpu_util() for server in self.servers])
        
        # Get current request type (if any)
        request_type_onehot = np.zeros(len(RequestType))
        if self.pending_requests:
            req_type_idx = self.pending_requests[0].request_type.value - 1  # Enum value starts at 1
            request_type_onehot[req_type_idx] = 1
        
        return {
            'server_utils': server_utils,
            'latency_history': np.array(self.latency_history),
            'decision_history': np.array(self.decision_history),
            'request_type': request_type_onehot
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
