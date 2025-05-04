# src/utils/workload_gen.py
import numpy as np
from typing import List, Dict, Callable
import collections

from src.environment.request import Request, QueryType

class WorkloadGenerator:
    """
    Generates workload patterns for testing load balancing algorithms.
    """
    def __init__(self, seed=None):
        """
        Initialize workload generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def uniform_workload(self, num_requests, size_range=(0.8, 1.2)):
        """
        Generate a uniform workload with randomly distributed request types.
        
        Args:
            num_requests: Number of requests to generate
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(QueryType)

        for i in range(num_requests):
            req_type = self.rng.choice(request_types)
            size = self.rng.uniform(size_range[0], size_range[1])
            arrival_time = i * 0.1  # Simple uniform arrival

            request = Request.create(
                request_type=req_type,
                arrival_time=arrival_time,
                size=size
            )
            requests.append(request)

        return requests

    def bursty_workload(self, num_requests, burst_factor=3.0, burst_prob=0.1, size_range=(0.8, 1.2)):
        """
        Generate a bursty workload with periods of high request frequency.
        
        Args:
            num_requests: Base number of requests to generate
            burst_factor: Multiplier for number of requests during bursts
            burst_prob: Probability of a burst occurring
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(QueryType)
        arrival_time = 0.0

        i = 0
        while i < num_requests:
            # Determine if this is a burst period
            is_burst = self.rng.random() < burst_prob

            # Number of requests in this period
            period_requests = int(burst_factor * num_requests / 10) if is_burst else int(num_requests / 10)

            # Generate requests for this period
            for j in range(period_requests):
                req_type = self.rng.choice(request_types)
                size = self.rng.uniform(size_range[0], size_range[1])

                # Bunched arrival times during burst, spread out otherwise
                if is_burst:
                    period_arrival = arrival_time + self.rng.exponential(0.01)
                else:
                    period_arrival = arrival_time + self.rng.exponential(0.1)

                request = Request.create(
                    request_type=req_type,
                    arrival_time=period_arrival,
                    size=size
                )
                requests.append(request)

            i += period_requests
            arrival_time += 1.0  # Move to next time period

        # Sort by arrival time
        requests.sort(key=lambda r: r.arrival_time)
        return requests

    def diurnal_workload(self, num_requests, period=24.0, peak_factor=2.0, size_range=(0.8, 1.2)):
        """
        Generate a workload with diurnal (day/night) pattern.
        
        Args:
            num_requests: Base number of requests to generate
            period: Time period for one complete cycle
            peak_factor: Multiplier for number of requests at peak
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(QueryType)

        for i in range(num_requests):
            # Calculate position in cycle (0 to 1)
            cycle_pos = (i % period) / period

            # Calculate rate multiplier based on sine wave
            rate_multiplier = 1.0 + (peak_factor - 1.0) * (np.sin(2 * np.pi * cycle_pos) + 1) / 2

            # Adjust arrival time based on rate
            if i > 0:
                interval = 1.0 / rate_multiplier
                arrival_time = requests[-1].arrival_time + self.rng.exponential(interval)
            else:
                arrival_time = 0.0

            # Select request type with bias toward more complex requests during low periods
            if rate_multiplier < 1.5:  # Low period
                type_weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # More complex queries
            else:  # High period
                type_weights = [0.4, 0.3, 0.1, 0.1, 0.1]  # More simple queries

            req_type = self.rng.choice(request_types, p=type_weights)
            size = self.rng.uniform(size_range[0], size_range[1])

            request = Request.create(
                request_type=req_type,
                arrival_time=arrival_time,
                size=size
            )
            requests.append(request)

        return requests

    def skewed_workload(self, num_requests, skew_factor=0.8, size_range=(0.8, 1.2)):
        """
        Generate a workload with skewed distribution of request types.
        
        Args:
            num_requests: Number of requests to generate
            skew_factor: Zipf parameter for request type distribution
            size_range: Range of request sizes (multiplier)
            
        Returns:
            List of Request objects
        """
        requests = []
        request_types = list(QueryType)

        # Calculate Zipf probabilities
        n = len(request_types)
        zipf_probs = np.array([1.0/((i+1)**skew_factor) for i in range(n)])
        zipf_probs /= zipf_probs.sum()

        for i in range(num_requests):
            # Select request type according to Zipf distribution
            req_type = self.rng.choice(request_types, p=zipf_probs)
            size = self.rng.uniform(size_range[0], size_range[1])
            arrival_time = i * 0.1  # Simple uniform arrival

            request = Request.create(
                request_type=req_type,
                arrival_time=arrival_time,
                size=size
            )
            requests.append(request)

        return requests
