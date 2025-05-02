import uuid
from enum import Enum, auto
from dataclasses import dataclass

class RequestType(Enum):
    "DEPRECATED. DONT USE DIRECTLY"
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
