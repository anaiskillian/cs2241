"""
A Gym-compatible environment that routes **live** SQL queries to real machines.

Key features
============
1.  *Real, asynchronous queries* - each `step()` fires a non-blocking query
    to the selected host via an `asyncpg` pool.  Latency is recorded when the
    result comes back and is fed into the reward signal.

2.  *Extensible query generation* - pluggable `query_generator`.  The default
    simply yields `"SELECT 1;"` with type `QueryType.SIMPLE_SELECT`, but you
    can swap in the rich generator produced by `generate_select_queries.py`
    at any time.

3.  *Live server-utilisation signal* - every step exposes a `(CPU, RAM)`
    vector per host.  For now these numbers are random placeholders; a
    background thread can update `self.server_utils` in real time without
    touching the RL loop.

4.  *Thread-safe latency queue* - `send_query_to_host()` pushes its results
    into an `asyncio.Queue`, so concurrent tasks never tread on each other.

---------------------------------------------------------------------------
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import asyncpg
import gym
import numpy as np
from gym import spaces

from .request import QueryType, Request
from .remote_utils import send_query_to_host

# from ..remote_utils import send_query_to_host


class RealServerCluster(gym.Env):
    """OpenAI Gymnasium environment that speaks to *real* PostgreSQL servers."""

    metadata = {"render.modes": ["human"]}

    # ------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------
    @staticmethod
    async def _build_pool(host: str, db_cfg: Dict) -> asyncpg.pool.Pool:
        """Create a fixed-size asyncpg pool for one host."""
        cfg = dict(db_cfg)  # shallow-copy so we can tweak
        cfg["host"] = host
        pool = await asyncpg.create_pool(**cfg)
        return pool

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def __init__(
        self,
        hosts: List[str],
        db_cfg: Dict,
        *,
        history_length: int = 10,
        max_steps: int = 1_000,
        pool_size: int = 99,  # leave room for one connection that is manual
        query_generator: Optional[Callable[[], Tuple[str, QueryType]]] = None,
        # --- util placeholders – can be hot-patched from another thread ----
        init_cpu_util: float = 0.1,
        init_ram_util: float = 0.1,
    ):
        """
        Parameters
        ----------
        hosts : list[str]
            DNS names or IPs of the PostgreSQL back-ends.
        db_cfg : dict
            Keys accepted by `asyncpg.create_pool` *except* `"host"`.  At
            minimum: `"user"`, `"password"`, `"database"`, `"port"`.
        history_length : int
            How many past latency/decision values to expose.
        max_steps : int
            Episode length before `done=True`.
        pool_size : int
            Fixed `min_size=max_size` for each host’s pool.
        query_generator : callable | None
            Function that returns `(sql_string, QueryType)` each time it is
            called.  Defaults to a trivial `"SELECT 1;"` generator.
        """
        super().__init__()

        # --------------------- asyncio event-loop -------------------------
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # --------------------- connection pools --------------------------
        db_cfg = dict(db_cfg)
        db_cfg.setdefault("min_size", pool_size)
        db_cfg.setdefault("max_size", pool_size)
        self.pools = self._loop.run_until_complete(
            asyncio.gather(
                *[self._build_pool(h, db_cfg) for h in hosts], loop=self._loop
            )
        )
        self.num_hosts = len(self.pools)

        # --------------------- query generation --------------------------
        if query_generator is None:

            def _default_generator() -> Tuple[str, QueryType]:
                return "SELECT 1;", QueryType.SIMPLE_SELECT

            self.query_generator = _default_generator
        else:
            self.query_generator = query_generator

        # --------------------- RL bookkeeping ----------------------------
        self.history_length = history_length
        self.max_steps = max_steps
        self.steps_taken = 0

        self.pending_requests: Deque[Request] = deque()
        self.active_requests: Dict[str, float] = {}  # request_id ➔ send_time
        self._latency_queue: asyncio.Queue[Tuple[str, float]] = (
            asyncio.Queue()
        )  # (request_id, latency)

        # rolling metrics
        self.latency_hist = deque([0.0] * history_length, maxlen=history_length)
        self.decision_hist = deque([0] * history_length, maxlen=history_length)
        self.total_latency = 0.0
        self.completed_cnt = 0

        # server-utilisation placeholder – shape (N, 2)  (cpu, ram) ∈ [0,1]
        self.server_utils = np.full(
            (self.num_hosts, 2), [init_cpu_util, init_ram_util], dtype=np.float32
        )

        # --------------------- Gym spaces --------------------------------
        self.action_space = spaces.Discrete(self.num_hosts)
        self.observation_space = spaces.Dict(
            {
                "server_utils": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_hosts, 2), dtype=np.float32
                ),
                "latency_history": spaces.Box(
                    low=0.0,
                    high=np.finfo(np.float32).max,
                    shape=(history_length,),
                    dtype=np.float32,
                ),
                "decision_history": spaces.Box(
                    low=0,
                    high=self.num_hosts,
                    shape=(history_length,),
                    dtype=np.int32,
                ),
                "request_type": spaces.Box(
                    low=0,
                    high=len(QueryType) - 1,
                    shape=(),
                    dtype=np.int32,
                ),
            }
        )

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.pending_requests.clear()
        self.active_requests.clear()
        self._drain_latency_queue()

        self.latency_hist.clear()
        self.decision_hist.clear()
        self.latency_hist.extend([0.0] * self.history_length)
        self.decision_hist.extend([0] * self.history_length)

        self.total_latency = 0.0
        self.completed_cnt = 0

        # prime the first request
        self._enqueue_new_request()
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid host index"
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps

        # 1. Pop (or generate) a request
        if not self.pending_requests:
            self._enqueue_new_request()

        req = self.pending_requests.popleft()
        now_send = time.perf_counter()
        self.active_requests[req.request_id] = now_send
        self.decision_hist.append(action)

        # 2. Fire the async SQL query
        self._loop.create_task(
            send_query_to_host(
                self.pools[action],
                req.sql,
                self._latency_queue,
                req.request_id,
            )
        )

        self._loop.run_until_complete(asyncio.sleep(0))

        # 4. Harvest any completed latencies since last step
        latencies_this_step = self._drain_latency_queue()

        # 5. Dummy-update util numbers (replace with real data feed later)
        self._jitter_utils()

        # 6. Compute reward
        reward = (
            1.0 / (1.0 + np.mean(latencies_this_step)) if latencies_this_step else 0.01
        )

        # 7. Enqueue the *next* request so there is always something waiting
        self._enqueue_new_request()

        info = {
            "avg_latency": self.average_latency,
            "completed": self.completed_cnt,
            "steps": self.steps_taken,
        }
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    # -- request lifecycle ------------------------------------------------
    def _enqueue_new_request(self):
        sql, q_type = self.query_generator()
        new_req = Request(
            request_id=str(uuid.uuid4()),
            query_type=q_type,
            sql=sql,
            arrival_time=time.perf_counter(),
        )
        self.pending_requests.append(new_req)

    def _drain_latency_queue(self) -> List[float]:
        """Pull *all* latencies that have arrived since last call."""
        drained: List[float] = []
        while not self._latency_queue.empty():
            req_id, latency = self._latency_queue.get_nowait()
            drained.append(latency)
            # bookkeeping
            self.latency_hist.append(latency)
            self.total_latency += latency
            self.completed_cnt += 1
            self.active_requests.pop(req_id, None)
        return drained

    # -- observation, util, render ---------------------------------------
    def _get_obs(self) -> Dict:
        # current request type → scalar, not one-hot (simpler for agents)
        if self.pending_requests:
            rt_val = int(self.pending_requests[0].query_type)
        else:  # should not happen
            rt_val = int(QueryType.SIMPLE_SELECT)

        return {
            "server_utils": self.server_utils.copy(),
            "latency_history": np.array(self.latency_hist, dtype=np.float32),
            "decision_history": np.array(self.decision_hist, dtype=np.int32),
            "request_type": np.array(rt_val, dtype=np.int32),
        }

    def _jitter_utils(self):
        """Fake CPU/RAM utilisation with a small random walk."""
        jitter = np.random.normal(0.0, 0.02, size=self.server_utils.shape)
        self.server_utils = np.clip(self.server_utils + jitter, 0.0, 1.0)

    # -- metrics ----------------------------------------------------------
    @property
    def average_latency(self) -> float:
        return (
            0.0 if self.completed_cnt == 0 else self.total_latency / self.completed_cnt
        )

    # -- pretty print -----------------------------------------------------
    def render(self, mode: str = "human"):
        if mode != "human":
            return
        print(
            f"[t={self.steps_taken:4d}] "
            f"avg-lat={self.average_latency*1e3:7.2f} ms, "
            f"completed={self.completed_cnt}"
        )
        for i, (cpu, ram) in enumerate(self.server_utils):
            print(f"  host[{i}] : CPU={cpu*100:5.1f} %,  RAM={ram*100:5.1f} %")
