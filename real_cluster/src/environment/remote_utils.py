"""
Tiny async helpers shared by real-mode classes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Tuple

import asyncpg
from ..config import EXCEPTION_PENALTY

async def send_query_to_host(
    pool: asyncpg.pool.Pool,
    sql: str,
    latency_queue: "asyncio.Queue[Tuple[str, float]]",
    request_id: str,
):
    """
    Fire *one* SQL string against *one* host and push its latency when done.

    This helper never raises, the environment treats failures as really high latency.
    """
    try:
        t0 = time.perf_counter()
        async with pool.acquire() as conn:
            await conn.execute(sql)
            latency = time.perf_counter() - t0

    except Exception as exc:
        print(f"[WARN] {request_id}: query failed - {exc}")
        latency = EXCEPTION_PENALTY  # 600 s penalty
    finally:
        # print(f"[INFO] {request_id}: query done - {latency:.2f} s")
        # Queue is *thread-safe* for asyncio tasks in the same loop.
        await latency_queue.put((request_id, latency))
