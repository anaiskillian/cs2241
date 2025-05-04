# This file is only relevant for knowing how the HOSTS array is structured.

HOSTS = [
    # "ec2-18-116-13-179.us-east-2.compute.amazonaws.com", # these are invalid.
    # "ec2-3-91-21-145.us-east-2.compute.amazonaws.com",
    # "ec2-18-191-234-182.us-east-2.compute.amazonaws.com",
    # "ec2-3-128-122-76.us-east-2.compute.amazonaws.com",
]
import asyncio
import asyncpg
import time


async def send_query(pool, query, latencies, idx):
    conn = await pool.acquire()
    try:
        start_time = time.perf_counter()  # Record send time
        await conn.execute(query)  # Send and wait for completion
        end_time = time.perf_counter()  # Record receive time
        latencies[idx] = end_time - start_time  # Store latency
    finally:
        print(f"Request {idx} is done.")
        await pool.release(conn)


async def main():
    pools = []
    for host in HOSTS:
        db_config = {
            "user": "postgres",
            "password": "2241proj",
            "database": "postgres",
            "host": host,
            "port": 5432,
            "min_size": 99,
            "max_size": 99,
        }
        pools.append(await asyncpg.create_pool(**db_config))

    num_requests = 1
    queries = [
        "SELECT col3, col1, SUM(col1) OVER (ORDER BY col3) AS running_total FROM tbl LIMIT 126;"
    ] * num_requests  # simple query repeated

    latencies = [0.0] * num_requests  # preallocate for latencies

    # Launch all requests asynchronously, round-robin assign to pools
    tasks = [
        asyncio.create_task(send_query(pools[i % len(pools)], queries[i], latencies, i))
        for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)

    for pool in pools:
        await pool.close()

    # After gathering all latencies
    for i, latency in enumerate(latencies):
        print(f"Request {i}: {latency * 1000:.2f} ms")

    print(f"Average latency: {sum(latencies)/len(latencies) * 1000:.2f} ms")


# Actually run it
if __name__ == "__main__":
    asyncio.run(main())
