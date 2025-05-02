HOSTS = [
    # "ec2-3-148-166-172.us-east-2.compute.amazonaws.com",
    # "ec2-3-91-21-145.us-east-2.compute.amazonaws.com",
    # "ec2-18-191-234-182.us-east-2.compute.amazonaws.com",
    # "ec2-3-128-122-76.us-east-2.compute.amazonaws.com",
]
import asyncio
import asyncpg
import time


async def send_query(pool, query, latencies, idx):
    conn = await pool.acquire()
    print(f"Connection acquired for {idx}")
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
            "min_size": 100,
            "max_size": 100,
        }
        pools.append(await asyncpg.create_pool(**db_config))

    num_requests = 1000
    queries = ["SELECT 1"] * num_requests  # simple query repeated
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
