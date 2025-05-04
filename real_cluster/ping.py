import asyncio
import asyncpg

from src.config import HOSTS


async def ping_postgres():
    for host in HOSTS:
        db_config = {
            "user": "postgres",
            "password": "2241proj",
            "database": "postgres",
            "host": host,
            "port": 5432,
        }
        try:
            conn = await asyncpg.connect(**db_config)
            result = await conn.fetchval("SELECT 1;")
            print(f"Ping successful for {host}, result: {result}")
            await conn.close()
        except Exception as e:
            print(f"Ping failed for {host}: {e}")


asyncio.run(ping_postgres())
