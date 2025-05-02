import asyncio
import asyncpg


async def ping_postgres():
    try:
        db_config = {
            "user": "postgres",
            "password": "2241proj",
            "database": "postgres",
            "host": "ec2-3-148-166-172.us-east-2.compute.amazonaws.com",
            "port": 5432,
        }
        conn = await asyncpg.connect(**db_config)
        result = await conn.fetchval("SELECT 1;")
        print(f"Ping successful, result: {result}")
        await conn.close()
    except Exception as e:
        print(f"Ping failed: {e}")


asyncio.run(ping_postgres())
