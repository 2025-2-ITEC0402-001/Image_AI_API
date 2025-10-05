import httpx
import asyncio
import time
import sys

async def fetch(client, url):
    try:
        await client.get(url, timeout=120.0)
        return 1
    except Exception:
        return 0

async def run_test(url, total_requests, concurrency):
    print(f"\n--- Testing {url} ---")
    print(f"Total Requests: {total_requests}, Concurrency: {concurrency}")

    start_time = time.time()
    tasks = []
    # Create all client sessions and tasks at once
    async with httpx.AsyncClient() as client:
        for _ in range(total_requests):
            tasks.append(fetch(client, url))

        successful_requests = 0
        # Run tasks in concurrent batches
        for i in range(0, total_requests, concurrency):
            batch = tasks[i:i+concurrency]
            results = await asyncio.gather(*batch)
            successful_requests += sum(results)

    total_time = time.time() - start_time
    rps = successful_requests / total_time if total_time > 0 else 0

    print(f"\n--- Results ---")
    print(f"Total time taken:    {total_time:.2f} seconds")
    print(f"Requests per second: {rps:.2f}")
    print("-" * 25)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_test.py <URL>")
        sys.exit(1)

    target_url = sys.argv[1]
    asyncio.run(run_test(url=target_url, total_requests=100, concurrency=10))