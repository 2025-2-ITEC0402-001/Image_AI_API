import asyncio
from fastapi import FastAPI

app = FastAPI()

# CPU를 많이 사용하는 무거운 작업을 흉내 내는 함수
def cpu_bound_task():
    total = 0
    for i in range(30_000_000):
        total += i

@app.get("/test")
async def cpu_test():
    # CPU-bound 작업을 비동기에서 올바르게 처리하려면, 별도의 스레드에서 실행해야 함
    await asyncio.to_thread(cpu_bound_task)
    return {"message": "FastAPI CPU Task OK"}