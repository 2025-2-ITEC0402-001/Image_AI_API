import asyncio
from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
async def async_test():
    await asyncio.sleep(1)
    return {"message": "FastAPI OK"}
