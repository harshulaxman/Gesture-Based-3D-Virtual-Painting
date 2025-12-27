import asyncio
import websockets
import json

UNITY_URL = "ws://localhost:8080"

async def send_position_to_unity(x, y, z):
    async with websockets.connect(UNITY_URL) as ws:
        data = {"x": x, "y": y, "z": z}
        await ws.send(json.dumps(data))
