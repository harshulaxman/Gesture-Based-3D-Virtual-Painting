import asyncio
import websockets
import json

clients = set()

async def handler(websocket):
    print("[PYTHON] Unity connected")
    clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        clients.remove(websocket)

async def broadcast(data):
    if clients:
        msg = json.dumps(data)
        await asyncio.gather(*(client.send(msg) for client in clients))

async def main():
    print("[PYTHON] WebSocket server running at ws://localhost:8080")
    async with websockets.serve(handler, "localhost", 8080):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
