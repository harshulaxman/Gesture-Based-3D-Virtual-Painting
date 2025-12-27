import asyncio
import websockets
import json
import threading

class UnityBridge:
    def __init__(self, uri="ws://localhost:8080"):
        self.uri = uri
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.start_loop, daemon=True)
        self.thread.start()
        self.websocket = None

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.connect())

    async def connect(self):
        try:
            print(f"[UNITY BRIDGE] Connecting to {self.uri} ...")
            self.websocket = await websockets.connect(self.uri)
            print("[UNITY BRIDGE] Connected!")
        except Exception as e:
            print("[UNITY BRIDGE] Connection failed:", e)

    def send(self, data: dict):
        if not self.websocket:
            return
        try:
            json_data = json.dumps(data)
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json_data), self.loop
            )
        except Exception as e:
            print("[UNITY BRIDGE] Send Error:", e)
