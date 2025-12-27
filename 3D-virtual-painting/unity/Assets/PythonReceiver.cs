using UnityEngine;
using System;
using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class PythonReceiver : MonoBehaviour
{
    public DrawIn3D drawSystem;
    public Transform brushTip;

    private ClientWebSocket socket;

    async void Start()
    {
        await Connect();
    }

    async Task Connect()
    {
        try
        {
            socket = new ClientWebSocket();
            Uri uri = new Uri("ws://localhost:8080");
            await socket.ConnectAsync(uri, CancellationToken.None);
            Debug.Log("[UNITY] Connected to Python WebSocket");

            await ReceiveLoop();
        }
        catch (Exception e)
        {
            Debug.LogError("[UNITY] Connection failed: " + e.Message);
        }
    }

    async Task ReceiveLoop()
    {
        byte[] buffer = new byte[1024];

        while (socket.State == WebSocketState.Open)
        {
            var result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);

            if (result.MessageType == WebSocketMessageType.Text)
            {
                string json = Encoding.UTF8.GetString(buffer, 0, result.Count);
                HandleMessage(json);
            }
        }
    }

    void HandleMessage(string json)
    {
        try
        {
            GestureData data = JsonConvert.DeserializeObject<GestureData>(json);

            // Move brush tip
            Vector3 worldPos = MapToWorld(data.x, data.y, data.z);
            brushTip.position = worldPos;

            // Drawing logic
            if (data.draw)
                drawSystem.StartDrawing();
            else
                drawSystem.StopDrawing();
        }
        catch (Exception e)
        {
            Debug.LogWarning("JSON parse error: " + e.Message);
        }
    }

    Vector3 MapToWorld(float x, float y, float z)
    {
        float wx = Mathf.Lerp(-5f, 5f, x / 1280f);
        float wy = Mathf.Lerp(-3f, 3f, y / 720f);
        return new Vector3(wx, wy, z);
    }
}

[Serializable]
public class GestureData
{
    public float x;
    public float y;
    public float z;
    public bool draw;
    public bool erase;
    public bool fist;
}
