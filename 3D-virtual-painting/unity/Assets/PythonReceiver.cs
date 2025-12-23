using UnityEngine;
using System;
using System.Text;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;

public class PythonReceiver : MonoBehaviour
{
    public DrawIn3D drawSystem;
    public Transform brushTip;
    private Vector3 smoothPosition;
    public float smoothSpeed = 12f;   // 
    ClientWebSocket ws;

    async void Start()
    {
        ws = new ClientWebSocket();
        Uri uri = new Uri("ws://localhost:8080/ws");
        smoothPosition = brushTip.position;

        try
        {
            await ws.ConnectAsync(uri, CancellationToken.None);
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

        while (ws.State == WebSocketState.Open)
        {
            var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
            string json = Encoding.UTF8.GetString(buffer, 0, result.Count);

            // Parse JSON
            GestureData data = JsonUtility.FromJson<GestureData>(json);

            // ✅ Debug log (correct scope)
            Debug.Log(
                $"[DATA] x={data.x}, y={data.y}, z={data.z}, draw={data.draw}, fist={data.fist}"
            );

            // Map to world space
            Vector3 targetPos = MapToWorld(x, y, z);

            smoothPosition = Vector3.Lerp(
                brushTip.position,
                targetPos,
                Time.deltaTime * smoothSpeed
            );

            brushTip.position = smoothPosition;


            // Gesture logic
            if (data.fist)
            {
                drawSystem.StopDrawing();
            }
            else if (data.draw)
            {
                drawSystem.StartDrawing();
            }
            else
            {
                drawSystem.StopDrawing();
            }

                    }
                }
        Vector3 MapToWorld(float x, float y, float z)
        {
            // Camera resolution
            float camW = 1280f;
            float camH = 720f;

            // Normalize to -0.5 → +0.5
            float nx = (x / camW) - 0.5f;
            float ny = (y / camH) - 0.5f;

            // Scale for Unity world
            return new Vector3(
                nx * 4f,      // X range
                -ny * 3f,     // Y inverted
                z
            );
        }


}

[Serializable]
public class GestureData
{
    public float x, y, z;
    public bool draw;
    public bool erase;
    public bool fist;
}
