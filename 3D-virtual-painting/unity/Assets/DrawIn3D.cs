using System.Collections.Generic;
using UnityEngine;

public class DrawIn3D : MonoBehaviour
{
    public Transform brushTip;              // Reference to your sphere
    public Material lineMaterial;           // Material for drawn lines
    public float lineWidth = 0.02f;         // Thickness of lines
    public float minDistance = 0.01f;       // Minimum movement before adding a new point
    public int roundness = 10;              // Must be INT for LineRenderer

    private LineRenderer currentLine;
    private List<Vector3> points = new List<Vector3>();
    private bool isDrawing = false;

    void Update()
    {
        if (isDrawing)
        {
            Draw();
        }
    }

    // Called when hand gesture = DRAW
    public void StartDrawing()
    {
        if (currentLine != null) return;

        GameObject lineObj = new GameObject("DrawnLine");
        currentLine = lineObj.AddComponent<LineRenderer>();

        currentLine.material = lineMaterial;
        currentLine.startWidth = lineWidth;
        currentLine.endWidth = lineWidth;
        currentLine.positionCount = 0;

        // Smooth round line caps
        currentLine.numCapVertices = roundness;
        currentLine.numCornerVertices = roundness;

        points.Clear();
        AddPoint(brushTip.position);
        isDrawing = true;
    }

    // Called when hand gesture = STOP
    public void StopDrawing()
    {
        isDrawing = false;
        currentLine = null;
    }

    void Draw()
    {
        if (points.Count == 0) return;

        Vector3 lastPoint = points[points.Count - 1];
        float distance = Vector3.Distance(lastPoint, brushTip.position);

        if (distance > minDistance)
        {
            AddPoint(brushTip.position);
        }
    }

    void AddPoint(Vector3 point)
    {
        points.Add(point);
        currentLine.positionCount = points.Count;
        currentLine.SetPosition(points.Count - 1, point);
    }
}
