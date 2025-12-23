using System.Collections.Generic;
using UnityEngine;

public class BrushManager : MonoBehaviour
{
    public GameObject brushPrefab;   // The sphere or 3D brush tip
    public float brushSize = 0.05f;  // Size of the 3D point

    private List<GameObject> strokes = new List<GameObject>();

    // Called from Python → Unity link
    public void DrawPoint(float x, float y)
    {
        // Convert 2D normalized screen coords → 3D world position
        Vector3 screenPos = new Vector3(x * Screen.width, y * Screen.height, 1.5f);
        Vector3 worldPos = Camera.main.ScreenToWorldPoint(screenPos);

        GameObject point = Instantiate(brushPrefab, worldPos, Quaternion.identity);
        point.transform.localScale = Vector3.one * brushSize;

        strokes.Add(point);
    }

    public void ClearCanvas()
    {
        foreach (GameObject obj in strokes)
            Destroy(obj);

        strokes.Clear();
    }
}
