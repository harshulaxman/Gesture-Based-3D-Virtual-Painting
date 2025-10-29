# src/core/draw_engine.py
import cv2
import numpy as np

class DrawEngine:
    def __init__(self, stroke_thickness=6, stroke_color=(255, 0, 0)):
        # Vector stroke history (for undo)
        self.strokes = [[]]                   # list[list[(x,y,color,thick)]]
        # Current brush
        self.stroke_thickness = stroke_thickness
        self.stroke_color = stroke_color
        # Raster paint layer (where we actually paint pixels)
        self.layer = None                     # np.uint8 HxWx3

    # ---------- internal helpers ----------
    def _ensure_layer(self, frame):
        if self.layer is None or self.layer.shape != frame.shape:
            self.layer = np.zeros_like(frame)  # black = no paint

    def _draw_segment_on_layer(self, p1, p2, color, thick):
        # main stroke
        cv2.line(self.layer, p1, p2, color, thick, lineType=cv2.LINE_AA)
        # soft edge glow (subtle)
        overlay = self.layer.copy()
        cv2.line(overlay, p1, p2, color, thick + 8, lineType=cv2.LINE_AA)
        self.layer = cv2.addWeighted(overlay, 0.08, self.layer, 0.92, 0)

    # ---------- public API ----------
    def update(self, point, mode):
        """
        mode: "DRAW" appends a point and paints on the layer
              "STOP" closes current stroke (starts a new one next time)
              "ERASE" undoes last stroke (vector undo)
        """
        if mode == "DRAW" and point:
            x, y = int(point[0]), int(point[1])
            # ensure a stroke exists
            if len(self.strokes) == 0:
                self.strokes = [[]]
            # append to current stroke
            self.strokes[-1].append((x, y, self.stroke_color, self.stroke_thickness))
            # also paint incrementally on raster layer
            if len(self.strokes[-1]) >= 2:
                (x1, y1, c1, t1) = self.strokes[-1][-2]
                (x2, y2, c2, t2) = self.strokes[-1][-1]
                self._draw_segment_on_layer((x1, y1), (x2, y2), c1, t1)

        elif mode == "STOP":
            if len(self.strokes) == 0 or len(self.strokes[-1]) > 0:
                self.strokes.append([])

        elif mode == "ERASE":
            # vector undo: remove last non-empty stroke and rebuild layer
            for i in range(len(self.strokes) - 1, -1, -1):
                if len(self.strokes[i]) > 0:
                    self.strokes.pop(i)
                    break
            self._rebuild_layer()  # redraw all remaining strokes onto layer

    def _rebuild_layer(self):
        if self.layer is None:
            return
        self.layer[:] = 0
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                x1, y1, col, th = stroke[i-1]
                x2, y2, _,  _   = stroke[i]
                self._draw_segment_on_layer((x1, y1), (x2, y2), col, th)

    def change_color(self, new_color):
        self.stroke_color = new_color

    def clear(self):
        self.strokes = [[]]
        if self.layer is not None:
            self.layer[:] = 0

    # >>> NEW: precise pixel eraser <<<
    def erase_at(self, center, radius=20):
        """
        Erase by punching a hole in the raster paint layer.
        Vector strokes remain (so undo still works for full strokes),
        but erased pixels disappear immediately (precise).
        """
        if center and self.layer is not None:
            x, y = int(center[0]), int(center[1])
            cv2.circle(self.layer, (x, y), int(radius), (0, 0, 0), -1)

    def draw(self, frame):
        """
        Composite paint layer over camera frame.
        """
        self._ensure_layer(frame)
        # simple additive composite (black adds nothing)
        out = cv2.addWeighted(frame, 1.0, self.layer, 1.0, 0)
        return out
