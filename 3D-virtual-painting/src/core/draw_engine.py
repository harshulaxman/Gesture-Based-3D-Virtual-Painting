# src/core/draw_engine.py
import cv2

class DrawEngine:
    def __init__(self, stroke_thickness=5):
        self.strokes = [[]]  # list of strokes, each stroke is a list of (x,y)
        self.stroke_thickness = stroke_thickness

    def update(self, point, mode):
        # ensure safety
        if len(self.strokes) == 0:
            self.strokes.append([])

        if mode == "DRAW":
            if point:
                self.strokes[-1].append(point)
        elif mode == "STOP":
            if len(self.strokes[-1]) > 0:
                self.strokes.append([])
        elif mode == "ERASE":
            # undo last non-empty stroke
            for i in range(len(self.strokes)-1, -1, -1):
                if len(self.strokes[i]) > 0:
                    self.strokes.pop(i)
                    break

    def draw(self, frame):
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                cv2.line(frame, stroke[i-1], stroke[i], (0, 0, 255),
                         self.stroke_thickness, lineType=cv2.LINE_AA)
        return frame

    def clear(self):
        self.strokes = [[]]
