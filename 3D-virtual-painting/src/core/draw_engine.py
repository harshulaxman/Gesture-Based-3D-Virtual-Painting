# src/core/draw_engine.py
import cv2
import numpy as np

class DrawEngine:
    def __init__(self, stroke_thickness=6, stroke_color=(255, 0, 0)):
        self.strokes = [[]]  # list of strokes [(x,y,color,thickness), ...]
        self.stroke_thickness = stroke_thickness
        self.stroke_color = stroke_color

    def update(self, point, mode):
        if len(self.strokes) == 0:
            self.strokes.append([])

        if mode == "DRAW" and point:
            # Append point with color and thickness
            self.strokes[-1].append((point[0], point[1], self.stroke_color, self.stroke_thickness))

        elif mode == "STOP":
            if len(self.strokes[-1]) > 0:
                self.strokes.append([])

        elif mode == "ERASE":
            # Undo last stroke
            for i in range(len(self.strokes) - 1, -1, -1):
                if len(self.strokes[i]) > 0:
                    self.strokes.pop(i)
                    break

    def change_color(self, new_color):
        """Update brush color"""
        self.stroke_color = new_color

    def clear(self):
        """Clear all strokes"""
        self.strokes = [[]]

    def draw(self, frame):
        """Smooth drawing with soft edges"""
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                x1, y1, color1, thick1 = stroke[i - 1]
                x2, y2, color2, thick2 = stroke[i]
                cv2.line(frame, (x1, y1), (x2, y2), color1, thick1, lineType=cv2.LINE_AA)

                # Soft brush glow effect
                overlay = frame.copy()
                cv2.line(overlay, (x1, y1), (x2, y2), color1, thick1 + 8)
                alpha = 0.08  # transparency
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame
