import cv2

class DrawEngine:
    def __init__(self, stroke_thickness=5):
        self.strokes = [[]]  # list of strokes, each stroke is a list of points
        self.stroke_thickness = stroke_thickness

    def update(self, point, mode):
        """point: (x,y) or None ; mode: 'DRAW'|'STOP'|'ERASE'"""
        # ✅ Always ensure at least one stroke exists
        if len(self.strokes) == 0:
            self.strokes.append([])

        if mode == "DRAW":
            if point:
                self.strokes[-1].append(point)

        elif mode == "STOP":
            # ✅ Start a new stroke ONLY if current one has points
            if len(self.strokes[-1]) > 0:
                self.strokes.append([])

        elif mode == "ERASE":
            # ✅ Undo last stroke safely
            if len(self.strokes) > 1:
                self.strokes.pop()
            else:
                self.strokes[0] = []  # Reset the stroke instead of deleting it

    def draw(self, frame):
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                cv2.line(frame, stroke[i - 1], stroke[i], (0, 0, 255),
                         self.stroke_thickness, lineType=cv2.LINE_AA)
        return frame

    def clear(self):
        self.strokes = [[]]  # ✅ Reset safely
