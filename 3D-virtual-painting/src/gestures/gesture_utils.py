import math

class GestureUtils:
    """
    Utility class for gesture calculations (distances, angles, etc.)
    Used by gesture_paint.py for pinch distance and gesture recognition.
    """

    @staticmethod
    def distance(p1, p2):
        """
        Calculates Euclidean distance between two points (x1, y1) and (x2, y2).
        Args:
            p1 (tuple): (x1, y1)
            p2 (tuple): (x2, y2)
        Returns:
            float: distance between points
        """
        if p1 is None or p2 is None:
            return 9999  # default large value if points missing
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
