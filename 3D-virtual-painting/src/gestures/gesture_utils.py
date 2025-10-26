# src/gestures/gesture_utils.py
import math

class GestureUtils:
    @staticmethod
    def distance(p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def is_pinch(index, thumb, threshold=45):
        """True when index and thumb are close enough (pixels)"""
        return GestureUtils.distance(index, thumb) < threshold

    @staticmethod
    def is_fist(fingers_status, max_fingers_up=0):
        """
        fingers_status: [thumb, index, middle, ring, pinky] booleans
        If number of fingers up <= max_fingers_up, consider it a fist.
        """
        if fingers_status is None:
            return False
        up_count = sum(1 for f in fingers_status if f)
        return up_count <= max_fingers_up
