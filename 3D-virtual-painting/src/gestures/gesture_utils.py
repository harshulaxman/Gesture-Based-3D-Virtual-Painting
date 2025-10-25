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
        return GestureUtils.distance(index, thumb) < threshold

    @staticmethod
    def is_fist(fingers_status, min_fingers_up=1):
        # fist means few or zero fingers up; count up fingers
        if fingers_status is None:
            return False
        return sum(1 for f in fingers_status if f) <= min_fingers_up  # allow tiny tolerance
