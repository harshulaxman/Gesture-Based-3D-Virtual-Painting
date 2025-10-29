# src/gestures/gesture_tracker.py
import cv2
import mediapipe as mp
from collections import deque

class HandTracker:
    def __init__(self, maxHands=1, detectionConfidence=0.6, trackConfidence=0.6, smooth_factor=5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=maxHands,
            min_detection_confidence=detectionConfidence,
            min_tracking_confidence=trackConfidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.smooth_factor = smooth_factor
        self.prev_points = deque(maxlen=smooth_factor)
        self.results = None
        self.hand_label = None  # 'Left' or 'Right'

    def findHands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        self.hand_label = None
        if self.results.multi_hand_landmarks:
            if self.results.multi_handedness:
                # Use first hand's handedness label
                try:
                    self.hand_label = self.results.multi_handedness[0].classification[0].label
                except Exception:
                    self.hand_label = None
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def _smooth_point(self, x, y):
        self.prev_points.append((x, y))
        sx = int(sum(p[0] for p in self.prev_points) / len(self.prev_points))
        sy = int(sum(p[1] for p in self.prev_points) / len(self.prev_points))
        return sx, sy

    def _landmark_to_point(self, lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def get_finger_positions(self, frame):
        """
        Returns dict: {'thumb':(x,y), 'index':(x,y), 'middle':..., 'ring':..., 'pinky':...}
        or None if no hand detected.
        Note: index gets smoothing applied, others are raw.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape
        hand = self.results.multi_hand_landmarks[0]
        tips_idx = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'pinky':20}
        points = {}
        for name, idx in tips_idx.items():
            lm = hand.landmark[idx]
            x, y = self._landmark_to_point(lm, w, h)
            if name == 'index':
                points[name] = self._smooth_point(x, y)
            else:
                points[name] = (x, y)
        return points

    def fingers_up(self, frame):
        """
        Returns list [thumb_up, index_up, middle_up, ring_up, pinky_up] or None
        Using tip vs pip positions (index->pinky). Thumb uses x relative to ip depending on handedness.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        hand = self.results.multi_hand_landmarks[0]
        # tip and pip indices
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        status = []
        for t, p in zip(tips, pips):
            tip = hand.landmark[t]
            pip = hand.landmark[p]
            if t != 4:
                status.append(tip.y < pip.y)  # True if tip is above pip (finger up)
            else:
                # thumb: for right hand, tip.x < ip.x when thumb is open to left side of image
                if self.hand_label == 'Right':
                    status.append(tip.x < pip.x)
                elif self.hand_label == 'Left':
                    status.append(tip.x > pip.x)
                else:
                    # unknown handedness -> fallback to comparing x with ip
                    status.append(tip.x < pip.x)
        return status
