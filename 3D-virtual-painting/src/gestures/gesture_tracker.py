
import cv2
import mediapipe as mp
from collections import deque

class HandTracker:
    def __init__(self, maxHands=1, detectionConfidence=0.7, trackConfidence=0.7, smooth_factor=5):
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
            # multi_handedness gives labels corresponding to landmarks order
            if self.results.multi_handedness:
                # take the first (and ideally only) hand's label
                self.hand_label = self.results.multi_handedness[0].classification[0].label

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
        """Return a dict of fingertip positions (index, thumb, middle, ring, pinky) or None if not available.
           Only returns for the detected hand(s); use hand_label to check side."""
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape
        hand = self.results.multi_hand_landmarks[0]  # first hand
        # tips: thumb(4), index(8), middle(12), ring(16), pinky(20)
        tips_idx = {'thumb':4, 'index':8, 'middle':12, 'ring':16, 'pinky':20}
        points = {}
        for name, idx in tips_idx.items():
            lm = hand.landmark[idx]
            x, y = self._landmark_to_point(lm, w, h)
            points[name] = self._smooth_point(x, y) if name == 'index' else (x, y)
        return points

    def fingers_up(self, frame):
        """Return list/tuple of booleans for fingers [thumb, index, middle, ring, pinky] True=up.
           Uses simple y-position test (tip y < pip y) for index->pinky; thumb uses x for right/left hand."""
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape
        hand = self.results.multi_hand_landmarks[0]

        # landmarks indexes:
        # thumb: tip=4, ip=3
        # index: tip=8, pip=6
        # middle: tip=12, pip=10
        # ring: tip=16, pip=14
        # pinky: tip=20, pip=18
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        fingers_status = []
        for t, p in zip(tips, pips):
            tip = hand.landmark[t]
            pip = hand.landmark[p]
            # For fingers other than thumb: tip.y < pip.y -> finger up (note: origin top-left)
            if t != 4:
                fingers_status.append(tip.y < pip.y)
            else:
                # thumb: use x comparison; for Right hand thumb.x < ip.x means thumb is to left (up)
                if self.hand_label == 'Right':
                    fingers_status.append(tip.x < pip.x)
                else:
                    fingers_status.append(tip.x > pip.x)
        # returns [thumb_up, index_up, middle_up, ring_up, pinky_up]
        return fingers_status
