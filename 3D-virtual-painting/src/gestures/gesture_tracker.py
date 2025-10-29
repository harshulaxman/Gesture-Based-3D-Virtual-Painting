import cv2
import mediapipe as mp

class HandTracker:
    """
    Tracks a single hand and provides finger landmark positions
    for gesture-based controls (draw, erase, undo, hover, etc.)
    """

    def __init__(self, maxHands=1, detectionConfidence=0.7, trackConfidence=0.6, smooth_factor=5):
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.smooth_factor = smooth_factor

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.hand_label = None

    def findHands(self, frame, draw=True):
        """
        Processes the frame and returns it with optional landmarks drawn.
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
            # Use first hand only
            handedness = self.results.multi_handedness[0].classification[0].label
            self.hand_label = handedness  # "Left" or "Right"
        else:
            self.hand_label = None

        return frame

    def get_finger_positions(self, frame):
        """
        Returns a dictionary of key finger landmark coordinates.
        Keys: thumb, index, middle, ring, pinky, wrist
        """
        lm_positions = {}
        if not self.results or not self.results.multi_hand_landmarks:
            return lm_positions

        h, w, _ = frame.shape
        handLms = self.results.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if id == 4: lm_positions["thumb"] = (cx, cy)
            elif id == 8: lm_positions["index"] = (cx, cy)
            elif id == 12: lm_positions["middle"] = (cx, cy)
            elif id == 16: lm_positions["ring"] = (cx, cy)
            elif id == 20: lm_positions["pinky"] = (cx, cy)
            elif id == 0: lm_positions["wrist"] = (cx, cy)
        return lm_positions

    def fingers_up(self, frame):
        """
        Detects which fingers are raised.
        Returns a list of 5 values [thumb, index, middle, ring, pinky]
        where 1 = finger up, 0 = down.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return [0, 0, 0, 0, 0]

        handLms = self.results.multi_hand_landmarks[0]
        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb (depends on hand orientation)
        if self.hand_label == "Right":
            fingers.append(1 if handLms.landmark[tips[0]].x < handLms.landmark[tips[0]-1].x else 0)
        else:
            fingers.append(1 if handLms.landmark[tips[0]].x > handLms.landmark[tips[0]-1].x else 0)

        # Other four fingers
        for tipId in tips[1:]:
            fingers.append(1 if handLms.landmark[tipId].y < handLms.landmark[tipId-2].y else 0)

        return fingers
