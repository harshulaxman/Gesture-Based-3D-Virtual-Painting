# tests/test_gestures.py
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import cv2
from gestures.gesture_tracker import HandTracker
from gestures.gesture_utils import GestureUtils
from core.draw_engine import DrawEngine
from core.controller import GestureController

# Initialize
cap = cv2.VideoCapture(0)
tracker = HandTracker(maxHands=1, detectionConfidence=0.6, trackConfidence=0.6, smooth_factor=5)
utils = GestureUtils()
drawer = DrawEngine(stroke_thickness=5)
controller = GestureController()

# Debounce & cooldown timers
fist_hold_start = None
fist_required_hold = 0.4  # seconds required to confirm a fist -> UNDO
erase_cooldown = 0.8
next_allowed_erase_time = 0.0

pinch_hold_start = None
pinch_required_hold = 0.05  # small hold to stabilize pinch detection

def draw_ui(frame, mode):
    # Mode label
    cv2.putText(frame, f"Mode: {mode}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
    # Tips
    tips = [
        "Draw: Pinch index + thumb",
        "Undo (Erase last): Make a tight fist and hold 0.4s",
        "Stop: Open hand (3+ fingers up)",
        "Clear: Press 'c'"
    ]
    y = 80
    for t in tips:
        cv2.putText(frame, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        y += 25

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = tracker.findHands(frame, draw=True)
    pts = tracker.get_finger_positions(frame)  # dict or None
    fingers = tracker.fingers_up(frame)  # [thumb,index,middle,ring,pinky] or None

    index_pos = pts.get('index') if pts else None
    thumb_pos = pts.get('thumb') if pts else None

    now = time.time()

    # Pinch (draw) detection (requires small stable hold)
    is_pinch = False
    if index_pos and thumb_pos:
        if utils.is_pinch(index_pos, thumb_pos, threshold=45):
            if pinch_hold_start is None:
                pinch_hold_start = now
            elif (now - pinch_hold_start) >= pinch_required_hold:
                is_pinch = True
        else:
            pinch_hold_start = None
    else:
        pinch_hold_start = None

    # Fist detection requires holding (to avoid accidental triggers)
    is_fist = False
    if fingers is not None:
        if utils.is_fist(fingers, max_fingers_up=0):
            if fist_hold_start is None:
                fist_hold_start = now
            elif (now - fist_hold_start) >= fist_required_hold and now >= next_allowed_erase_time:
                is_fist = True
                next_allowed_erase_time = now + erase_cooldown
                fist_hold_start = None  # reset
        else:
            fist_hold_start = None

    # Map gestures (priority in controller)
    mode = controller.update_mode(is_pinch, is_fist, fingers)


    # Update drawing engine
    if mode == "ERASE" and mode != controller.mode:
        # handled by controller priority already; actual erase action is performed by update call
        pass

    if mode == "DRAW":
        drawer.update(index_pos, "DRAW")
    elif mode == "STOP":
        drawer.update(None, "STOP")
    elif mode == "ERASE":
        drawer.update(None, "ERASE")

    frame = drawer.draw(frame)
    draw_ui(frame, mode)

    cv2.imshow("Gesture Painter - Stable (Day3)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc
        break
    elif key == ord('c'):
        drawer.clear()

cap.release()
cv2.destroyAllWindows()
