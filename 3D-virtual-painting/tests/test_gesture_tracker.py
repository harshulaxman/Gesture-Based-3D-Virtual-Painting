# tests/test_gestures.py
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import cv2
from gestures.gesture_tracker import HandTracker
from gestures.gesture_utils import GestureUtils
from core.draw_engine import DrawEngine
from core.controller import GestureController

cap = cv2.VideoCapture(0)
tracker = HandTracker(maxHands=1, detectionConfidence=0.6, trackConfidence=0.6, smooth_factor=5)
utils = GestureUtils()
drawer = DrawEngine(stroke_thickness=5)
controller = GestureController()

last_mode = None
cooldown_until = 0.0

def draw_ui(frame, mode):
    # Mode label
    cv2.putText(frame, f"Mode: {mode}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
    # Tips
    tips = [
        "Draw: Pinch index + thumb",
        "Undo (Erase last): Make a fist",
        "Stop: Open hand",
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

    # detect draw pinch (index+thumb)
    draw_gesture = utils.is_pinch(index_pos, thumb_pos, threshold=45) if index_pos and thumb_pos else False

    # detect fist (few fingers up) -> erase/undo
    erase_gesture = utils.is_fist(fingers, min_fingers_up=0) if fingers is not None else False

    # basic cooldown to avoid repeated accidental triggers for erase
    now = time.time()
    if erase_gesture and now < cooldown_until:
        # ignore erase during cooldown
        erase_gesture = False

    mode = controller.update_mode(draw_gesture, erase_gesture)

    # whenever a new erase action occurs, set small cooldown and do update
    if mode == "ERASE" and (last_mode != "ERASE"):
        drawer.update(None, "ERASE")
        cooldown_until = time.time() + 0.8  # 0.8s cooldown
    elif mode == "DRAW":
        drawer.update(index_pos, "DRAW")
    else:
        drawer.update(None, "STOP")

    last_mode = mode
    frame = drawer.draw(frame)
    draw_ui(frame, mode)

    cv2.imshow("Gesture Painter - Day3 Stable", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        drawer.clear()

cap.release()
cv2.destroyAllWindows()
