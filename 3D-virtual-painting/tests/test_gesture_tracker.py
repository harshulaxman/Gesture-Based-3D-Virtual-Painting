"""
tests/gesture_paint.py  (DEBUG-FRIENDLY VERSION)
Replaces the earlier version with added debug overlay and window-close handling.
Copy-paste this entire file and run: python tests/gesture_paint.py
"""

import os, sys, time, math
from collections import deque
from datetime import datetime

# reduce noisy logs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import cv2
    import numpy as np
except Exception as e:
    print("ERROR: missing opencv-python or numpy. Install with: pip install opencv-python numpy")
    raise e

# ensure src on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# safe imports for project modules
try:
    from gestures.gesture_tracker import HandTracker
    from gestures.gesture_utils import GestureUtils
    from core.draw_engine import DrawEngine
    from core.controller import GestureController
except Exception as e:
    print("ERROR importing project modules. Ensure src/gestures/gesture_tracker.py and src/core/draw_engine.py exist.")
    raise e

# ---------------- CONFIG ----------------
CAM_INDEX = 0
WIN_W, WIN_H = 1280, 720
BAR_H = 90
HOVER_DELAY = 0.5
FIST_HOLD = 0.4
ERASE_COOLDOWN = 0.8
PINCH_HOLD = 0.05
POINTER_RADIUS = 7
HELP_DURATION = 3.0
SAFE_MODE = True

PALETTE = [
    (0,0,255), (255,0,0), (0,255,0), (0,255,255),
    (255,255,255), (255,0,255), (0,128,255), (42,42,165)
]
COLOR_NAMES = ["Red","Blue","Green","Yellow","White","Purple","Orange","Brown"]

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- helpers ----------------
def toolbar_rects(win_w=WIN_W, win_h=WIN_H, bar_h=BAR_H):
    rects = []
    padding_x = 30
    spacing = (win_w - 2 * padding_x) // 12
    x = padding_x + 10
    y1 = win_h - bar_h + 12
    y2 = win_h - 18
    for i in range(8):
        rects.append(((x,y1),(x+48,y2)))
        x += spacing
    rects.append(((win_w-210,y1),(win_w-150,y2)))
    rects.append(((win_w-140,y1),(win_w-80,y2)))
    rects.append(((win_w-60,y1),(win_w-20,y2)))
    return rects

def point_in_rect(x,y,rect):
    (x1,y1),(x2,y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

# toolbar icon drawing (minimal)
def draw_minimal_icons(frame, rects, current_color, active_tool, highlight_index=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, WIN_H - BAR_H), (WIN_W, WIN_H), (40,40,40), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    # colors
    for i, ((x1,y1),(x2,y2)) in enumerate(rects[:8]):
        col = PALETTE[i]
        cv2.rectangle(frame, (x1,y1),(x2,y2), col, -1)
        cv2.rectangle(frame, (x1,y1),(x2,y2), (20,20,20), 1)
        if col == current_color:
            cv2.rectangle(frame, (x1-3,y1-3),(x2+3,y2+3),(255,255,255),2)
        if highlight_index == i:
            cv2.rectangle(frame, (x1-4,y1-4),(x2+4,y2+4),(0,255,255),2)
    # brush
    bx1,by1 = rects[8][0]; bx2,by2 = rects[8][1]
    cv2.rectangle(frame,(bx1,by1),(bx2,by2),(200,200,200),-1)
    cv2.putText(frame,"B",(bx1+14,by2-14),cv2.FONT_HERSHEY_SIMPLEX,0.9,(20,20,20),2)
    if active_tool=="BRUSH":
        cv2.rectangle(frame,(bx1-4,by1-4),(bx2+4,by2+4),(255,255,255),2)
    if highlight_index==8:
        cv2.rectangle(frame,(bx1-6,by1-6),(bx2+6,by2+6),(0,255,255),2)
    # eraser
    ex1,ey1 = rects[9][0]; ex2,ey2 = rects[9][1]
    cv2.rectangle(frame,(ex1,ey1),(ex2,ey2),(120,120,120),-1)
    pts = np.array([[ex1+6,ey2-10],[ex1+16,ey1+10],[ex2-6,ey1+10],[ex2-6,ey2-6]])
    cv2.fillPoly(frame,[pts],(200,200,200))
    if active_tool=="ERASER":
        cv2.rectangle(frame,(ex1-4,ey1-4),(ex2+4,ey2+4),(255,255,255),2)
    if highlight_index==9:
        cv2.rectangle(frame,(ex1-6,ey1-6),(ex2+6,ey2+6),(0,255,255),2)
    # save
    sx1,sy1 = rects[10][0]; sx2,sy2 = rects[10][1]
    cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),(60,0,0),-1)
    cv2.putText(frame,"S",(sx1+12,sy2-14),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    if highlight_index==10:
        cv2.rectangle(frame,(sx1-6,sy1-6),(sx2+6,sy2+6),(0,255,255),2)

# safe save
def save_canvas_image(frame, prefix="drawing"):
    try:
        fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(path, frame)
        print("[SAVE] saved to", path)
    except Exception as e:
        print("Save failed:", e)

# smoothing helpers
def lerp(a,b,t): return a + (b-a)*t

def smooth_point_deque(buffer, point, mix=0.65):
    if point is None:
        return None
    buffer.append(point)
    pts = list(buffer)
    n = len(pts)
    if n==0: return point
    weights = [i+1 for i in range(n)]
    total = sum(weights)
    avg_x = int(sum(p[0]*w for p,w in zip(pts,weights))/total)
    avg_y = int(sum(p[1]*w for p,w in zip(pts,weights))/total)
    last_x, last_y = pts[-1]
    sx = int(lerp(avg_x,last_x,mix)); sy = int(lerp(avg_y,last_y,mix))
    return (sx,sy)

# debug overlay
def draw_debug_overlay(frame, detected_hand, is_pinch, is_palm_open, is_fist, mode, gesture_on, hover_highlight):
    x, y = 10, 20
    lines = [
        f"HandDetected: {detected_hand}",
        f"GestureOn: {gesture_on}",
        f"Pinch: {is_pinch}",
        f"PalmOpen: {is_palm_open}",
        f"Fist: {is_fist}",
        f"Mode: {mode}",
        f"HoverHigh: {hover_highlight}"
    ]
    for i,ln in enumerate(lines):
        cv2.putText(frame, ln, (x, y + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)

# ---------------- Setup modules ----------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

tracker = HandTracker(maxHands=1, detectionConfidence=0.6, trackConfidence=0.6, smooth_factor=5)
utils = GestureUtils()
drawer = DrawEngine(stroke_thickness=6)
controller = GestureController()

# runtime state
gesture_on = True
hover_timers = {}
hover_selected = None
hover_highlight = None
hover_selected_index = None

last_erase_time = 0.0
fist_start = None
pinch_start = None

finger_buffer = deque(maxlen=8)
pointer_pos = None

start_time = time.time()
show_help_until = start_time + HELP_DURATION

mediapipe_ok = True

rects = toolbar_rects()

# active tool and color
active_tool = "BRUSH"
current_color = PALETTE[0]
drawer.change_color(current_color)

# function to set active tool/color when hover selection triggers
def apply_hover_selection(sel):
    global active_tool, current_color, hover_selected_index
    if sel is None: return
    if 0 <= sel <= 7:
        current_color = PALETTE[sel]
        drawer.change_color(current_color)
        active_tool = "BRUSH"
        hover_selected_index = sel
    elif sel == 8:
        active_tool = "BRUSH"
    elif sel == 9:
        active_tool = "ERASER"
    elif sel == 10:
        # save current view
        combined = frame_for_save.copy()
        combined = drawer.draw(combined)
        save_canvas_image(combined, prefix="drawing")
    # clear timers
    hover_timers.clear()

# helper to detect window closed
WINDOW_NAME = "Gesture Painter (AR)"

# main loop
print("[INFO] Running debug demo. Close the window to exit.")
while True:
    try:
        ret, cam = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        cam = cv2.flip(cam, 1)
        cam = cv2.resize(cam, (WIN_W, WIN_H))
        frame = cam.copy()

        # run mediapipe detection
        try:
            frame = tracker.findHands(frame, draw=True)
            points = tracker.get_finger_positions(frame)
            fingers = tracker.fingers_up(frame)
            hand_label = getattr(tracker, "hand_label", None)
            mediapipe_ok = True
        except Exception as e:
            mediapipe_ok = False
            points = None
            fingers = None
            hand_label = None

        index_raw = points.get("index") if points else None
        thumb_raw = points.get("thumb") if points else None

        # pointer smoothing
        if index_raw:
            pointer_pos = smooth_point_deque(finger_buffer, index_raw, mix=0.65)
        else:
            if SAFE_MODE:
                pointer_pos = None

        # pinch detection
        is_pinch = False
        if points and index_raw and thumb_raw:
            if utils.is_pinch(index_raw, thumb_raw, threshold=45):
                if pinch_start is None:
                    pinch_start = time.time()
                elif time.time() - pinch_start >= PINCH_HOLD:
                    is_pinch = True
            else:
                pinch_start = None
        else:
            pinch_start = None

        # fist detection (undo)
        is_fist = False
        now_t = time.time()
        if fingers is not None:
            if utils.is_fist(fingers, max_fingers_up=0):
                if fist_start is None:
                    fist_start = now_t
                elif now_t - fist_start >= FIST_HOLD and now_t >= last_erase_time + ERASE_COOLDOWN:
                    drawer.update(None, "ERASE")
                    last_erase_time = now_t
                    fist_start = None
                    is_fist = True
            else:
                fist_start = None

        # palm detection for erase mode
        is_palm_open = False
        if fingers is not None:
            try:
                is_palm_open = sum(1 for f in fingers if f) >= 4
            except Exception:
                is_palm_open = False

        # hover selection when gesture_on and pointer present
        hovered_index = None
        if gesture_on and points and points.get("index") and fingers is not None:
            hx, hy = points["index"]
            # update hover timers only when palm open to avoid accidental selection
            if sum(1 for f in fingers if f) >= 4:
                for i, rect in enumerate(rects):
                    (x1,y1),(x2,y2) = rect
                    if x1 <= hx <= x2 and y1 <= hy <= y2:
                        if i not in hover_timers:
                            hover_timers[i] = now_t
                        else:
                            elapsed = now_t - hover_timers[i]
                            hover_highlight = i
                            if elapsed >= HOVER_DELAY:
                                hover_selected = i
                                hovered_index = i
                                apply_hover_selection(i)
                                hover_timers.clear()
                                hover_highlight = None
                                break
                    else:
                        if i in hover_timers: del hover_timers[i]
            else:
                hover_timers.clear()
                hover_highlight = None

        # V sign toggle (index+middle up, ring+pinky down) - simple gating
        if points and fingers is not None:
            try:
                idx_up = fingers[1]; mid_up = fingers[2]; ring_up = fingers[3]; pinky_up = fingers[4]
                if idx_up and mid_up and (not ring_up) and (not pinky_up):
                    if 'v_toggle_time' not in globals(): globals()['v_toggle_time'] = 0
                    if time.time() - globals()['v_toggle_time'] > 1.0:
                        gesture_on = not gesture_on
                        globals()['v_toggle_time'] = time.time()
            except Exception:
                pass

        # controller mapping
        mode = controller.update_mode(is_pinch if gesture_on else False,
                                      is_palm_open if gesture_on else False,
                                      fingers)

        # ensure active_tool/global variables consistent
        # maintain variables in module-level globals to be accessible in hover logic
        if 'current_tool' not in globals():
            globals()['current_tool'] = active_tool
        else:
            active_tool = globals()['current_tool']

        # Apply drawing / erasing logic
        # If active_tool is BRUSH -> draw when mode == DRAW
        if active_tool == "BRUSH":
            if mode == "DRAW":
                drawer.update(pointer_pos, "DRAW")
            elif mode == "STOP":
                drawer.update(None, "STOP")
            elif mode == "ERASE":
                # while palm open replace area
                if pointer_pos:
                    px,py = pointer_pos
                    cv2.circle(frame, (px,py), 18, (20,20,20), -1)
                    # remove stroke points near pointer
                    new_strokes = []
                    erase_radius = 25
                    for stroke in drawer.strokes:
                        new_stroke = [pt for pt in stroke if math.hypot(pt[0]-px, pt[1]-py) > erase_radius]
                        if new_stroke: new_strokes.append(new_stroke)
                    drawer.strokes = new_strokes if new_strokes else [[]]
        elif active_tool == "ERASER":
            if pointer_pos:
                px,py = pointer_pos
                cv2.circle(frame,(px,py),18,(20,20,20),-1)
                new_strokes=[]
                erase_radius=25
                for stroke in drawer.strokes:
                    new_stroke = [pt for pt in stroke if math.hypot(pt[0]-px,pt[1]-py) > erase_radius]
                    if new_stroke: new_strokes.append(new_stroke)
                drawer.strokes = new_strokes if new_strokes else [[]]

        # render strokes
        frame_for_save = frame.copy()
        rendered = drawer.draw(frame_for_save)

        # draw pointer
        if pointer_pos:
            cv2.circle(rendered, pointer_pos, POINTER_RADIUS, (255,255,255), 2)

        # draw toolbar
        draw_minimal_icons(rendered, rects, drawer.stroke_color if hasattr(drawer,'stroke_color') else PALETTE[0], active_tool, highlight_index=hover_highlight)

        # help overlay at start
        if time.time() < show_help_until:
            overlay = rendered.copy()
            cv2.rectangle(overlay, (20,20),(WIN_W-20,120),(10,10,10),-1)
            cv2.addWeighted(overlay, 0.7, rendered, 0.3, 0, rendered)
            cv2.putText(rendered, "Pinch (index+thumb) = Draw   |   Palm = Erase   |   Fist = Undo", (40,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220),2)
            cv2.putText(rendered, "Hover toolbar to select colors/tools. Toggle gestures: V sign.", (40,95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180),1)

        # DEBUG overlay (you asked for debug to diagnose)
        draw_debug_overlay(rendered, bool(points), is_pinch, is_palm_open, is_fist, mode, gesture_on, hover_highlight)

        # show
        cv2.imshow(WINDOW_NAME, rendered)

        # detect window close properly
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed by user. Exiting...")
            break

        # small wait
        if cv2.waitKey(1) & 0xFF == 27:
            # still allow ESC to close for debugging, user previously didn't want keyboard exit but
            # keeping this here as an emergency; you can ignore pressing ESC in demo.
            print("[INFO] ESC pressed. Exiting.")
            break

    except KeyboardInterrupt:
        break
    except Exception as err:
        print("[ERROR] main loop exception:", err)
        time.sleep(0.05)
        continue

# cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Exited cleanly.")
