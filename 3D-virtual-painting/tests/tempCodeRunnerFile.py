"""
Gesture Painter — Smooth, Continuous Brush
Pinch = Draw | Spread Fingers = Erase | Fist = Undo | Hover color/tool to select
"""

import os, sys, time, math, cv2, numpy as np
from collections import deque

# ---------- PATH FIX ----------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# ---------- MODULES ----------
from gestures.gesture_tracker import HandTracker
from gestures.gesture_utils import GestureUtils
from core.draw_engine import DrawEngine
from core.controller import GestureController

# ---------- CONFIG ----------
CAM_INDEX = 0
WIN_W, WIN_H = 1280, 720
BAR_H = 100
POINTER_RADIUS = 7
HELP_DURATION = 3.0
HOVER_DELAY = 0.4
FIST_HOLD = 0.4
UNDO_COOLDOWN = 1.2
EMA_ALPHA = 0.6
INTERP_STEP = 4
PINCH_MEMORY = 0.25

# 🎨 Extended Paint Palette (includes Orange + Dark Green)
PALETTE = [
    (0, 0, 0),        # Black
    (255, 255, 255),  # White
    (0, 0, 255),      # Blue
    (0, 128, 0),      # Dark Green
    (0, 255, 0),      # Bright Green
    (255, 0, 0),      # Red
    (255, 165, 0),    # Orange
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (128, 0, 128),    # Purple
    (128, 128, 0)     # Olive
]

COLOR_NAMES = [
    "Black", "White", "Blue", "Dark Green", "Green", "Red",
    "Orange", "Yellow", "Cyan", "Magenta", "Purple", "Olive"
]
TOTAL_COLORS = len(PALETTE)
TOOL_BRUSH = TOTAL_COLORS
TOOL_ERASER = TOTAL_COLORS + 1
TOOL_SAVE = TOTAL_COLORS + 2

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HELPERS ----------
def toolbar_rects():
    rects = []
    padding_x = 20
    total_items = TOTAL_COLORS + 3
    slot_w = (WIN_W - 2 * padding_x) // total_items

    x = padding_x
    y1 = WIN_H - BAR_H + 10
    y2 = WIN_H - 15

    for _ in range(TOTAL_COLORS):
        rects.append(((x, y1), (x + slot_w - 6, y2)))
        x += slot_w

    for _ in range(3):  # Brush, Eraser, Save
        rects.append(((x, y1), (x + slot_w - 6, y2)))
        x += slot_w

    return rects


def draw_toolbar(frame, rects, current_color, active_tool, highlight=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, WIN_H - BAR_H), (WIN_W, WIN_H), (170,170,170), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.line(frame, (0, WIN_H - BAR_H), (WIN_W, WIN_H - BAR_H), (80,80,80), 2)

    # --- Colors ---
    for i in range(TOTAL_COLORS):
        (x1, y1), (x2, y2) = rects[i]
        col = PALETTE[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 1)
        if col == current_color:
            cv2.rectangle(frame, (x1-3,y1-3),(x2+3,y2+3),(255,255,255),2)
        if highlight == i:
            cv2.rectangle(frame, (x1-4,y1-4),(x2+4,y2+4),(0,255,255),2)

    # --- Tool Buttons ---
    tool_labels = ["B", "E", "S"]
    tool_names = ["BRUSH", "ERASER", "SAVE"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, label in enumerate(tool_labels):
        i = TOTAL_COLORS + idx
        (x1, y1), (x2, y2) = rects[i]
        color = (255,255,255) if active_tool == tool_names[idx] else (200,200,200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50,50,50), 1)
        cv2.putText(frame, label, (x1 + 14, y2 - 14), font, 0.9, (20,20,20), 2)
        if highlight == i:
            cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (0,255,255), 2)

    # --- Status (above toolbar) ---
    status_y = WIN_H - BAR_H - 15
    cv2.rectangle(frame, (10, status_y - 25), (300, status_y), (190,190,190), -1)
    cv2.rectangle(frame, (10, status_y - 25), (300, status_y), (100,100,100), 1)
    cv2.putText(frame, f"Mode: {active_tool}", (20, status_y - 5), font, 0.6, (0,0,0), 2)

def draw_header(frame):
    """Draws a top header bar with title 'GesturePaint'."""
    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIN_W, bar_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title text
    title = "GesturePaint"
    subtitle = "Pinch = Draw   |   Spread Fingers = Erase   |   Fist = Undo"
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, title, (40, 35), font, 1.1, (255, 255, 255), 2)
    cv2.putText(frame, subtitle, (350, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)


def ema(prev, curr, alpha=EMA_ALPHA):
    if prev is None: return curr
    return (int(alpha*curr[0] + (1-alpha)*prev[0]),
            int(alpha*curr[1] + (1-alpha)*prev[1]))


def interpolate_points(p1, p2, step=INTERP_STEP):
    if p1 is None or p2 is None: return []
    x1,y1 = p1; x2,y2 = p2
    dist = math.hypot(x2-x1, y2-y1)
    if dist <= step:
        return [p2]
    n = int(dist // step)
    return [(int(x1 + (i/n)*(x2-x1)), int(y1 + (i/n)*(y2-y1))) for i in range(1, n+1)]

# ---------- MAIN SETUP ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

tracker = HandTracker(maxHands=1)
utils = GestureUtils()
drawer = DrawEngine(stroke_thickness=6)
controller = GestureController()

rects = toolbar_rects()
active_tool = "BRUSH"
current_color = PALETTE[0]
drawer.change_color(current_color)

hover_timers = {}
hover_highlight = None
fist_start = None
last_undo_time = 0
pointer_pos = None
smoothed_pos = None
pinch_last_time = 0.0
prev_draw_pos = None
show_help_until = time.time() + HELP_DURATION

WINDOW_NAME = "Gesture Painter (AR)"
print("[INFO] Running... (Pinch=Draw, Spread=Erase, Fist=Undo, Hover color/tool to select)")

# ---------- MAIN LOOP ----------
while True:
    ret, cam = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    cam = cv2.flip(cam, 1)
    frame = cv2.resize(cam, (WIN_W, WIN_H))
    now_t = time.time()

    frame = tracker.findHands(frame, draw=True)
    points = tracker.get_finger_positions(frame)
    fingers = tracker.fingers_up(frame)

    index_raw = points.get("index") if points else None
    thumb_raw = points.get("thumb") if points else None

    pointer_pos = index_raw if index_raw else None
    if pointer_pos:
        smoothed_pos = ema(smoothed_pos, pointer_pos)
    else:
        smoothed_pos = ema(smoothed_pos, smoothed_pos) if smoothed_pos else None

    mode = "STOP"
    if points and "index" in points and "thumb" in points:
        ix, iy = points["index"]; tx, ty = points["thumb"]
        dist = math.hypot(ix - tx, iy - ty)
        if dist < 50:
            mode = "DRAW"
            pinch_last_time = now_t
        elif dist > 100:
            mode = "ERASE"
    elif now_t - pinch_last_time < PINCH_MEMORY:
        mode = "DRAW"

    # --- FIST for Undo ---
    fingers_up_count = sum(fingers) if fingers else 5
    avg_dist = 999
    if points and fingers:
        wrist = points.get("wrist")
        tips = [points.get(n) for n in ["index","middle","ring","pinky"] if points.get(n)]
        if wrist and tips:
            avg_dist = sum(math.hypot(t[0]-wrist[0], t[1]-wrist[1]) for t in tips) / len(tips)
    if fingers_up_count <= 2 and avg_dist < 80:
        if fist_start is None:
            fist_start = now_t
        elif (now_t - fist_start) > FIST_HOLD and (now_t - last_undo_time) > UNDO_COOLDOWN:
            if hasattr(drawer, "strokes") and drawer.strokes:
                drawer.strokes.pop()
                print("[GESTURE] ✊ Fist → Undo")
            last_undo_time = now_t
            fist_start = None
    else:
        fist_start = None

    # --- Hover select color/tool ---
    hovered_index = None
    if points and points.get("index"):
        hx, hy = points["index"]
        for i, ((x1,y1),(x2,y2)) in enumerate(rects):
            inside = x1 <= hx <= x2 and y1 <= hy <= y2
            if inside:
                if i not in hover_timers:
                    hover_timers[i] = now_t
                elif (now_t - hover_timers[i]) >= HOVER_DELAY:
                    hovered_index = i
                    hover_highlight = i
                    if 0 <= i < TOTAL_COLORS:
                        current_color = PALETTE[i]
                        drawer.change_color(current_color)
                        active_tool = "BRUSH"
                        print(f"[COLOR] 🎨 {COLOR_NAMES[i]}")
                    elif i == TOOL_BRUSH:
                        active_tool = "BRUSH"
                        print("[TOOL] 🖌️ Brush Selected")
                    elif i == TOOL_ERASER:
                        active_tool = "ERASER"
                        print("[TOOL] 🧽 Eraser Selected")
                    elif i == TOOL_SAVE:
                        frame_to_save = drawer.draw(frame.copy())
                        filename = os.path.join(OUTPUT_DIR, f"drawing_{int(time.time())}.png")
                        cv2.imwrite(filename, frame_to_save)
                        print(f"[SAVE] 💾 Drawing saved to {filename}")
                    hover_timers.clear()
                    hover_highlight = None
                    break
            else:
                hover_timers.pop(i, None)

    # --- Apply Draw/Erase ---
    if mode == "DRAW" and smoothed_pos:
        for p in interpolate_points(prev_draw_pos, smoothed_pos):
            drawer.update(p, "DRAW")
        prev_draw_pos = smoothed_pos
        active_tool = "BRUSH"
    elif mode == "ERASE" and smoothed_pos:
        drawer.erase_at(smoothed_pos, radius=25)
        active_tool = "ERASER"
        prev_draw_pos = None
        cv2.circle(frame, smoothed_pos, 25, (0,0,255), 2)
        cv2.putText(frame, "ERASING...", (40,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    else:
        drawer.update(None, "STOP")
        prev_draw_pos = None

    rendered = drawer.draw(frame.copy())

    if smoothed_pos:
        cv2.circle(rendered, smoothed_pos, POINTER_RADIUS, (255,255,255), 2)

    draw_toolbar(rendered, rects, current_color, active_tool, highlight=hover_highlight)
    draw_header(rendered)

    if time.time() < show_help_until:
        cv2.putText(rendered, "Pinch=Draw | Spread=Erase | Fist=Undo | Hover=Select",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow(WINDOW_NAME, rendered)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Exited cleanly.")
