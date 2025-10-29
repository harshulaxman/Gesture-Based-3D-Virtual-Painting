# src/core/controller.py
class GestureController:
    def __init__(self):
        self.mode = "STOP"

    def update_mode(self, draw_gesture, erase_gesture, fingers):
        """
        Priority order:
        1. ERASE only when fist (0 fingers up)
        2. DRAW only when pinch and other fingers NOT closed like fist
        3. Otherwise STOP
        """
        # Count how many fingers are up
        fingers_up_count = sum(fingers) if fingers else 0

        # ✅ Highest priority – ERASE with real fist
        if erase_gesture and fingers_up_count == 0:
            self.mode = "ERASE"

        # ✅ DRAW only if pinch detected and not fist-like
        elif draw_gesture and fingers_up_count >= 1:
            self.mode = "DRAW"

        # ✅ Default
        else:
            self.mode = "STOP"

        return self.mode
