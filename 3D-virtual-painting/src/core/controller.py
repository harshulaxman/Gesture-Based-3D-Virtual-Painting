# src/core/controller.py
class GestureController:
    def __init__(self):
        self.mode = "STOP"

    def update_mode(self, draw_gesture, erase_gesture):
        # draw has highest priority (intentional pinch)
        if draw_gesture:
            self.mode = "DRAW"
        elif erase_gesture:
            self.mode = "ERASE"
        else:
            self.mode = "STOP"
        return self.mode
