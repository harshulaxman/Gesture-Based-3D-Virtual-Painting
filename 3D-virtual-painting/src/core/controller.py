class GestureController:
    """
    Handles gesture-to-mode mapping for virtual painting.
    Maps:
      - Pinch → DRAW mode
      - Palm → ERASE mode
      - No gesture → STOP mode
    """

    def __init__(self):
        self.mode = "STOP"

    def update_mode(self, is_pinch: bool, is_palm: bool, fingers=None) -> str:
        """
        Updates current gesture mode based on detected hand state.
        Args:
            is_pinch (bool): True if pinch gesture detected.
            is_palm (bool): True if palm open.
            fingers (list): Optional list of finger states (for future logic).
        Returns:
            str: "DRAW", "ERASE", or "STOP"
        """
        if is_pinch:
            self.mode = "DRAW"
        elif is_palm:
            self.mode = "ERASE"
        else:
            self.mode = "STOP"
        return self.mode
