import cv2
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
cap.release()
cv2.destroyAllWindows()
