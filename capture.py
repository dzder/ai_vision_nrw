"""OpenCV capture boundary: unmodified HxWx3 uint8 BGR frames."""
import os
import cv2


class Webcam:
    def __init__(self, index=0, width=640, height=480):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open camera {index}. Check camera permissions or --camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Best effort: some camera backends ignore this setting.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stopped returning frames.")
        return frame

    def close(self):
        self.cap.release()
