"""A clicked image point tracked between processed frames using optical flow."""
import cv2
import numpy as np


class PointProbe:
    def __init__(self):
        self.gray = None
        self.point = None
        self.lost = False
        self.baseline = None

    def clear(self):
        self.point = None
        self.baseline = None
        self.lost = False

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.clear()
        elif event == cv2.EVENT_LBUTTONDOWN and self.gray is not None:
            h, w = self.gray.shape
            # Both panels represent the same camera coordinates; footer is ignored.
            if 0 <= x < 2*w and 0 <= y < h:
                self.point = np.array([x % w, y], dtype=np.float32)
                self.baseline = None
                self.lost = False

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.point is not None and self.gray is not None:
            if gray.shape != self.gray.shape:
                self.clear()
                self.lost = True
            else:
                previous = self.point.reshape(1, 1, 2)
                options = dict(winSize=(31, 31), maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                         30, 0.01))
                current, ok, error = cv2.calcOpticalFlowPyrLK(
                    self.gray, gray, previous, None, **options)
                valid = (current is not None and ok is not None and bool(ok[0, 0])
                         and np.isfinite(current).all()
                         and error is not None and float(error[0, 0]) < 30)
                if valid:
                    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
                        gray, self.gray, current, None, **options)
                    h, w = gray.shape
                    x, y = current[0, 0]
                    valid = (back is not None and back_ok is not None
                             and bool(back_ok[0, 0]) and np.isfinite(back).all()
                             and float(np.linalg.norm(back-previous)) < 1.5
                             and 0 <= x < w and 0 <= y < h)
                if valid:
                    self.point = current[0, 0]
                else:
                    self.clear()
                    self.lost = True
        self.gray = gray

    def draw(self, display, width, normal):
        x, y = (int(round(v)) for v in self.point)
        x, y = min(x, width-1), min(y, self.gray.shape[0]-1)
        # Fixed 60-pixel vector scale; XY projection becomes short when facing Z.
        for offset in (0, width):
            origin = (x+offset, y)
            end = (int(round(x+offset+60*normal[0])), int(round(y+60*normal[1])))
            cv2.circle(display, origin, 6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(display, origin, end, (0, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.25)
