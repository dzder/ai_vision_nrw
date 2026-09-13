import json

def rewrite_notebook():
    with open('test_l1.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'class HandTracker:' in source:
                print(f'Found HandTracker in cell {i}')
                new_source = """class HandTracker:
    def __init__(self, smoothing_alpha=0.3, max_num_hands=1):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        import os, urllib.request
        
        self.smoothing_alpha = smoothing_alpha
        self.prev_pos = None
        
        # Ensure model exists
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            if os.path.exists("models/hand_landmarker.task"):
                model_path = "models/hand_landmarker.task"
            else:
                print("Downloading hand model...")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    model_path
                )
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.hands = HandLandmarker.create_from_options(options)
        self.timestamp = 0
        
    def get_hand_position(self, frame_bgr):
        import mediapipe as mp
        import cv2
        import numpy as np
        
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        self.timestamp += 33
        results = self.hands.detect_for_video(mp_image, self.timestamp)
        
        if not results.hand_landmarks:
            return None, frame_bgr
            
        lm = results.hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        
        # Draw skeleton manually
        _HAND_CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                       (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                       (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in _HAND_CONNS:
            cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(frame_bgr, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)
            
        wrist = lm[0]
        mcp = lm[9]
        cx = (wrist.x + mcp.x) / 2.0
        cy = (wrist.y + mcp.y) / 2.0
        
        # Scale to match what lighting module expects
        raw_pos = np.array([
            (cx * w - (w - 1) / 2.0) / max(w, h),
            (cy * h - (h - 1) / 2.0) / max(w, h),
            np.clip(-0.3 - (1.0 - wrist.z) * 0.5, -3.0, -0.05)
        ], dtype=np.float32)
        
        if self.prev_pos is None:
            self.prev_pos = raw_pos
        else:
            self.prev_pos = self.smoothing_alpha * raw_pos + (1.0 - self.smoothing_alpha) * self.prev_pos
            
        return self.prev_pos, frame_bgr
"""
                # Replace the cell
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                cell['source'][-1] = cell['source'][-1].strip('\n')

    with open('test_l1.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
rewrite_notebook()
print("Notebook hand tracker updated!")
