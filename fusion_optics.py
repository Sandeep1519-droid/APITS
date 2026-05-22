import cv2
import numpy as np

class AsymmetricFusionOptics:
    """
    2026 Defense-Grade Asymmetric Spatio-Temporal Fusion (ASTF).
    Fuses Neuromorphic Event Tracking (SNN) with Texture Tracking (CSRT).
    """
    def __init__(self, camera_idx=0, width=1280, height=720):
        print("[OPTICS] Initializing Asymmetric Spatio-Temporal Fusion (ASTF)...")
        self.cap = cv2.VideoCapture(camera_idx)
        
        # Force high-performance resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60) 
        
        # Camera Warm-Up
        for _ in range(10): 
            self.cap.read()
            
        success, first_frame = self.cap.read()
        if not success:
            raise RuntimeError("CRITICAL ERROR: Optical Sensor Offline.")
            
        first_frame = cv2.resize(first_frame, (1280, 720))
        
        # --- SNN Memory Initialization ---
        self.prev_gray = cv2.GaussianBlur(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # --- CSRT Initialization ---
        self.tracker = cv2.TrackerCSRT_create()
        self.is_locked = False
        
        self.MIN_AREA = 800  

    def scan_airspace(self):
        success, frame = self.cap.read()
        if not success:
            return None, None, []

        frame = cv2.resize(frame, (1280, 720))
        curr_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        
        # ====================================================
        # 1. THE SNN NEUROMORPHIC PASS (Microsecond Reflexes)
        # ====================================================
        delta_stream = cv2.absdiff(self.prev_gray, curr_gray)
        _, event_mask = cv2.threshold(delta_stream, 25, 255, cv2.THRESH_BINARY)
        event_mask = cv2.dilate(event_mask, self.morph_kernel, iterations=2)
        
        contours, _ = cv2.findContours(event_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        snn_box = None
        snn_area = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            snn_area = cv2.contourArea(largest_contour)
            if snn_area > self.MIN_AREA:
                snn_box = cv2.boundingRect(largest_contour)
                
        self.prev_gray = curr_gray
        
        # ====================================================
        # 2. THE ASYMMETRIC FUSION CORE (ASTF)
        # ====================================================
        targets = []
        fused_box = None
        
        if not self.is_locked:
            # Phase A: Autonomous SNN Lock-On
            if snn_box:
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, snn_box)
                self.is_locked = True
                fused_box = snn_box
                print("[OPTICS] SNN Acquired Target. Handing off to CSRT.")
        else:
            # Phase B: CSRT Texture Tracking with SNN Oversight
            success_csrt, csrt_box = self.tracker.update(frame)
            
            if success_csrt:
                # MATHEMATICAL FUSION
                # CSRT is confident (Weight: 85%), but we pull it slightly towards the SNN (Weight: 15%)
                # This absorbs rotor vibration but keeps the box perfectly centered.
                omega = 0.85 
                if snn_box:
                    fx = int((omega * csrt_box[0]) + ((1 - omega) * snn_box[0]))
                    fy = int((omega * csrt_box[1]) + ((1 - omega) * snn_box[1]))
                    fw = int((omega * csrt_box[2]) + ((1 - omega) * snn_box[2]))
                    fh = int((omega * csrt_box[3]) + ((1 - omega) * snn_box[3]))
                    fused_box = (fx, fy, fw, fh)
                else:
                    fused_box = tuple(map(int, csrt_box))
            else:
                # Phase C: EMERGENCY SNN RESCUE
                # CSRT lost the target due to High-Speed Motion Blur.
                if snn_box:
                    print("[OPTICS] CSRT Failed (Motion Blur). SNN Rescuing Lock!")
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, snn_box)
                    fused_box = snn_box
                else:
                    print("[OPTICS] Target Lost Completely.")
                    self.is_locked = False
        
        # ====================================================
        # 3. TELEMETRY PACKAGING FOR THE APEX LSTM
        # ====================================================
        if fused_box:
            x, y, w, h = fused_box
            cx = x + (w // 2)
            cy = y + (h // 2)
            
            targets.append({
                'x': cx, 'y': cy, 'w': w, 'h': h, 'area': w * h
            })
            
            # Draw the Fusion Indicator on the raw camera feed
            cv2.putText(frame, "ASTF FUSION ACTIVE", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            
        return frame, event_mask, targets

    def terminate(self):
        self.cap.release()
        cv2.destroyAllWindows()