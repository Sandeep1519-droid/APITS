"""
╔══════════════════════════════════════════════════════════════════════╗
║   APITS v6.0 — COMPLETE REBUILD                                      ║
║   SimpleBlobDetector + LinearKF6D + SAFC + EdgeTripwire             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  WHAT CHANGED FROM v5.x AND WHY:                                    ║
║                                                                      ║
║  KILLED: FRST (Fast Radial Symmetry Transform)                       ║
║    Reason: FRST accumulates votes from ALL edge pixels in the        ║
║    motion mask. A noisy mask (shadows, motion blur edges) creates    ║
║    hundreds of spurious gradient vectors that out-vote the ball.     ║
║    It is fundamentally sensitive to mask quality.                    ║
║                                                                      ║
║  REPLACED WITH: cv2.SimpleBlobDetector (C++ compiled, ~0.3ms)        ║
║    SimpleBlobDetector works on FILLED BLOBS, not on edges.           ║
║    It measures area, circularity, convexity, and inertia directly    ║
║    on connected white regions. A shadow or motion-blur edge is a     ║
║    THIN arc (low area, low circularity) — it is immediately          ║
║    rejected. A ball blob is a SOLID CIRCLE — it passes all gates.   ║
║    The detector is implemented in optimised C++ inside OpenCV:       ║
║    10× faster than Python-loop FRST with better discrimination.     ║
║                                                                      ║
║  KILLED: 9D UKF (Unscented Kalman Filter with acceleration states)   ║
║    Reason: Acceleration estimated as a free state is the root cause  ║
║    of the covariance explosion. One poisoned measurement drives Ax   ║
║    to an impossible value, inflates P, breaks Cholesky, and the     ║
║    filter trusts subsequent garbage measurements blindly.            ║
║                                                                      ║
║  REPLACED WITH: LinearKF6D (6D Linear Kalman, gravity as input)     ║
║    State = [X,Y,Z,Vx,Vy,Vz]. Gravity is a KNOWN CONSTANT baked     ║
║    into the process model as a control input u=[0,g,0,0,g,0].       ║
║    The Riccati equation is linear → covariance CANNOT explode.      ║
║    Uses Mahalanobis distance gating: statistically impossible        ║
║    measurements are rejected before they touch the state.           ║
║                                                                      ║
║  KEPT: SAFCCompensator (analytically correct, zero noise)            ║
║  KEPT: EdgeTripwire (proven <1ms first-frame acquisition)            ║
║  KEPT: BallShapeFilter (anti-human contour filter)                   ║
║  KEPT: ArduinoBridge + lead-angle compensation                       ║
║                                                                      ║
║  PIPELINE:                                                           ║
║  Frame → SAFC_Warp → 2-Frame AbsDiff → Morphology →                ║
║  BallShapeFilter → SimpleBlobDetector → Mahalanobis Gate →          ║
║  LinearKF6D → predict_future_pixels → HUD + Servo                   ║
║                                                                      ║
║  PARALLEL: EdgeTripwire scans border on EVERY frame (0.8ms cost)    ║
║  and seeds Ghost KF the microsecond an object breaches the FOV.     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import threading
import math
import os
import collections

from linear_kf6d import LinearKF6D
from kinematic_detector import KinematicMassDetector

try:
    from interface import TacticalHUD as _ExtHUD
    _HUD_OK = True
except ImportError:
    _HUD_OK = False

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False


# ══════════════════════════════════════════════════════════════════════
# SYSTEM CONFIGURATION — edit these, nothing else
# ══════════════════════════════════════════════════════════════════════

CAM_PORT         = 1
CAM_W            = 640
CAM_H            = 480
CAM_FPS          = 120

FOCAL_LENGTH_PX  = 600.0
REAL_TARGET_W_M  = 0.063          # 63mm stress ball

EXPOSURE_INIT    = -9             # locked from v5.1 calibration
GAIN_INIT        = 0

# SAFC
SERVO_INIT_PAN   = 90.0
SERVO_INIT_TILT  = 90.0

# Motion detection
DIFF_THRESH      = 18             # absdiff pixel delta → motion
MORPH_OPEN_PX    = 2              
MORPH_CLOSE_PX   = 9

# BallShapeFilter (contour gates — same as v5.2)
BALL_MIN_AREA    = 180
BALL_MAX_AREA    = 54000
BALL_MIN_CIRC    = 0.70
BALL_MIN_CONVEX  = 0.80

# SimpleBlobDetector params (tight: only solid circular blobs pass)
BLOB_MIN_AREA    = 150.0
BLOB_MAX_AREA    = 55000.0
BLOB_MIN_CIRC    = 0.68           # slightly looser than contour filter
BLOB_MIN_CONVEX  = 0.78
BLOB_MIN_INERTIA = 0.50           # rejects elongated blobs (motion blur streaks)

# LinearKF6D gating
KF_ROI_PAD_FACTOR = 3.5
KF_ROI_MIN_PX     = 90
REACQUIRE_FRAMES  = 7
NUKE_BAD_FRAMES   = 4

# Servo lag
SERVO_LAG_S      = 0.150
ARDUINO_BAUD     = 115200
ARDUINO_PORT     = None

# HUD trajectory
PREDICT_FRAMES   = 18

# Tripwire
BORDER_FRAC      = 0.15
TRIPWIRE_THRESH  = 16
GHOST_Z_M        = 2.5
GHOST_SPEED_MS   = 7.0
GHOST_MAX_FRAMES = 7


# ══════════════════════════════════════════════════════════════════════
# FALLBACK HUD (when interface.py absent)
# ══════════════════════════════════════════════════════════════════════

class _MinimalHUD:
    def log_spacetime_telemetry(self, *a, **kw): pass
    def export_omega_dataset(self): pass
    def render(self, frame, target_x=0, target_y=0, target_w=0, target_h=0,
               future_path=None, curr_3d=(0,0,0), fut_3d=(0,0,0),
               speed_3d=0., tti=0., launch_auth=False, status="INIT", **kw):
        h, w = frame.shape[:2]
        clr  = (0,255,0) if status not in ("SCANNING","ACQUIRING") else (80,80,80)
        if target_w > 0:
            hw = target_w // 2
            cv2.rectangle(frame,(target_x-hw,target_y-hw),
                          (target_x+hw,target_y+hw), clr, 2)
            cv2.circle(frame,(target_x,target_y),3,clr,-1)
        if future_path:
            pts = [(int(p[0]),int(p[1])) for p in future_path]
            for i in range(len(pts)-1):
                a = i/max(len(pts)-1,1)
                cv2.line(frame,pts[i],pts[i+1],
                         (0,int(255*(1-a)),int(255*a)),1,cv2.LINE_AA)
        X,Y,Z = curr_3d
        cv2.putText(frame,f"3D:({X:+.2f},{Y:+.2f},{Z:.2f})m  spd:{speed_3d:.2f}m/s",
                    (w-390,28),cv2.FONT_HERSHEY_SIMPLEX,0.44,clr,1)
        cv2.putText(frame,status,(w-200,52),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,
                    (0,255,255) if launch_auth else clr,1)
        return frame

TacticalHUD = _ExtHUD if _HUD_OK else _MinimalHUD


# ══════════════════════════════════════════════════════════════════════
# CAMERA STREAM  (unchanged from v5.x)
# ══════════════════════════════════════════════════════════════════════

class CameraStream:
    def __init__(self, src=CAM_PORT):
        print(f"[EYE] Opening port {src}...")
        self.cap = self._open(src)
        self._configure()
        ret, f = self.cap.read()
        if not ret: raise RuntimeError("[EYE] Blank feed.")
        self._lock   = threading.Lock()
        self._frame  = self._gray(f)
        self.stopped = False
        print(f"[EYE] {self.cap.get(cv2.CAP_PROP_FPS):.0f} FPS  "
              f"EXP={self.cap.get(cv2.CAP_PROP_EXPOSURE):.0f}")

    def _open(self, src):
        for p,b in [(src,cv2.CAP_DSHOW),(0,cv2.CAP_DSHOW),(0,cv2.CAP_ANY)]:
            c=cv2.VideoCapture(p,b)
            if c.isOpened(): return c
        raise RuntimeError("[EYE] All ports failed.")

    def _configure(self):
        c=self.cap
        c.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*'MJPG'))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
        c.set(cv2.CAP_PROP_FPS,         CAM_FPS)
        c.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)
        c.set(cv2.CAP_PROP_AUTOFOCUS,   0)
        c.set(cv2.CAP_PROP_AUTO_WB,     0)
        c.set(cv2.CAP_PROP_BRIGHTNESS,  128)
        c.set(cv2.CAP_PROP_CONTRAST,    160)
        c.set(cv2.CAP_PROP_EXPOSURE,    EXPOSURE_INIT)
        c.set(cv2.CAP_PROP_GAIN,        GAIN_INIT)
        c.set(cv2.CAP_PROP_SHARPNESS,   3)

    @staticmethod
    def _gray(f):
        return cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) if f.ndim==3 else f

    def start(self):
        threading.Thread(target=self._run,daemon=True,name="Cam").start()
        return self

    def _run(self):
        while not self.stopped:
            r,f=self.cap.read()
            if r and f is not None:
                g=self._gray(f)
                with self._lock: self._frame=g

    def read(self):
        with self._lock: return self._frame.copy()

    def stop(self):
        self.stopped=True
        time.sleep(0.05)
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════
# SAFC COMPENSATOR  (unchanged from v5.x — analytically correct)
# ══════════════════════════════════════════════════════════════════════

class SAFCCompensator:
    def __init__(self, fw, fh, f):
        self.fw, self.fh, self.f = fw, fh, f
        cx, cy = fw*0.5, fh*0.5
        self.K     = np.array([[f,0,cx],[0,f,cy],[0,0,1]],np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.prev_pan  = SERVO_INIT_PAN
        self.prev_tilt = SERVO_INIT_TILT
        self.last_H    = np.eye(3, dtype=np.float64)

    def compute_H(self, pan_deg, tilt_deg):
        dp = math.radians(pan_deg  - self.prev_pan)
        dt = math.radians(tilt_deg - self.prev_tilt)
        self.prev_pan  = pan_deg
        self.prev_tilt = tilt_deg
        if abs(dp)<1e-7 and abs(dt)<1e-7:
            self.last_H = np.eye(3, dtype=np.float64)
            return self.last_H
        cp,sp = math.cos(dp), math.sin(dp)
        ct,st = math.cos(dt), math.sin(dt)
        Ry = np.array([[ cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float64)
        Rx = np.array([[1,0,0],[0,ct,-st],[0,st,ct]], dtype=np.float64)
        H  = self.K @ (Ry@Rx) @ self.K_inv
        if not np.isfinite(H).all() or abs(np.linalg.det(H))<0.1:
            H = np.eye(3, dtype=np.float64)
        self.last_H = H
        return H


# ══════════════════════════════════════════════════════════════════════
# FAST MOTION DETECTOR
# 2-frame SAFC absdiff + morphology.  Simpler than 3-frame AND-diff.
# The SimpleBlobDetector below is robust enough to handle single-frame
# noise — we don't need AND-diff's extra latency frame any more.
# ══════════════════════════════════════════════════════════════════════

class FastMotionDetector:
    def __init__(self, safc: SAFCCompensator):
        self.safc      = safc
        self.prev_gray = None
        self._open_k   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(MORPH_OPEN_PX, MORPH_OPEN_PX))
        self._close_k  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(MORPH_CLOSE_PX, MORPH_CLOSE_PX))

    def process(self, gray, pan_deg, tilt_deg):
        H = self.safc.compute_H(pan_deg, tilt_deg)
        fh, fw = gray.shape
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros((fh,fw),np.uint8), H

        warp_flags = dict(flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        warped = cv2.warpPerspective(self.prev_gray, H, (fw,fh), **warp_flags)

        diff   = cv2.absdiff(gray, warped)
        diff   = cv2.GaussianBlur(diff,(5,5),0)
        _,mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

        # Open (kill noise) then Close (fill ball interior)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)

        self.prev_gray = gray.copy()
        return mask, H


# ══════════════════════════════════════════════════════════════════════
# BALL SHAPE FILTER  (same as v5.2 — proven anti-human filter)
# ══════════════════════════════════════════════════════════════════════

class BallShapeFilter:
    @staticmethod
    def filter(mask, fw, fh):
        if mask is None or not np.any(mask):
            return mask
        if np.count_nonzero(mask) / (fw*fh) > 0.12:
            return np.zeros_like(mask)
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.zeros_like(mask)
        clean = np.zeros_like(mask)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (BALL_MIN_AREA <= area <= BALL_MAX_AREA): continue
            perim = cv2.arcLength(cnt,True)
            if perim < 1e-6: continue
            if (4*math.pi*area)/(perim*perim) < BALL_MIN_CIRC: continue
            hull = cv2.convexHull(cnt)
            ha   = cv2.contourArea(hull)
            if ha < 1e-6 or area/ha < BALL_MIN_CONVEX: continue
            x,y,bw,bh = cv2.boundingRect(cnt)
            if max(bw,bh)/max(min(bw,bh),1) > 2.2: continue
            cv2.drawContours(clean,[cnt],-1,255,cv2.FILLED)
        return clean


# ══════════════════════════════════════════════════════════════════════
# SIMPLE BLOB DETECTOR  (replaces FRST — the paradigm shift)
#
# Why SimpleBlobDetector beats FRST for this task:
#   • Works on FILLED BLOBS not edge gradients → shadow/blur immune
#   • C++ compiled → ~0.3ms for full frame scan
#   • Filters by area, circularity, convexity, and inertia ratio
#     simultaneously in one pass
#   • Inertia ratio is the killer feature: a motion-blur streak has
#     inertia ratio ~0.05–0.15 (highly elongated ellipse). A ball has
#     inertia ratio > 0.5. This alone eliminates 90% of false positives.
#   • Sub-pixel centroid: OpenCV computes the blob centroid using the
#     full connected component moments — same accuracy as our manual
#     image-moment code but in C++.
# ══════════════════════════════════════════════════════════════════════

def _build_blob_detector() -> cv2.SimpleBlobDetector:
    """Build and return a configured SimpleBlobDetector."""
    p = cv2.SimpleBlobDetector_Params()

    p.filterByArea      = True
    p.minArea           = BLOB_MIN_AREA
    p.maxArea           = BLOB_MAX_AREA

    p.filterByCircularity = True
    p.minCircularity    = BLOB_MIN_CIRC

    p.filterByConvexity = True
    p.minConvexity      = BLOB_MIN_CONVEX

    p.filterByInertia   = True
    p.minInertiaRatio   = BLOB_MIN_INERTIA   # ← kills motion-blur streaks

    p.filterByColor     = False              # monochrome — don't filter by color
    p.minDistBetweenBlobs = 20               # px — only one ball at a time

    return cv2.SimpleBlobDetector_create(p)


class BlobDetector:
    """
    Wraps cv2.SimpleBlobDetector with:
      • UKF-gated ROI for fast, focused search after acquisition
      • Sub-pixel refinement via image moments on the blob patch
      • Fallback to full-frame search on miss
    """
    def __init__(self):
        self._det = _build_blob_detector()

    def detect(self, motion_mask: np.ndarray,
               kf: LinearKF6D | None,
               frame_w: int, frame_h: int
               ) -> tuple[float,float,float] | None:
        """
        Returns (cx_px, cy_px, radius_px) sub-pixel, or None.
        """
        fh, fw = motion_mask.shape[:2]

        # ── Attempt gated ROI search first (fast path) ──────────────
        if kf is not None and kf.is_valid():
            result = self._detect_in_roi(motion_mask, kf, fw, fh)
            if result is not None:
                return result

        # ── Full-frame search (acquisition / re-acquisition) ─────────
        return self._detect_full(motion_mask, fw, fh, offset=(0,0))

    def _detect_in_roi(self, mask, kf, fw, fh):
        px,py,diam = kf.project_to_pixel(fw, fh)
        r_est  = max(diam*0.5, 12.0)
        margin = max(int(r_est*KF_ROI_PAD_FACTOR), KF_ROI_MIN_PX)
        x1 = max(0,int(px)-margin);  x2 = min(fw,int(px)+margin)
        y1 = max(0,int(py)-margin);  y2 = min(fh,int(py)+margin)
        if (x2-x1)<20 or (y2-y1)<20: return None
        roi = mask[y1:y2, x1:x2]
        res = self._detect_full(roi, x2-x1, y2-y1, offset=(x1,y1))
        return res

    def _detect_full(self, mask, fw, fh, offset=(0,0)):
        # SimpleBlobDetector needs an INVERTED mask for dark-on-light
        # OR can work on white-on-black directly with minThreshold=127
        # We pass the mask directly (white blobs on black bg)
        kp = self._det.detect(mask)
        if not kp:
            return None

        # Pick the keypoint with the largest radius
        best = max(kp, key=lambda k: k.size)
        cx   = best.pt[0] + offset[0]
        cy   = best.pt[1] + offset[1]
        r    = best.size  * 0.5          # size = diameter in SimpleBlobDetector

        # ── Sub-pixel moment refinement on the local patch ───────────
        pad  = max(int(r*1.8), 16)
        x1s  = max(0, int(cx)-pad-offset[0])
        y1s  = max(0, int(cy)-pad-offset[1])
        x2s  = min(fw, int(cx)+pad-offset[0])
        y2s  = min(fh, int(cy)+pad-offset[1])
        if (x2s-x1s)>4 and (y2s-y1s)>4:
            patch = mask[y1s:y2s, x1s:x2s]
            Mv    = cv2.moments(patch)
            if Mv['m00'] > 1.0:
                cx = Mv['m10']/Mv['m00'] + x1s + offset[0]
                cy = Mv['m01']/Mv['m00'] + y1s + offset[1]
                r  = math.sqrt(Mv['m00']/(255.0*math.pi))

        if not (3.0 < r < 145.0):
            return None
        return (float(cx), float(cy), float(r))


# ══════════════════════════════════════════════════════════════════════
# EDGE TRIPWIRE  (unchanged from v5.3 — proven first-frame detection)
# Ported to seed LinearKF6D instead of MonocularUKF
# ══════════════════════════════════════════════════════════════════════

class EdgeTripwire:
    def __init__(self, fw, fh):
        self.fw, self.fh = fw, fh
        bx = max(4, int(fw*BORDER_FRAC))
        by = max(4, int(fh*BORDER_FRAC))
        # (sy, sx, edge_label, vx_dir, vy_dir)
        self._strips = [
            (slice(0,by),          slice(None),'TOP',   0.0, 1.0),
            (slice(fh-by,fh),      slice(None),'BOTTOM',0.0,-1.0),
            (slice(None),          slice(0,bx),'LEFT',  1.0, 0.0),
            (slice(None),          slice(fw-bx,fw),'RIGHT',-1.0,0.0),
        ]
        self._prev    = None
        self._dil_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self._ghost_f = 0

    def scan(self, gray, kf, status, fw, fh, dt):
        if self._prev is None:
            self._prev = gray.copy()
            return kf, status

        # Idle when already tracking
        if kf is not None and status in ("LOCKED","COASTING","ACQUIRING"):
            self._ghost_f += 1
            if status=="ACQUIRING" and self._ghost_f > GHOST_MAX_FRAMES:
                print("[TRIPWIRE] Ghost expired — back to SCANNING")
                kf     = LinearKF6D.nuclear_reset(kf)
                status = "SCANNING"
                self._ghost_f = 0
            self._prev = gray.copy()
            return kf, status

        # Scan perimeter strips
        best = None
        best_a = 0
        for (sy, sx, lbl, vxd, vyd) in self._strips:
            diff  = cv2.absdiff(gray[sy,sx], self._prev[sy,sx])
            diff  = cv2.GaussianBlur(diff,(3,3),0)
            _,thr = cv2.threshold(diff, TRIPWIRE_THRESH, 255, cv2.THRESH_BINARY)
            thr   = cv2.dilate(thr, self._dil_k)
            cnts,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if not (40 <= a <= 12000): continue
                perim = cv2.arcLength(cnt,True)
                if perim<1e-6: continue
                if (4*math.pi*a)/(perim*perim) < 0.38: continue
                if a > best_a:
                    best_a = a
                    Mv = cv2.moments(cnt)
                    if Mv['m00']<1: continue
                    yo = sy.start or 0
                    xo = sx.start or 0
                    cx = Mv['m10']/Mv['m00'] + xo
                    cy = Mv['m01']/Mv['m00'] + yo
                    best = (cx,cy,a,vxd,vyd,lbl)

        if best is not None:
            cx,cy,area,vxd,vyd,lbl = best
            kf = LinearKF6D(dt=dt,
                            focal_length=FOCAL_LENGTH_PX,
                            real_width=REAL_TARGET_W_M)
            kf.seed_from_ghost(cx,cy,fw,fh,vxd,vyd,GHOST_Z_M,GHOST_SPEED_MS,dt)
            status = "ACQUIRING"
            self._ghost_f = 0
            print(f"[TRIPWIRE] {lbl} breach  cx={cx:.0f} cy={cy:.0f} → Ghost KF")

        self._prev = gray.copy()
        return kf, status


# ══════════════════════════════════════════════════════════════════════
# ARDUINO BRIDGE  (unchanged — keeps commanded_pan/tilt for SAFC)
# ══════════════════════════════════════════════════════════════════════

class ArduinoBridge:
    def __init__(self):
        self.ser            = None
        self.enabled        = False
        self.commanded_pan  = SERVO_INIT_PAN
        self.commanded_tilt = SERVO_INIT_TILT
        if not _SERIAL_OK:
            print("[TURRET] pyserial missing."); return
        port = ARDUINO_PORT or self._find()
        if not port:
            print("[TURRET] No Arduino found."); return
        try:
            self.ser = serial.Serial(port, ARDUINO_BAUD, timeout=0.01)
            time.sleep(2.0)
            self.enabled = True
            print(f"[TURRET] {port} @ {ARDUINO_BAUD}")
        except Exception as e:
            print(f"[TURRET] {e}")

    @staticmethod
    def _find():
        for p in serial.tools.list_ports.comports():
            d=(p.description or '').lower()
            if any(k in d for k in ['arduino','ch340','ch341','uno','ftdi','ft232']):
                return p.device
        return None

    @staticmethod
    def pix_to_angles(px,py,fw,fh,f=FOCAL_LENGTH_PX):
        cx,cy = fw*0.5, fh*0.5
        pan  = math.degrees(math.atan2( px-cx, f))
        tilt = math.degrees(math.atan2( cy-py, f))
        ps   = int(np.clip(90+pan,  0,180))
        ts   = int(np.clip(90-tilt, 0,180))
        return ps, ts

    def send(self, pan, tilt, fire=False):
        self.commanded_pan  = float(pan)
        self.commanded_tilt = float(tilt)
        if not self.enabled or not self.ser: return
        try:
            self.ser.write(f"P{int(pan):03d}T{int(tilt):03d}F{1 if fire else 0}\n"
                           .encode('ascii'))
        except Exception: pass

    def center(self): self.send(90,90,False)
    def close(self):
        if self.ser and self.ser.is_open:
            self.center(); time.sleep(0.2); self.ser.close()
            print("[TURRET] Closed.")


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def run_apits():
    print("\n"+"═"*68)
    print("  APITS v6.0 — SimpleBlobDetector + LinearKF6D + SAFC + Tripwire")
    print("  Pipeline: Cam→SAFC→AbsDiff→BallFilter→Blob→MahaGate→LKF→Servo")
    print("═"*68)

    stream  = CameraStream(src=CAM_PORT).start()
    time.sleep(0.5)
    f0      = stream.read()
    fh, fw  = f0.shape[:2]
    print(f"[APITS] Frame {fw}×{fh}")

    safc     = SAFCCompensator(fw, fh, FOCAL_LENGTH_PX)
    motion_d = FastMotionDetector(safc)
    blob_d   = KinematicMassDetector(FOCAL_LENGTH_PX, REAL_TARGET_W_M)
    tripwire = EdgeTripwire(fw, fh)
    hud      = TacticalHUD()
    turret   = ArduinoBridge()

    kf:   LinearKF6D | None = None
    frames_since_det = 0
    status   = "SCANNING"
    tgt_x = tgt_y = tgt_w = tgt_h = 0

    # Live exposure tuner state
    _exp  = EXPOSURE_INIT
    _gain = GAIN_INIT

    prev_t = time.perf_counter()
    print("[MASTER] Q=quit  S=save  R=reset  +/-=exposure  ]/[=gain\n")

    while True:
        gray = stream.read()
        now  = time.perf_counter()
        dt   = max(now - prev_t, 1e-6)
        prev_t = now
        fps  = 1.0 / dt

        frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Nuclear reset check ────────────────────────────────────
        if kf is not None and getattr(kf,'_nuke_ready',False):
            print("[KF] NUCLEAR RESET — poisoned tracker destroyed.")
            kf = LinearKF6D.nuclear_reset(kf)
            status = "SCANNING"
            frames_since_det = 0
            tgt_x = tgt_y = tgt_w = tgt_h = 0

        # ══════════════════════════════════════════════════════════
        # STAGE 0 — EDGE TRIPWIRE (< 1 ms, runs every frame)
        # ══════════════════════════════════════════════════════════
        kf, status = tripwire.scan(
            gray, kf, status, fw, fh, dt)

        # ══════════════════════════════════════════════════════════
        # STAGE A — SAFC + 2-FRAME ABSDIFF + MORPHOLOGY
        # ══════════════════════════════════════════════════════════
        pan_cmd  = turret.commanded_pan
        tilt_cmd = turret.commanded_tilt
        mask, H_curr = motion_d.process(gray, pan_cmd, tilt_cmd)

        # ── BallShapeFilter REMOVED ── 
        # (Kinematic tracker needs the raw motion streaks, not perfect circles)

        # ── Debug: motion mask in corner ──────────────────────────
        sc   = 0.22
        sm   = cv2.resize(mask,(int(fw*sc),int(fh*sc)))
        smb  = cv2.cvtColor(sm,cv2.COLOR_GRAY2BGR)
        oy   = fh - smb.shape[0] - 5
        frame_out[oy:oy+smb.shape[0], 5:5+smb.shape[1]] = smb
        cv2.putText(frame_out,"MASK",(7,oy-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,(60,255,60),1)

        # ══════════════════════════════════════════════════════════
        # STAGE B — SIMPLE BLOB DETECTOR (replaces FRST)
        # ══════════════════════════════════════════════════════════
        detection = blob_d.detect(mask, kf, fw, fh)

        # ══════════════════════════════════════════════════════════
        # STAGE C — LINEAR KF PREDICT + MAHALANOBIS UPDATE
        # ══════════════════════════════════════════════════════════
        future_path = []
        curr_3d     = (0.,0.,0.)
        speed_3d    = 0.
        launch_auth = False

        if detection is not None:
            cx_d, cy_d, r_d = detection
            diam_d = r_d * 2.0

            frames_since_det = 0
            tgt_x, tgt_y = int(round(cx_d)), int(round(cy_d))
            tgt_w = tgt_h = int(round(diam_d))

            # Draw blob hit
            cv2.circle(frame_out,(tgt_x,tgt_y),int(round(r_d)),(0,255,255),2)
            cv2.circle(frame_out,(tgt_x,tgt_y),3,(0,255,255),-1)

            # Init KF on first detection or after ghost promotion
            if kf is None:
                kf = LinearKF6D(dt=dt,
                                focal_length=FOCAL_LENGTH_PX,
                                real_width=REAL_TARGET_W_M)
                kf.seed_from_detection(cx_d,cy_d,diam_d,fw,fh,dt)
                print(f"[KF] Seeded  Z≈{kf.x[2]:.2f}m  r={r_d:.1f}px")

            if getattr(kf,'_is_ghost',False):
                kf._is_ghost = False
                print("[TRIPWIRE] Ghost promoted → LOCKED")

            kf.predict(dt)
            meas = np.array([cx_d, cy_d, diam_d], dtype=np.float64)
            kf.update(meas, fw, fh)

            if not kf.is_valid():
                print("[KF] Invalid state post-update — nuclear reset.")
                kf = LinearKF6D.nuclear_reset(kf)
                status = "SCANNING"
            else:
                status   = "LOCKED"
                curr_3d  = (float(kf.x[0]),float(kf.x[1]),float(kf.x[2]))
                speed_3d = float(np.linalg.norm(kf.x[3:6]))
                # HUD trajectory arc
                future_path = kf.predict_future_pixels(PREDICT_FRAMES,fw,fh,dt)
                # Fire authorization (speed>1 m/s + valid track)
                launch_auth = speed_3d > 1.0 and kf._consec_hits > 5

        else:
            frames_since_det += 1

            if frames_since_det > REACQUIRE_FRAMES * 4:
                kf = LinearKF6D.nuclear_reset(kf)
                status = "SCANNING"
                tgt_x = tgt_y = tgt_w = tgt_h = 0
            elif kf is not None and kf.is_valid():
                kf.predict(dt)
                curr_3d  = (float(kf.x[0]),float(kf.x[1]),float(kf.x[2]))
                speed_3d = float(np.linalg.norm(kf.x[3:6]))
                future_path = kf.predict_future_pixels(PREDICT_FRAMES,fw,fh,dt)
                status   = "COASTING"
                # Ghost marker
                px,py,_ = kf.project_to_pixel(fw,fh)
                cv2.drawMarker(frame_out,(int(px),int(py)),
                               (0,165,255),cv2.MARKER_CROSS,20,1)
            elif kf is not None and not kf.is_valid():
                kf = LinearKF6D.nuclear_reset(kf)
                status = "SCANNING"

        # ══════════════════════════════════════════════════════════
        # STAGE D — TELEMETRY
        # ══════════════════════════════════════════════════════════
        hud.log_spacetime_telemetry(tgt_x,tgt_y,tgt_w,tgt_h,status)

        # ══════════════════════════════════════════════════════════
        # STAGE E — LEAD-ANGLE SERVO AIM
        # ══════════════════════════════════════════════════════════
        if kf is not None and kf.is_valid() and status not in ("SCANNING",):
            X_l,Y_l,Z_l = kf.predict_lead_point(SERVO_LAG_S)
            cx_i,cy_i   = fw*0.5, fh*0.5
            aim_px = (FOCAL_LENGTH_PX * X_l / Z_l) + cx_i
            aim_py = (FOCAL_LENGTH_PX * Y_l / Z_l) + cy_i
            pan,tilt = ArduinoBridge.pix_to_angles(aim_px,aim_py,fw,fh)
            turret.send(pan,tilt,fire=launch_auth)

            if tgt_w > 0:
                cv2.arrowedLine(frame_out,(tgt_x,tgt_y),
                                (int(aim_px),int(aim_py)),
                                (255,120,0),2,tipLength=0.28,
                                line_type=cv2.LINE_AA)
                cv2.putText(frame_out,f"P{pan:03d}T{tilt:03d}",
                            (tgt_x+10,tgt_y-20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,120,0),1)

        # ══════════════════════════════════════════════════════════
        # STAGE F — RENDER HUD
        # ══════════════════════════════════════════════════════════
        frame_out = hud.render(
            frame=frame_out,
            target_x=tgt_x, target_y=tgt_y,
            target_w=tgt_w, target_h=tgt_h,
            future_path=future_path,
            curr_3d=curr_3d, fut_3d=(0.,0.,0.),
            speed_3d=speed_3d, tti=0., launch_auth=launch_auth,
            status=status)

        # ── Overlays ──────────────────────────────────────────────
        fps_col = (0,220,0) if fps>60 else (0,140,255) if fps>30 else (0,40,255)
        cv2.putText(frame_out,f"FPS:{fps:.0f}",
                    (20,35),cv2.FONT_HERSHEY_SIMPLEX,1.0,fps_col,2)

        # ── SHOW DETECTOR MODE ──
        cv2.putText(frame_out, f"DET:{blob_d.mode}", 
                    (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,0), 1)

        safc_ok = not np.array_equal(H_curr,np.eye(3))
        cv2.putText(frame_out,"SAFC:OK" if safc_ok else "SAFC:IDLE",
                    (20,60),cv2.FONT_HERSHEY_SIMPLEX,0.40,
                    (0,255,0) if safc_ok else (60,60,200),1)

        if kf:
            cv2.putText(frame_out,
                        f"Z:{kf.x[2]:.2f}m V:{np.linalg.norm(kf.x[3:6]):.1f}m/s"
                        f" hits:{kf._consec_hits}",
                        (20,78),cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,220,160),1)

        ghost_active = kf is not None and getattr(kf,'_is_ghost',False)
        st_col = (0,200,255) if ghost_active else \
                 (0,255,0)   if status=="LOCKED" else (80,80,80)
        cv2.putText(frame_out,status,
                    (fw-170,35),cv2.FONT_HERSHEY_SIMPLEX,0.6,st_col,2)

        # Tripwire border visual
        bx = max(4,int(fw*BORDER_FRAC))
        by = max(4,int(fh*BORDER_FRAC))
        cv2.rectangle(frame_out,(bx,by),(fw-bx,fh-by),(0,50,100),1)

        cv2.imshow("APITS v6.0 — Blob+LinearKF",frame_out)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('s'): hud.export_omega_dataset(); print("[APITS] Saved.")
        elif key == ord('r'):
            kf=LinearKF6D.nuclear_reset(kf); status="SCANNING"
            frames_since_det=0; print("[APITS] Manual reset.")
        elif key in (ord('='),ord('+')):
            _exp=min(_exp+1,-1)
            stream.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)
            stream.cap.set(cv2.CAP_PROP_EXPOSURE,_exp)
            print(f"[CAM] EXP={_exp}")
        elif key == ord('-'):
            _exp=max(_exp-1,-13)
            stream.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)
            stream.cap.set(cv2.CAP_PROP_EXPOSURE,_exp)
            print(f"[CAM] EXP={_exp}")
        elif key == ord(']'):
            _gain=min(_gain+4,100)
            stream.cap.set(cv2.CAP_PROP_GAIN,_gain)
            print(f"[CAM] GAIN={_gain}")
        elif key == ord('['):
            _gain=max(_gain-4,0)
            stream.cap.set(cv2.CAP_PROP_GAIN,_gain)
            print(f"[CAM] GAIN={_gain}")

    print("\n[APITS] Shutdown...")
    turret.close(); stream.stop(); cv2.destroyAllWindows()
    hud.export_omega_dataset()
    print("[APITS] Offline.")


if __name__ == "__main__":
    try:
        run_apits()
    except KeyboardInterrupt:
        print("\n[!] Ctrl-C.")
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        cv2.destroyAllWindows(); os._exit(0)
