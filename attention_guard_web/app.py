"""
Attention Guard — Flask Web UI
==============================
Run:  python app.py
Then open http://localhost:5000 in your browser.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
from flask import Flask, Response, jsonify, request, render_template

app = Flask(__name__)

# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
state = {
    "running":          False,
    "status":           "stopped",     # stopped | focused | drifting | alert
    "reason":           "",
    "alert_word":       "STOP",
    "sensitivity":      1.0,           # 0.5 = very sensitive, 2.0 = lenient
    "grace_period":     1.5,
    "session_start":    None,
    "focused_start":    None,
    "total_focused":    0.0,           # seconds focused this session
    "total_distracted": 0.0,
    "alert_count":      0,
    "last_frame":       None,
    "debug":            {},
}
state_lock = threading.Lock()

# Thresholds (base values, scaled by sensitivity)
BASE_PITCH_THRESHOLD     = 0.62
BASE_GAZE_DOWN_THRESHOLD = 0.68
BASE_YAW_THRESHOLD       = 0.26
BASE_GAZE_SIDE_THRESHOLD = 0.28


# ─────────────────────────────────────────────
#  TTS ENGINE
# ─────────────────────────────────────────────
class AlertEngine:
    def __init__(self):
        self._active = False
        self._word   = "STOP"
        self._lock   = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _speak(self):
        try:
            e = pyttsx3.init()
            e.setProperty("rate", 155)
            e.setProperty("volume", 1.0)
            with self._lock: word = self._word
            e.say(word)
            e.runAndWait()
            e.stop()
        except Exception:
            pass

    def _loop(self):
        while True:
            with self._lock: active = self._active
            if active:
                self._speak()
                time.sleep(0.4)
            else:
                time.sleep(0.05)

    def start(self, word=None):
        with self._lock:
            if word: self._word = word
            self._active = True

    def stop(self):
        with self._lock: self._active = False

    def set_word(self, word):
        with self._lock: self._word = word

    @property
    def is_active(self):
        with self._lock: return self._active

alert_engine = AlertEngine()


# ─────────────────────────────────────────────
#  DETECTION HELPERS
# ─────────────────────────────────────────────
def pt(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_head_pitch_ratio(lm, w, h):
    forehead = pt(lm, 10,  w, h)
    nose_tip = pt(lm, 1,   w, h)
    chin     = pt(lm, 152, w, h)
    face_h   = chin[1] - forehead[1]
    if face_h < 1: return 0.5
    return (nose_tip[1] - forehead[1]) / face_h

def get_yaw_ratio(lm, w, h):
    lc       = pt(lm, 234, w, h)
    rc       = pt(lm, 454, w, h)
    nose_tip = pt(lm, 1,   w, h)
    fw       = dist(lc, rc)
    if fw < 1: return 0.0
    return (nose_tip[0] - (lc[0]+rc[0])/2) / fw

def get_eye_gaze(lm, w, h):
    try:
        l_top  = pt(lm,159,w,h); l_bot  = pt(lm,145,w,h)
        l_left = pt(lm,33, w,h); l_right= pt(lm,133,w,h)
        l_iris = pt(lm,468,w,h)
        r_top  = pt(lm,386,w,h); r_bot  = pt(lm,374,w,h)
        r_left = pt(lm,362,w,h); r_right= pt(lm,263,w,h)
        r_iris = pt(lm,473,w,h)
        l_eh = max(l_bot[1]-l_top[1],1); r_eh = max(r_bot[1]-r_top[1],1)
        gaze_v = ((l_iris[1]-l_top[1])/l_eh + (r_iris[1]-r_top[1])/r_eh)/2
        l_ew = max(l_right[0]-l_left[0],1); r_ew = max(r_right[0]-r_left[0],1)
        gaze_h = ((l_iris[0]-l_left[0])/l_ew + (r_iris[0]-r_left[0])/r_ew)/2
        return gaze_v, gaze_h
    except:
        return 0.5, 0.5

def classify(lm, w, h, sensitivity):
    s         = max(sensitivity, 0.1)
    pitch     = get_head_pitch_ratio(lm, w, h)
    yaw       = get_yaw_ratio(lm, w, h)
    gv, gh    = get_eye_gaze(lm, w, h)
    gs        = abs(gh - 0.5)

    pt_thresh = BASE_PITCH_THRESHOLD     + (s - 1.0) * 0.06
    gd_thresh = BASE_GAZE_DOWN_THRESHOLD + (s - 1.0) * 0.06
    ya_thresh = BASE_YAW_THRESHOLD       + (s - 1.0) * 0.04
    gs_thresh = BASE_GAZE_SIDE_THRESHOLD + (s - 1.0) * 0.04

    reasons = []
    if abs(yaw) > ya_thresh:
        reasons.append("looking sideways")
    if (pitch > pt_thresh) or (gv > gd_thresh):
        reasons.append("looking down")
    elif (pitch > pt_thresh - 0.03) and (gv > gd_thresh - 0.03):
        reasons.append("looking down")
    if gs > gs_thresh:
        reasons.append("eyes sideways")

    return len(reasons) > 0, reasons, {
        "p": f"{pitch:.2f}", "gv": f"{gv:.2f}",
        "y": f"{yaw:+.2f}", "gh": f"{gh:.2f}"
    }


# ─────────────────────────────────────────────
#  FACE UI ON FRAME
# ─────────────────────────────────────────────
def draw_brackets(img, x1, y1, x2, y2, col, length=20, thick=2):
    for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+dx*length,cy),col,thick,cv2.LINE_AA)
        cv2.line(img,(cx,cy),(cx,cy+dy*length),col,thick,cv2.LINE_AA)

def draw_face_ui(frame, lm, color):
    fh,fw = frame.shape[:2]
    xs=[int(p.x*fw) for p in lm]; ys=[int(p.y*fh) for p in lm]
    pad=22
    bx1=max(min(xs)-pad,0); by1=max(min(ys)-pad,0)
    bx2=min(max(xs)+pad,fw); by2=min(max(ys)+pad,fh)

    glow=frame.copy()
    draw_brackets(glow,bx1-3,by1-3,bx2+3,by2+3,color,26,6)
    cv2.addWeighted(glow,0.18,frame,0.82,0,frame)
    draw_brackets(frame,bx1,by1,bx2,by2,color,20,2)

    sy=by1+int((time.time()%1.8)/1.8*(by2-by1))
    sc=frame.copy()
    cv2.line(sc,(bx1+3,sy),(bx2-3,sy),color,1)
    cv2.addWeighted(sc,0.22,frame,0.78,0,frame)

    label="YOU"; font=cv2.FONT_HERSHEY_DUPLEX; fs=0.5
    (lw,lh),_=cv2.getTextSize(label,font,fs,1)
    lx=bx1+(bx2-bx1)//2-lw//2; ly=max(by1-10,lh+8); pp=6
    pill=frame.copy()
    cv2.rectangle(pill,(lx-pp,ly-lh-pp),(lx+lw+pp,ly+pp),color,-1)
    cv2.addWeighted(pill,0.6,frame,0.4,0,frame)
    cv2.putText(frame,label,(lx,ly),font,fs,(255,255,255),1,cv2.LINE_AA)


# ─────────────────────────────────────────────
#  CAMERA THREAD
# ─────────────────────────────────────────────
def camera_loop():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        with state_lock:
            state["status"] = "error"
        return

    inattentive_since = None
    last_tick         = time.time()

    while True:
        with state_lock:
            running     = state["running"]
            sensitivity = state["sensitivity"]
            grace       = state["grace_period"]
            alert_word  = state["alert_word"]

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        now   = time.time()
        dt    = now - last_tick
        last_tick = now

        if not running:
            # Show idle frame
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(w,h),(10,10,10),-1)
            cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
            msg="GUARD STOPPED"
            (mw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_DUPLEX,0.9,2)
            cv2.putText(frame,msg,(w//2-mw//2,h//2),
                        cv2.FONT_HERSHEY_DUPLEX,0.9,(60,60,60),2,cv2.LINE_AA)
            _,buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state_lock:
                state["last_frame"] = buf.tobytes()
                state["status"]     = "stopped"
            inattentive_since = None
            alert_engine.stop()
            time.sleep(0.05)
            continue

        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            inattentive, reasons, dbg = classify(lm, w, h, sensitivity)

            if inattentive:
                if inattentive_since is None:
                    inattentive_since = now
                elapsed = now - inattentive_since

                with state_lock:
                    state["total_distracted"] += dt
                    state["focused_start"]     = None

                if elapsed >= grace:
                    if not alert_engine.is_active:
                        alert_engine.start(alert_word)
                        state["alert_count"] += 1
                    color = (0, 0, 210)
                    with state_lock:
                        state["status"] = "alert"
                        state["reason"] = " + ".join(reasons)
                else:
                    color = (0, 140, 255)
                    with state_lock:
                        state["status"] = "drifting"
                        state["reason"] = f"{elapsed:.1f}s"
            else:
                inattentive_since = None
                alert_engine.stop()
                color = (0, 210, 90)
                with state_lock:
                    if state["focused_start"] is None:
                        state["focused_start"] = now
                    state["total_focused"] += dt
                    state["status"] = "focused"
                    state["reason"] = ""

            with state_lock:
                state["debug"] = dbg

            draw_face_ui(frame, lm, color)

            if alert_engine.is_active:
                bd=frame.copy()
                cv2.rectangle(bd,(0,0),(w,h),(0,0,180),16)
                cv2.addWeighted(bd,0.45,frame,0.55,0,frame)
        else:
            inattentive_since = None
            alert_engine.stop()
            with state_lock:
                state["status"] = "no_face"
                state["reason"] = ""

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with state_lock:
            state["last_frame"] = buf.tobytes()

    cap.release()

threading.Thread(target=camera_loop, daemon=True).start()


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    while True:
        with state_lock:
            frame = state["last_frame"]
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    with state_lock:
        s = state.copy()
    session_secs = (time.time() - s["session_start"]) if s["session_start"] else 0
    total = max(s["total_focused"] + s["total_distracted"], 1)
    return jsonify({
        "running":      s["running"],
        "status":       s["status"],
        "reason":       s["reason"],
        "alert_count":  s["alert_count"],
        "session_secs": round(session_secs),
        "focused_pct":  round(s["total_focused"] / total * 100),
        "focused_secs": round(s["total_focused"]),
        "debug":        s["debug"],
        "sensitivity":  s["sensitivity"],
        "alert_word":   s["alert_word"],
        "grace_period": s["grace_period"],
    })

@app.route("/api/start", methods=["POST"])
def api_start():
    with state_lock:
        state["running"]          = True
        state["session_start"]    = time.time()
        state["total_focused"]    = 0.0
        state["total_distracted"] = 0.0
        state["alert_count"]      = 0
        state["focused_start"]    = None
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        state["running"]       = False
        state["session_start"] = None
    alert_engine.stop()
    return jsonify({"ok": True})

@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.json
    with state_lock:
        if "sensitivity"  in data: state["sensitivity"]  = float(data["sensitivity"])
        if "alert_word"   in data: state["alert_word"]   = str(data["alert_word"])
        if "grace_period" in data: state["grace_period"] = float(data["grace_period"])
    alert_engine.set_word(state["alert_word"])
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Starting Attention Guard at http://localhost:5000")
    app.run(debug=False, threaded=True, port=5000)