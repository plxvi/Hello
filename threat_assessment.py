"""
Behavioral Threat Assessment System
Analyzes live video and audio to detect potential threat indicators.

DISCLAIMER: This tool is intended for authorized security research and
professional security contexts only. Results are probabilistic and must
be reviewed by trained human operators. Do not use as sole basis for
any decision. Ensure compliance with applicable laws before deployment.

Dependencies:
    pip install opencv-python mediapipe speechrecognition transformers
    pip install torch numpy pyaudio
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import queue
import speech_recognition as sr
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ThreatLevel(Enum):
    LOW = ("LOW", (0, 200, 0))          # green
    MODERATE = ("MODERATE", (0, 165, 255))  # orange
    HIGH = ("HIGH", (0, 0, 255))        # red
    CRITICAL = ("CRITICAL", (128, 0, 128))  # purple

    def __init__(self, label: str, color: tuple):
        self.label = label
        self.color = color


@dataclass
class BehaviorSignals:
    """Aggregated behavioral signals at a point in time."""
    # Physical
    arms_raised: bool = False
    rapid_movement: bool = False
    aggressive_stance: bool = False
    concealment_behavior: bool = False
    erratic_motion: bool = False
    movement_velocity: float = 0.0

    # Facial
    anger_score: float = 0.0
    fear_score: float = 0.0
    neutral_score: float = 1.0

    # Speech
    threat_keywords_detected: bool = False
    negative_sentiment_score: float = 0.0
    speech_rate_elevated: bool = False
    latest_transcript: str = ""

    # Derived
    threat_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.LOW


# ---------------------------------------------------------------------------
# Video / Pose Analyzer
# ---------------------------------------------------------------------------

class VideoAnalyzer:
    """Analyzes video frames for physical behavioral indicators."""

    # Indices for MediaPipe Pose landmarks
    _LEFT_SHOULDER  = 11
    _RIGHT_SHOULDER = 12
    _LEFT_ELBOW     = 13
    _RIGHT_ELBOW    = 14
    _LEFT_WRIST     = 15
    _RIGHT_WRIST    = 16
    _LEFT_HIP       = 23
    _RIGHT_HIP      = 24
    _NOSE           = 0

    def __init__(self, motion_history_len: int = 30):
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw = mp.solutions.drawing_utils
        self._pose_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

        # Rolling history of landmark positions for velocity/jitter analysis
        self._landmark_history: deque = deque(maxlen=motion_history_len)
        self._velocity_history: deque = deque(maxlen=motion_history_len)

    def analyze_frame(self, frame: np.ndarray) -> tuple[np.ndarray, BehaviorSignals]:
        """Run full analysis on a single BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        pose_results = self._pose.process(rgb)
        face_results = self._face_mesh.process(rgb)

        rgb.flags.writeable = True
        annotated = frame.copy()

        signals = BehaviorSignals()

        if pose_results.pose_landmarks:
            self._draw.draw_landmarks(
                annotated, pose_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._pose_spec,
            )
            signals = self._analyze_pose(pose_results.pose_landmarks.landmark, signals)

        if face_results.multi_face_landmarks:
            for fl in face_results.multi_face_landmarks:
                self._draw.draw_landmarks(
                    annotated, fl,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                        .get_default_face_mesh_contours_style(),
                )

        return annotated, signals

    def _analyze_pose(self, lm, signals: BehaviorSignals) -> BehaviorSignals:
        def pt(idx):
            return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

        nose       = pt(self._NOSE)
        l_shoulder = pt(self._LEFT_SHOULDER)
        r_shoulder = pt(self._RIGHT_SHOULDER)
        l_wrist    = pt(self._LEFT_WRIST)
        r_wrist    = pt(self._RIGHT_WRIST)
        l_elbow    = pt(self._LEFT_ELBOW)
        r_elbow    = pt(self._RIGHT_ELBOW)
        l_hip      = pt(self._LEFT_HIP)
        r_hip      = pt(self._RIGHT_HIP)

        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2

        # Arms raised above shoulder line
        signals.arms_raised = (
            l_wrist[1] < shoulder_mid_y - 0.05 or
            r_wrist[1] < shoulder_mid_y - 0.05
        )

        # Wide aggressive stance: elbows flared outward past shoulders
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        elbow_width    = abs(l_elbow[0] - r_elbow[0])
        signals.aggressive_stance = elbow_width > shoulder_width * 1.4

        # Concealment: arms crossed tightly in front of torso
        wrist_dist = np.linalg.norm(l_wrist[:2] - r_wrist[:2])
        signals.concealment_behavior = (
            wrist_dist < shoulder_width * 0.3 and
            l_wrist[1] > shoulder_mid_y and
            r_wrist[1] > shoulder_mid_y
        )

        # Movement velocity via centroid tracking
        centroid = np.mean([l_hip[:2], r_hip[:2]], axis=0)
        self._landmark_history.append(centroid)
        if len(self._landmark_history) >= 5:
            recent = list(self._landmark_history)[-5:]
            velocity = float(np.mean([
                np.linalg.norm(np.array(recent[i]) - np.array(recent[i-1]))
                for i in range(1, len(recent))
            ]))
            self._velocity_history.append(velocity)
            signals.movement_velocity = velocity
            signals.rapid_movement = velocity > 0.015

        # Erratic motion: high variance in recent velocities
        if len(self._velocity_history) >= 10:
            signals.erratic_motion = float(np.std(self._velocity_history)) > 0.008

        return signals

    def release(self):
        self._pose.close()
        self._face_mesh.close()


# ---------------------------------------------------------------------------
# Speech Analyzer
# ---------------------------------------------------------------------------

THREAT_KEYWORDS = {
    # high-weight phrases
    "i will kill": 0.9,
    "i'm going to kill": 0.9,
    "blow this up": 0.9,
    "bomb": 0.75,
    "shoot": 0.7,
    "knife": 0.6,
    "weapon": 0.65,
    "attack": 0.6,
    "destroy": 0.55,
    "hurt you": 0.7,
    "i have a gun": 0.95,
    "threaten": 0.5,
    "hostage": 0.8,
}


class SpeechAnalyzer:
    """Continuous background speech capture and analysis."""

    def __init__(self, signals_queue: queue.Queue, language: str = "en-US"):
        self._queue = signals_queue
        self._language = language
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = 300
        self._recognizer.dynamic_energy_threshold = True
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._word_timestamps: deque = deque(maxlen=50)

        # Optional sentiment model
        self._sentiment = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self._sentiment = hf_pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,  # CPU
                )
            except Exception:
                pass

    def start(self):
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _listen_loop(self):
        try:
            mic = sr.Microphone()
        except Exception:
            return  # No microphone available

        with mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1)

        while not self._stop_event.is_set():
            try:
                with mic as source:
                    audio = self._recognizer.listen(source, timeout=2, phrase_time_limit=6)
                try:
                    text = self._recognizer.recognize_google(audio, language=self._language)
                    self._queue.put(("transcript", text, time.time()))
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
            except sr.WaitTimeoutError:
                pass
            except Exception:
                time.sleep(0.5)

    def analyze_transcript(self, text: str, timestamp: float) -> dict:
        text_lower = text.lower()
        result = {
            "transcript": text,
            "threat_keywords": False,
            "keyword_score": 0.0,
            "sentiment_score": 0.0,
            "speech_rate_elevated": False,
        }

        # Keyword matching
        max_kw_score = 0.0
        for phrase, weight in THREAT_KEYWORDS.items():
            if phrase in text_lower:
                max_kw_score = max(max_kw_score, weight)
        result["threat_keywords"] = max_kw_score > 0
        result["keyword_score"] = max_kw_score

        # Speech rate (words per second estimate)
        word_count = len(text.split())
        self._word_timestamps.append((timestamp, word_count))
        if len(self._word_timestamps) >= 2:
            t_delta = self._word_timestamps[-1][0] - self._word_timestamps[0][0]
            total_words = sum(w for _, w in self._word_timestamps)
            if t_delta > 0:
                wps = total_words / t_delta
                result["speech_rate_elevated"] = wps > 3.5  # ~210 wpm

        # Sentiment
        if self._sentiment and text.strip():
            try:
                out = self._sentiment(text[:512])[0]
                # NEGATIVE label → threat signal
                if out["label"] == "NEGATIVE":
                    result["sentiment_score"] = float(out["score"])
            except Exception:
                pass

        return result


# ---------------------------------------------------------------------------
# Threat Scorer
# ---------------------------------------------------------------------------

class ThreatScorer:
    """Combines physical and speech signals into a unified threat score."""

    WEIGHTS = {
        "arms_raised":              0.10,
        "rapid_movement":           0.12,
        "aggressive_stance":        0.15,
        "concealment_behavior":     0.08,
        "erratic_motion":           0.10,
        "threat_keywords":          0.30,
        "negative_sentiment":       0.10,
        "speech_rate_elevated":     0.05,
    }

    THRESHOLDS = {
        ThreatLevel.CRITICAL: 0.70,
        ThreatLevel.HIGH:     0.45,
        ThreatLevel.MODERATE: 0.20,
        ThreatLevel.LOW:      0.00,
    }

    def score(self, signals: BehaviorSignals) -> tuple[float, ThreatLevel]:
        s = 0.0
        s += self.WEIGHTS["arms_raised"]          * float(signals.arms_raised)
        s += self.WEIGHTS["rapid_movement"]        * float(signals.rapid_movement)
        s += self.WEIGHTS["aggressive_stance"]     * float(signals.aggressive_stance)
        s += self.WEIGHTS["concealment_behavior"]  * float(signals.concealment_behavior)
        s += self.WEIGHTS["erratic_motion"]        * float(signals.erratic_motion)
        s += self.WEIGHTS["threat_keywords"]       * float(signals.threat_keywords_detected)
        s += self.WEIGHTS["negative_sentiment"]    * signals.negative_sentiment_score
        s += self.WEIGHTS["speech_rate_elevated"]  * float(signals.speech_rate_elevated)

        # Keyword weight already includes severity; add bonus for high-confidence keyword
        # combined with movement
        if signals.threat_keywords_detected and signals.rapid_movement:
            s = min(1.0, s + 0.10)

        s = max(0.0, min(1.0, s))

        level = ThreatLevel.LOW
        for tl in (ThreatLevel.CRITICAL, ThreatLevel.HIGH, ThreatLevel.MODERATE):
            if s >= self.THRESHOLDS[tl]:
                level = tl
                break

        return s, level


# ---------------------------------------------------------------------------
# HUD overlay rendering
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, signals: BehaviorSignals, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent panel
    panel_w, panel_h = 340, 260
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    level   = signals.threat_level
    color   = level.color
    score_pct = int(signals.threat_score * 100)

    # Title bar
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 40), color, -1)
    cv2.putText(frame, f"THREAT: {level.label}  ({score_pct}%)",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Threat score bar
    bar_x, bar_y, bar_w, bar_h = 18, 48, panel_w - 16, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * signals.threat_score)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)

    def indicator(y: int, label: str, active: bool, value: str = ""):
        dot_color = (0, 220, 0) if not active else (0, 0, 255)
        cv2.circle(frame, (22, y), 5, dot_color, -1)
        txt = f"{label}"
        if value:
            txt += f"  {value}"
        cv2.putText(frame, txt, (32, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

    y0 = 78
    dy = 20
    indicator(y0,       "Arms Raised",        signals.arms_raised)
    indicator(y0+dy,    "Rapid Movement",     signals.rapid_movement,
              f"v={signals.movement_velocity:.4f}")
    indicator(y0+2*dy,  "Aggressive Stance",  signals.aggressive_stance)
    indicator(y0+3*dy,  "Concealment",        signals.concealment_behavior)
    indicator(y0+4*dy,  "Erratic Motion",     signals.erratic_motion)
    indicator(y0+5*dy,  "Threat Keywords",    signals.threat_keywords_detected)
    indicator(y0+6*dy,  "Elevated Speech",    signals.speech_rate_elevated)
    indicator(y0+7*dy,  "Neg. Sentiment",     signals.negative_sentiment_score > 0.5,
              f"{signals.negative_sentiment_score:.2f}")

    # Transcript
    if signals.latest_transcript:
        text = signals.latest_transcript[:60]
        cv2.putText(frame, f'"{text}"', (18, y0 + 8*dy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 90, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Border flash on HIGH/CRITICAL
    if level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)

    return frame


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def run(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    video_analyzer = VideoAnalyzer()
    scorer = ThreatScorer()

    speech_queue: queue.Queue = queue.Queue()
    speech_analyzer = SpeechAnalyzer(speech_queue)
    speech_analyzer.start()

    # Persistent speech signals (updated asynchronously)
    speech_state = {
        "transcript": "",
        "threat_keywords": False,
        "keyword_score": 0.0,
        "sentiment_score": 0.0,
        "speech_rate_elevated": False,
    }

    prev_time = time.time()

    print("[INFO] Starting threat assessment. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Drain speech queue
            while not speech_queue.empty():
                try:
                    event_type, text, ts = speech_queue.get_nowait()
                    if event_type == "transcript":
                        result = speech_analyzer.analyze_transcript(text, ts)
                        speech_state.update(result)
                        print(f"[SPEECH] {text}")
                except queue.Empty:
                    break

            # Analyze video frame
            annotated, signals = video_analyzer.analyze_frame(frame)

            # Merge speech signals
            signals.threat_keywords_detected = speech_state["threat_keywords"]
            signals.negative_sentiment_score = speech_state["sentiment_score"]
            signals.speech_rate_elevated     = speech_state["speech_rate_elevated"]
            signals.latest_transcript        = speech_state["transcript"]

            # Score
            signals.threat_score, signals.threat_level = scorer.score(signals)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Render HUD
            output = draw_hud(annotated, signals, fps)

            cv2.imshow("Behavioral Threat Assessment", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        speech_analyzer.stop()
        video_analyzer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Session ended.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral Threat Assessment System")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    run(camera_index=args.camera)
