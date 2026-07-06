"""Hybrid phone monitor for Focus Buddy.

Adds hand/finger proximity to the YOLO + OpenCV tracker pipeline.

Run:
    python yolo_detection_hand.py --camera 0 --model yolov8s.pt --detection-hz 5 --process-width 640 --yolo-imgsz 320

Optional dependencies:
    pip install ultralytics opencv-contrib-python mediapipe
    # Newer MediaPipe builds also require a local hand_landmarker.task model file.

Main logic:
- YOLO detects the phone at low frequency.
- OpenCV tracker follows the phone between YOLO detections.
- MediaPipe Hands detects hand landmarks at low/medium frequency.
- HAND_NEAR means the user's fingers are close to the phone while the phone is
  still stable, useful for a soft robot reaction.
- IN_HAND requires hand proximity plus phone motion, or disappearance after a
  HAND_NEAR state, useful for the actual distraction alert.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("hybrid_phone_monitor")

BBox = Tuple[float, float, float, float]  # x, y, w, h in processed-frame coordinates
Point = Tuple[float, float]


class PhoneState(str, Enum):
    UNKNOWN = "UNKNOWN"
    RESTING = "RESTING"
    HAND_NEAR = "HAND_NEAR"
    MOVING = "MOVING"
    IN_HAND = "IN_HAND"
    MISSING = "MISSING"


@dataclass
class Observation:
    timestamp: float
    bbox: Optional[BBox]
    confidence: Optional[float]
    source: str  # "yolo", "tracker", or "none"
    speed: float
    state: PhoneState
    event: Optional[str]

    # Hand/finger proximity fields.
    hand_near: bool
    phone_in_hand: bool
    hand_bbox: Optional[BBox]
    fingertip_points: List[Point]
    min_hand_phone_distance: Optional[float]  # normalized by phone bbox diagonal


@dataclass
class HandObservation:
    timestamp: float
    all_points: List[Point]
    fingertip_points: List[Point]
    bbox: Optional[BBox]


class JsonlEventLogger:
    """Small JSONL logger for distraction events."""

    def __init__(self, path: str = "focus_buddy_events.jsonl") -> None:
        # Create path
        self.path = Path(path)

    def log(self, obs: Observation) -> None:
        # Only if there is an event it saves the observation
        if obs.event is None:
            return
        # Create a dictionary
        record = asdict(obs)
        record["state"] = obs.state.value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class MediaPipeHandDetector:
    """Thin wrapper around MediaPipe hand landmarks.
    """

    FINGERTIP_IDS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky tips

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
        model_asset_path: str = "hand_landmarker.task",
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_asset_path = model_asset_path

        self.mp = None #Mediapipe
        self.mp_hands = None #Mediapipe hands using solution attribute (but only old version)
        self.hands = None
        self.landmarker = None # Mediapipe new version witout solution attribute
        self.backend = "none" # Mediapipe with solution attribute or with .task file?
        self.initialized = False # Is mediapipe initialized
        self.available = False # In mediapipe avilable after the initialization?
        self.last_timestamp_ms = -1

    def initialize(self) -> bool:
        if self.initialized:
            return self.available
        self.initialized = True

        try:
            import mediapipe as mp
            self.mp = mp

            # 1) Prefer the old API if this installation still has it. The old API has the solution attribute
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    model_complexity=0,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.backend = "solutions"
                self.available = True
                logger.info("MediaPipe Hands initialized with legacy mp.solutions backend")
                return True

            # 2) Fallback for newer MediaPipe builds: Tasks API.
            model_path = Path(self.model_asset_path)
            if not model_path.exists():
                logger.warning(
                    "MediaPipe Tasks API is available, but the hand landmarker model is missing: %s. "
                    "Download hand_landmarker.task or pass --hand-landmarker-model /path/to/hand_landmarker.task",
                    model_path,
                )
                self.available = False
                return False

            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.backend = "tasks"
            self.available = True
            logger.info("MediaPipe Hands initialized with Tasks HandLandmarker backend")
            return True

        except Exception as exc:
            logger.warning(
                "MediaPipe hand detector is not available: %s. Install with: pip install mediapipe",
                exc,
            )
            self.available = False
            return False

    def detect(self, frame_bgr: np.ndarray, now: float) -> HandObservation:
        if not self.initialize():
            return HandObservation(now, [], [], None)

        if self.backend == "solutions" and self.hands is not None:
            return self._detect_with_solutions(frame_bgr, now)

        if self.backend == "tasks" and self.landmarker is not None and self.mp is not None:
            return self._detect_with_tasks(frame_bgr, now)

        return HandObservation(now, [], [], None)

    def _detect_with_solutions(self, frame_bgr: np.ndarray, now: float) -> HandObservation:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self.hands.process(frame_rgb)

        all_points: List[Point] = []
        fingertip_points: List[Point] = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    px = float(landmark.x * w)
                    py = float(landmark.y * h)
                    all_points.append((px, py))
                    if idx in self.FINGERTIP_IDS:
                        fingertip_points.append((px, py))

        bbox = self._points_bbox(all_points) if all_points else None
        return HandObservation(now, all_points, fingertip_points, bbox)

    def _detect_with_tasks(self, frame_bgr: np.ndarray, now: float) -> HandObservation:
        h, w = frame_bgr.shape[:2]
        # ATtention: The MediaPipe Tasks API requires RGB images
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Save image in contigous way --> Mediapipe requires order, no strides
        frame_rgb = np.ascontiguousarray(frame_rgb)
        # Create an object 'image' of mediapipe
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(now * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        all_points: List[Point] = []
        fingertip_points: List[Point] = []
        # results are a list of hand landmarks, each with 21 points.
        # Results are 'relative' coordinates in [0, 1] range, so we need to scale them to the image size.
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks):
                    px = float(landmark.x * w)
                    py = float(landmark.y * h)
                    all_points.append((px, py))
                    if idx in self.FINGERTIP_IDS:
                        fingertip_points.append((px, py))

        bbox = self._points_bbox(all_points) if all_points else None
        return HandObservation(now, all_points, fingertip_points, bbox)

    @staticmethod
    def _points_bbox(points: List[Point]) -> BBox:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


class HybridPhoneMonitor:
    """YOLO low-rate detector + OpenCV high-rate tracker + hand proximity.

    Key idea:
    - The camera loop can run at 15-30 FPS.
    - YOLO runs only every 1 / detection_hz seconds.
    - In intermediate frames, an OpenCV tracker updates the last phone bbox.
    - MediaPipe Hands runs at a separate hand_detection_hz.
    - The state machine separates HAND_NEAR from IN_HAND:
      HAND_NEAR = hand close to a stable phone;
      IN_HAND = hand close plus phone motion, or phone disappears after HAND_NEAR.
    """

    PHONE_CLASS_ID = 67  # COCO "cell phone"

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        confidence: float = 0.45,
        detection_hz: float = 5.0,
        process_width: int = 640, # Images smaller --> Yolo works on smaller images, Tracker is faster, MediaPipe is faster, lower moemory, lower cpu 
        yolo_imgsz: int = 320, # Smaller the size, 
        tracker_type: str = "KCF",
        stable_speed_threshold: float = 0.035,
        motion_speed_threshold: float = 0.16,
        stable_confirm_frames: int = 15,
        moving_confirm_frames: int = 3,
        missing_confirm_frames: int = 8,
        cooldown_seconds: float = 8.0,
        enable_hand_proximity: bool = True,
        hand_detection_hz: float = 10.0,
        hand_max_age_seconds: float = 0.25,
        hand_phone_distance_threshold: float = 0.12,
        hand_confirm_frames: int = 3,
        hand_release_frames: int = 4,
        in_hand_motion_speed_threshold: float = 0.06,
        in_hand_motion_confirm_frames: int = 2,
        hand_min_detection_confidence: float = 0.45,
        hand_min_tracking_confidence: float = 0.45,
        hand_landmarker_model_path: str = "hand_landmarker.task",
        event_logger: Optional[JsonlEventLogger] = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.detection_period = 1.0 / max(detection_hz, 0.1)
        self.process_width = process_width
        self.yolo_imgsz = yolo_imgsz
        self.tracker_type = tracker_type.upper()

        self.stable_speed_threshold = stable_speed_threshold
        self.motion_speed_threshold = motion_speed_threshold
        self.stable_confirm_frames = stable_confirm_frames
        self.moving_confirm_frames = moving_confirm_frames
        self.missing_confirm_frames = missing_confirm_frames
        self.cooldown_seconds = cooldown_seconds

        self.enable_hand_proximity = enable_hand_proximity
        self.hand_detection_period = 1.0 / max(hand_detection_hz, 0.1)
        self.hand_max_age_seconds = hand_max_age_seconds
        self.hand_phone_distance_threshold = hand_phone_distance_threshold
        self.hand_confirm_frames = hand_confirm_frames
        self.hand_release_frames = hand_release_frames
        self.in_hand_motion_speed_threshold = in_hand_motion_speed_threshold
        self.in_hand_motion_confirm_frames = in_hand_motion_confirm_frames
        self.hand_detector = MediaPipeHandDetector(
            min_detection_confidence=hand_min_detection_confidence,
            min_tracking_confidence=hand_min_tracking_confidence,
            model_asset_path=hand_landmarker_model_path,
        ) if enable_hand_proximity else None

        self.event_logger = event_logger

        self.model = None
        self.device = "cpu"
        self.initialized = False

        self.tracker = None
        self.tracker_active = False
        self.next_yolo_time = 0.0
        self.next_hand_time = 0.0

        self.state = PhoneState.UNKNOWN
        self.prev_center: Optional[Tuple[float, float]] = None
        self.prev_time: Optional[float] = None
        self.smoothed_speed = 0.0
        self.stable_frames = 0
        self.moving_frames = 0
        self.missing_frames = 0
        self.hand_near_frames = 0
        self.hand_far_frames = 0
        self.hand_motion_frames = 0
        self.last_event_time = 0.0
        self.distraction_count = 0

        self.last_bbox: Optional[BBox] = None
        self.last_confidence: Optional[float] = None
        self.last_source: str = "none"
        self.last_hand_obs = HandObservation(0.0, [], [], None)

    def initialize(self) -> bool:
        if self.initialized:
            return True
        try:
            import torch
            from ultralytics import YOLO

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            logger.info("Loading %s on %s", self.model_path, self.device)
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.initialized = True

            if self.hand_detector is not None:
                self.hand_detector.initialize()

            return True
        except Exception as exc:
            logger.exception("Failed to initialize YOLO: %s", exc)
            return False

    def _resize_for_processing(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.process_width <= 0 or w <= self.process_width:
            return frame
        scale = self.process_width / float(w)
        new_h = int(round(h * scale))
        return cv2.resize(frame, (self.process_width, new_h), interpolation=cv2.INTER_AREA)

    def _detect_phone_yolo(self, frame: np.ndarray) -> Tuple[Optional[BBox], Optional[float]]:
        """Run YOLO on the processed frame and return the best phone bbox."""
        if not self.initialized and not self.initialize():
            return None, None

        assert self.model is not None
        results = self.model.predict(
            frame,
            imgsz=self.yolo_imgsz,
            conf=self.confidence,
            classes=[self.PHONE_CLASS_ID],
            verbose=False,
        )

        best_bbox = None
        best_conf = None
        best_score = -1.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf <= best_score:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                best_bbox = (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
                best_conf = conf
                best_score = conf

        return best_bbox, best_conf

    def _create_tracker(self):
        """
        Create an OpenCV tracker with fallbacks across OpenCV builds.
        e.g.: cv2.TrackerCSRT_create()
        """
        constructors = []

        if hasattr(cv2, "legacy"):
            constructors.extend([
                ("KCF", getattr(cv2.legacy, "TrackerKCF_create", None)),
                ("CSRT", getattr(cv2.legacy, "TrackerCSRT_create", None)),
                ("MOSSE", getattr(cv2.legacy, "TrackerMOSSE_create", None)),
                ("MIL", getattr(cv2.legacy, "TrackerMIL_create", None)),
            ])

        constructors.extend([
            ("KCF", getattr(cv2, "TrackerKCF_create", None)),
            ("CSRT", getattr(cv2, "TrackerCSRT_create", None)),
            ("MOSSE", getattr(cv2, "TrackerMOSSE_create", None)),
            ("MIL", getattr(cv2, "TrackerMIL_create", None)),
        ])

        ordered = [item for item in constructors if item[0] == self.tracker_type]
        ordered += [item for item in constructors if item[0] != self.tracker_type]

        for name, ctor in ordered:
            if ctor is None:
                continue
            try:
                logger.debug("Using OpenCV %s tracker", name)
                return ctor()
            except Exception:
                continue

        raise RuntimeError("No OpenCV tracker is available. Install opencv-contrib-python.")

    def _init_tracker(self, frame: np.ndarray, bbox: BBox) -> None:
        try:
            self.tracker = self._create_tracker()
            # Some OpenCV builds reject numpy floats here, so force plain ints.
            x, y, w, h = bbox
            tracker_bbox = (
                int(round(x)),
                int(round(y)),
                int(round(w)),
                int(round(h)),
            )
            self.tracker.init(frame, tracker_bbox)
            self.tracker_active = True
        except Exception as exc:
            logger.warning("Could not initialize tracker: %s", exc)
            self.tracker = None
            self.tracker_active = False

    def _update_tracker(self, frame: np.ndarray) -> Optional[BBox]:
        if not self.tracker_active or self.tracker is None:
            return None
        try:
            ok, bbox = self.tracker.update(frame)
            if not ok:
                self.tracker_active = False
                return None
            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1:
                self.tracker_active = False
                return None
            return (x, y, w, h)
        except Exception:
            self.tracker_active = False
            return None

    @staticmethod
    def _center(bbox: BBox) -> Tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _point_to_bbox_distance(point: Point, bbox: BBox) -> float:
        px, py = point
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return math.hypot(dx, dy)

    def _update_motion(self, bbox: Optional[BBox], frame_shape: Tuple[int, int, int], now: float) -> float:
        if bbox is None:
            self.smoothed_speed *= 0.9
            self.prev_center = None
            self.prev_time = now
            return self.smoothed_speed

        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))
        center = self._center(bbox)

        if self.prev_center is None or self.prev_time is None:
            instant_speed = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            pixel_dist = math.hypot(center[0] - self.prev_center[0], center[1] - self.prev_center[1])
            instant_speed = (pixel_dist / diag) / dt

        alpha = 0.35
        self.smoothed_speed = alpha * instant_speed + (1.0 - alpha) * self.smoothed_speed
        self.prev_center = center
        self.prev_time = now
        return self.smoothed_speed

    def _get_hand_observation(self, frame: np.ndarray, now: float) -> HandObservation:
        if self.hand_detector is None:
            return HandObservation(now, [], [], None)

        should_run = now >= self.next_hand_time
        last_is_fresh = now - self.last_hand_obs.timestamp <= self.hand_max_age_seconds

        if should_run or not last_is_fresh:
            self.next_hand_time = now + self.hand_detection_period
            self.last_hand_obs = self.hand_detector.detect(frame, now)

        if now - self.last_hand_obs.timestamp <= self.hand_max_age_seconds:
            return self.last_hand_obs
        return HandObservation(now, [], [], None)

    def _update_hand_proximity(
        self,
        frame: np.ndarray,
        bbox: Optional[BBox],
        now: float,
    ) -> Tuple[bool, Optional[float], HandObservation]:
        """Return whether the hand is confirmed near the phone.

        Important detail: if the phone bbox disappears right after a confirmed
        HAND_NEAR state, keep the near signal alive for a few frames. That lets
        the state machine treat disappearance-after-near as a likely pickup or
        occlusion by the hand, instead of immediately losing all context.
        """
        hand_obs = self._get_hand_observation(frame, now)

        if bbox is None:
            self.hand_far_frames += 1
            if self.hand_far_frames >= self.hand_release_frames:
                self.hand_near_frames = 0
            hand_near = self.hand_near_frames >= self.hand_confirm_frames
            return hand_near, None, hand_obs

        if not hand_obs.all_points:
            self.hand_far_frames += 1
            if self.hand_far_frames >= self.hand_release_frames:
                self.hand_near_frames = 0
            hand_near = self.hand_near_frames >= self.hand_confirm_frames
            return hand_near, None, hand_obs

        phone_diag = max(1.0, math.hypot(bbox[2], bbox[3]))

        # Primary signal: fingertips close to the phone.
        candidate_points = hand_obs.fingertip_points if hand_obs.fingertip_points else hand_obs.all_points
        min_tip_dist_px = min(self._point_to_bbox_distance(p, bbox) for p in candidate_points)
        min_tip_dist_norm = min_tip_dist_px / phone_diag

        # Backup signal: any hand landmark close to/inside the phone bbox. This helps
        # when fingertips are occluded by the phone, which is common during a grip.
        min_any_dist_px = min(self._point_to_bbox_distance(p, bbox) for p in hand_obs.all_points)
        min_any_dist_norm = min_any_dist_px / phone_diag

        min_dist_norm = min(min_tip_dist_norm, min_any_dist_norm)
        is_near_now = min_dist_norm <= self.hand_phone_distance_threshold

        if is_near_now:
            self.hand_near_frames += 1
            self.hand_far_frames = 0
        else:
            self.hand_far_frames += 1
            if self.hand_far_frames >= self.hand_release_frames:
                self.hand_near_frames = 0

        hand_near = self.hand_near_frames >= self.hand_confirm_frames
        return hand_near, min_dist_norm, hand_obs

    def _maybe_emit(self, event: str, now: float) -> Optional[str]:
        if now - self.last_event_time < self.cooldown_seconds:
            return None
        self.last_event_time = now
        if event in ("picked_up", "phone_in_hand"):
            self.distraction_count += 1
        return event

    def _update_state(
        self,
        bbox: Optional[BBox],
        speed: float,
        hand_near: bool,
        now: float,
    ) -> Optional[str]:
        """Update the phone state.

        The key distinction is:
        - HAND_NEAR: hand is close, but the phone is still stable;
        - IN_HAND: hand is close and the phone moves, or it disappears right
          after HAND_NEAR.
        """
        event = None

        if bbox is None:
            self.missing_frames += 1
            self.stable_frames = 0
            self.moving_frames = 0
            if not hand_near:
                self.hand_motion_frames = 0
            # Missing phone
            if self.missing_frames >= self.missing_confirm_frames:
                if self.state in (PhoneState.HAND_NEAR, PhoneState.IN_HAND):
                    if self.state != PhoneState.IN_HAND:
                        self.state = PhoneState.IN_HAND
                        event = self._maybe_emit("phone_in_hand", now)
                    return event

                if self.state in (PhoneState.RESTING, PhoneState.MOVING):
                    event = self._maybe_emit("picked_up", now)
                self.state = PhoneState.MISSING
            return event

        self.missing_frames = 0

        is_stable = speed < self.stable_speed_threshold
        is_moving = speed > self.motion_speed_threshold
        has_hand_motion = hand_near and speed > self.in_hand_motion_speed_threshold

        if is_stable:
            self.stable_frames += 1
        else:
            self.stable_frames = 0

        if is_moving:
            self.moving_frames += 1
        else:
            self.moving_frames = 0

        if has_hand_motion:
            self.hand_motion_frames += 1
        elif not hand_near:
            self.hand_motion_frames = 0
        else:
            # Hand is near but the phone is not moving enough to be called a pickup.
            self.hand_motion_frames = 0

        # Once the phone has been classified as IN_HAND, keep that state while
        # the hand is still near. This covers the important case where the user
        # picks up the phone, then holds it very still.
        if self.state == PhoneState.IN_HAND:
            if hand_near:
                return None

            # Do not immediately leave IN_HAND when MediaPipe flickers.
            if self.hand_far_frames < self.hand_release_frames:
                return None

            if self.stable_frames >= self.stable_confirm_frames:
                self.state = PhoneState.RESTING
                return "put_down"
            if self.moving_frames >= self.moving_confirm_frames:
                self.state = PhoneState.MOVING
                return None
            return None

        # Hand close + phone motion is the actual pickup / distraction signal.
        if hand_near and self.hand_motion_frames >= self.in_hand_motion_confirm_frames:
            self.state = PhoneState.IN_HAND
            return self._maybe_emit("phone_in_hand", now)

        # Hand close but phone still stable is only a pre-pickup / intent signal.
        if hand_near:
            if self.state != PhoneState.HAND_NEAR:
                self.state = PhoneState.HAND_NEAR
                event = "hand_near_phone"
            return event

        # Leaving HAND_NEAR without pickup: return to the ordinary motion logic.
        if self.state == PhoneState.HAND_NEAR:
            if self.hand_far_frames < self.hand_release_frames:
                return None
            if self.stable_frames >= self.stable_confirm_frames:
                self.state = PhoneState.RESTING
                return "hand_left_phone"
            if self.moving_frames >= self.moving_confirm_frames:
                self.state = PhoneState.MOVING
                return None
            return None

        if self.state in (PhoneState.UNKNOWN, PhoneState.MISSING):
            if self.stable_frames >= self.stable_confirm_frames:
                self.state = PhoneState.RESTING
                event = "phone_resting"
            elif self.moving_frames >= self.moving_confirm_frames:
                self.state = PhoneState.MOVING
                event = self._maybe_emit("picked_up", now)

        elif self.state == PhoneState.RESTING:
            if self.moving_frames >= self.moving_confirm_frames:
                self.state = PhoneState.MOVING
                event = self._maybe_emit("picked_up", now)

        elif self.state == PhoneState.MOVING:
            if self.stable_frames >= self.stable_confirm_frames:
                self.state = PhoneState.RESTING
                event = "put_down"

        return event

    def process_frame(self, frame: np.ndarray) -> Tuple[Observation, np.ndarray]:
        """Process one camera frame.

        Returns an Observation and the resized frame used internally. Bounding boxes
        and hand points are expressed in coordinates of this returned frame.
        """
        now = time.time()
        proc = self._resize_for_processing(frame)

        # 1) Cheap tracker update every frame.
        bbox = self._update_tracker(proc)
        source = "tracker" if bbox is not None else "none"
        confidence = self.last_confidence if bbox is not None else None

        # 2) Expensive YOLO correction only at low frequency, or when tracker is lost.
        should_run_yolo = now >= self.next_yolo_time or bbox is None
        if should_run_yolo:
            self.next_yolo_time = now + self.detection_period
            yolo_bbox, yolo_conf = self._detect_phone_yolo(proc)
            if yolo_bbox is not None:
                bbox = yolo_bbox
                confidence = yolo_conf
                source = "yolo"
                self._init_tracker(proc, yolo_bbox)

        self.last_bbox = bbox
        self.last_confidence = confidence
        self.last_source = source

        speed = self._update_motion(bbox, proc.shape, now)
        hand_near, min_hand_phone_distance, hand_obs = self._update_hand_proximity(proc, bbox, now)
        event = self._update_state(bbox, speed, hand_near, now)
        phone_in_hand = self.state == PhoneState.IN_HAND

        obs = Observation(
            timestamp=now,
            bbox=bbox,
            confidence=confidence,
            source=source,
            speed=speed,
            state=self.state,
            event=event,
            hand_near=hand_near,
            phone_in_hand=phone_in_hand,
            hand_bbox=hand_obs.bbox,
            fingertip_points=hand_obs.fingertip_points,
            min_hand_phone_distance=min_hand_phone_distance,
        )
        if self.event_logger is not None:
            self.event_logger.log(obs)
        return obs, proc

    def draw_debug(self, frame: np.ndarray, obs: Observation) -> np.ndarray:
        out = frame.copy()

        # Draw hand first so the phone bbox remains visible on top.
        if obs.hand_bbox is not None:
            hx, hy, hw, hh = map(int, obs.hand_bbox)
            hand_color = (255, 180, 0)
            cv2.rectangle(out, (hx, hy), (hx + hw, hy + hh), hand_color, 1)
            cv2.putText(out, "hand", (hx, max(20, hy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        for px, py in obs.fingertip_points:
            cv2.circle(out, (int(px), int(py)), 5, (255, 0, 255), -1)

        if obs.bbox is not None:
            x, y, w, h = map(int, obs.bbox)
            if obs.state == PhoneState.RESTING:
                color = (0, 255, 0)
            elif obs.state == PhoneState.HAND_NEAR:
                color = (0, 255, 255)
            elif obs.state == PhoneState.IN_HAND:
                color = (255, 0, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"phone {obs.source}"
            if obs.confidence is not None:
                label += f" {obs.confidence:.2f}"
            cv2.putText(out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        dist_text = "-" if obs.min_hand_phone_distance is None else f"{obs.min_hand_phone_distance:.2f}"
        status = (
            f"state={obs.state.value} speed={obs.speed:.3f} "
            f"near={obs.hand_near} in_hand={obs.phone_in_hand} dist={dist_text} "
            f"event={obs.event or '-'} count={self.distraction_count}"
        )
        cv2.putText(out, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(out, "q: quit | r: reset", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out

    def reset(self) -> None:
        self.tracker = None
        self.tracker_active = False
        self.next_yolo_time = 0.0
        self.next_hand_time = 0.0
        self.state = PhoneState.UNKNOWN
        self.prev_center = None
        self.prev_time = None
        self.smoothed_speed = 0.0
        self.stable_frames = 0
        self.moving_frames = 0
        self.missing_frames = 0
        self.hand_near_frames = 0
        self.hand_far_frames = 0
        self.hand_motion_frames = 0
        self.last_bbox = None
        self.last_confidence = None
        self.last_source = "none"
        self.last_hand_obs = HandObservation(0.0, [], [], None)


def run_webcam_demo(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    monitor = HybridPhoneMonitor(
        model_path=args.model,
        confidence=args.confidence,
        detection_hz=args.detection_hz,
        process_width=args.process_width,
        yolo_imgsz=args.yolo_imgsz,
        tracker_type=args.tracker,
        stable_speed_threshold=args.stable_speed_threshold,
        motion_speed_threshold=args.motion_speed_threshold,
        stable_confirm_frames=args.stable_confirm_frames,
        moving_confirm_frames=args.moving_confirm_frames,
        missing_confirm_frames=args.missing_confirm_frames,
        cooldown_seconds=args.cooldown,
        enable_hand_proximity=not args.no_hand_proximity,
        hand_detection_hz=args.hand_detection_hz,
        hand_max_age_seconds=args.hand_max_age_seconds,
        hand_phone_distance_threshold=args.hand_phone_distance_threshold,
        hand_confirm_frames=args.hand_confirm_frames,
        hand_release_frames=args.hand_release_frames,
        in_hand_motion_speed_threshold=args.in_hand_motion_speed_threshold,
        in_hand_motion_confirm_frames=args.in_hand_motion_confirm_frames,
        hand_min_detection_confidence=args.hand_min_detection_confidence,
        hand_min_tracking_confidence=args.hand_min_tracking_confidence,
        hand_landmarker_model_path=args.hand_landmarker_model,
        event_logger=JsonlEventLogger(args.log_path),
    )

    if not monitor.initialize():
        raise RuntimeError("Could not initialize YOLO")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    logger.info("Starting webcam loop. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.warning("Failed to read frame")
            break

        obs, proc = monitor.process_frame(frame)
        if obs.event is not None:
            logger.info(
                "EVENT: %s state=%s speed=%.3f near=%s in_hand=%s dist=%s",
                obs.event,
                obs.state.value,
                obs.speed,
                obs.hand_near,
                obs.phone_in_hand,
                None if obs.min_hand_phone_distance is None else round(obs.min_hand_phone_distance, 3),
            )

        debug = monitor.draw_debug(proc, obs)
        cv2.imshow("Focus Buddy - Phone + Hand Intent Monitor", debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            monitor.reset()
            logger.info("Monitor reset")

    cap.release()
    cv2.destroyAllWindows()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo_detection_hand_intent.py",
        description=(
            "Focus Buddy webcam demo: YOLO detects the phone, OpenCV tracks it, "
            "and MediaPipe Hands checks whether fingers are close to the phone."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    camera_group = parser.add_argument_group("Camera input")
    camera_group.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index used by OpenCV VideoCapture. Usually 0 for the default webcam.",
    )
    camera_group.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested capture width in pixels. The camera may ignore unsupported values.",
    )
    camera_group.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested capture height in pixels. The camera may ignore unsupported values.",
    )
    camera_group.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Requested camera frame rate. This is the acquisition FPS, not the YOLO FPS.",
    )

    yolo_group = parser.add_argument_group("YOLO phone detection")
    yolo_group.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="YOLO model path or model name. Use yolov8n.pt for speed, yolov8s.pt for better accuracy.",
    )
    yolo_group.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO confidence for accepting a cell-phone detection.",
    )
    yolo_group.add_argument(
        "--detection-hz",
        type=float,
        default=5.0,
        help="How many times per second YOLO is allowed to run. Lower is lighter for modest hardware.",
    )
    yolo_group.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Resize frames to this width before processing. Use 0 to keep the original size.",
    )
    yolo_group.add_argument(
        "--yolo-imgsz",
        type=int,
        default=320,
        help="Image size passed to YOLO inference. Smaller is faster, larger may detect small phones better.",
    )

    tracker_group = parser.add_argument_group("OpenCV tracking")
    tracker_group.add_argument(
        "--tracker",
        type=str,
        default="KCF",
        choices=["KCF", "CSRT", "MOSSE", "MIL"],
        help="OpenCV tracker used between YOLO detections. KCF is a good speed/accuracy default; CSRT is slower but often more precise.",
    )

    state_group = parser.add_argument_group("State machine and events")
    state_group.add_argument(
        "--stable-speed-threshold",
        type=float,
        default=0.035,
        help="Below this normalized speed, the phone is considered stable/resting.",
    )
    state_group.add_argument(
        "--motion-speed-threshold",
        type=float,
        default=0.16,
        help="Above this normalized speed, the phone is considered moving/picked up.",
    )
    state_group.add_argument(
        "--stable-confirm-frames",
        type=int,
        default=8,
        help="Number of consecutive stable frames required before switching to RESTING.",
    )
    state_group.add_argument(
        "--moving-confirm-frames",
        type=int,
        default=8,
        help="Number of consecutive moving frames required before switching to MOVING.",
    )
    state_group.add_argument(
        "--missing-confirm-frames",
        type=int,
        default=8,
        help="Number of consecutive frames without a phone bbox before switching to MISSING.",
    )
    state_group.add_argument(
        "--cooldown",
        type=float,
        default=8.0,
        help="Minimum seconds between two distraction events, to avoid repeated alerts for the same pickup.",
    )
    state_group.add_argument(
        "--log-path",
        type=str,
        default="focus_buddy_events.jsonl",
        help="Path of the JSONL file where detected events are appended.",
    )

    hand_group = parser.add_argument_group("MediaPipe hand/finger proximity")
    hand_group.add_argument(
        "--no-hand-proximity",
        action="store_true",
        help="Disable MediaPipe hand proximity and use only phone motion/missing logic.",
    )
    hand_group.add_argument(
        "--hand-landmarker-model",
        type=str,
        default="hand_landmarker.task",
        help="Path to the MediaPipe Tasks hand_landmarker.task model. Required when mp.solutions is not available.",
    )
    hand_group.add_argument(
        "--hand-detection-hz",
        type=float,
        default=10.0,
        help="How many times per second MediaPipe Hands is allowed to run.",
    )
    hand_group.add_argument(
        "--hand-max-age-seconds",
        type=float,
        default=0.25,
        help="How long a previous hand detection remains valid when MediaPipe is not run on the current frame.",
    )
    hand_group.add_argument(
        "--hand-phone-distance-threshold",
        type=float,
        default=0.12,
        help="Maximum normalized distance between hand landmarks/fingertips and the phone bbox to count as near. Lower is stricter.",
    )
    hand_group.add_argument(
        "--hand-confirm-frames",
        type=int,
        default=3,
        help="Consecutive near-hand frames required before switching to HAND_NEAR.",
    )
    hand_group.add_argument(
        "--hand-release-frames",
        type=int,
        default=4,
        help="Consecutive far/no-hand frames required before leaving HAND_NEAR or IN_HAND, useful against MediaPipe flicker.",
    )
    hand_group.add_argument(
        "--in-hand-motion-speed-threshold",
        type=float,
        default=0.06,
        help="Minimum normalized phone speed, while the hand is near, required to promote HAND_NEAR to IN_HAND. Lower is more sensitive.",
    )
    hand_group.add_argument(
        "--in-hand-motion-confirm-frames",
        type=int,
        default=2,
        help="Consecutive hand-near frames with phone motion required before switching to IN_HAND.",
    )
    hand_group.add_argument(
        "--hand-min-detection-confidence",
        type=float,
        default=0.45,
        help="Minimum confidence for initial MediaPipe hand detection.",
    )
    hand_group.add_argument(
        "--hand-min-tracking-confidence",
        type=float,
        default=0.45,
        help="Minimum confidence for MediaPipe hand landmark tracking after a hand is found.",
    )

    return parser

if __name__ == "__main__":
    run_webcam_demo(build_argparser().parse_args())
