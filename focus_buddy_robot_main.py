"""Focus Buddy + Reachy Mini integration, reaction version 3.

This version keeps the robot head completely still. Reactions use only:
- antenna poses;
- optional body/base yaw;
- the same synthesized sounds from v2.

Behavior mapping:
1. HAND_NEAR, but phone is not in hand:
   - subtle antenna focus toward the phone side;
   - optional small body/base turn toward the phone;
   - short high chirp.

2. IN_HAND / picked up:
   - confirmed distraction;
   - antennas slump and alternate;
   - optional small body/base "no" shake;
   - longer low descending groan.

Run the Reachy Mini simulator first, in another terminal:
    mjpython -m reachy_mini.daemon.app.main --sim

Then run this script from the same folder as yolo_detection_hand_intent.py:
    python focus_buddy_robot_v3_no_head.py --camera 0 --model yolo11s.pt

Debug without the robot:
    python focus_buddy_robot_v3_no_head.py --no-robot --camera 0 --model yolo11s.pt
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from yolo_detection_hand_intent import (
    HybridPhoneMonitor,
    JsonlEventLogger,
    Observation,
    PhoneState,
)

logger = logging.getLogger("focus_buddy_robot")


class ReactionLevel(Enum):
    """Semantic robot reaction level."""

    NONE = "none"
    SOFT = "soft_hand_near"
    HARD = "hard_phone_in_hand"


@dataclass(frozen=True)
class RobotReactionConfig:
    """Timing, amplitude, and audio parameters for robot reactions."""

    # The soft reaction is allowed to repeat more often because it is only a
    # warning posture, not a punishment.
    soft_repeat_seconds: float = 1.35

    # The hard reaction is more expressive, so it repeats less frequently.
    hard_repeat_seconds: float = 3.8

    # Neutral reset rate after the phone is put down or the session is paused.
    rest_repeat_seconds: float = 1.4

    # Shared motion scale. Antennas are expressed in radians.
    antenna_amplitude: float = 0.65
    soft_duration: float = 0.16
    hard_duration: float = 0.24
    rest_duration: float = 0.45

    # Body/base yaw is optional and expressed in degrees at the CLI level.
    # It is converted to radians before being sent to Reachy Mini.
    base_rotation_enabled: bool = True
    soft_base_yaw_degrees: float = 4.0
    hard_base_yaw_degrees: float = 13.0

    # Audio. The two sounds are intentionally very different:
    # soft = short high chirp, hard = low descending groan.
    audio_enabled: bool = True
    soft_volume: float = 0.11
    hard_volume: float = 0.30
    soft_chirp_seconds: float = 0.16
    hard_groan_seconds: float = 1.15


class TonePlayer:
    """Non-blocking audio player for short synthesized robot sounds.

    Audio is intentionally handled in a separate worker so the robot can move
    while the sound is playing. If a previous sound is still running, the new
    sound is skipped. This prevents overlapping audio buffers.
    """

    def __init__(self, mini: Any, enabled: bool) -> None:
        self.mini = mini
        self.enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def play_soft_chirp(self, seconds: float, volume: float) -> None:
        self._submit("soft_chirp", lambda: self._soft_chirp(seconds, volume))

    def play_hard_groan(self, seconds: float, volume: float) -> None:
        self._submit("hard_groan", lambda: self._hard_groan(seconds, volume))

    def _submit(self, name: str, fn: Callable[[], None]) -> None:
        if not self.enabled:
            return
        if self._future is not None and not self._future.done():
            return
        self._future = self._executor.submit(self._safe_run, name, fn)

    def _safe_run(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            logger.debug("Audio reaction '%s' failed: %s", name, exc)

    def _sample_rate(self) -> int:
        return int(self.mini.media.get_output_audio_samplerate())

    def _soft_chirp(self, seconds: float, volume: float) -> None:
        sample_rate = self._sample_rate()
        seconds = max(0.04, seconds)
        n = max(1, int(sample_rate * seconds))
        t = np.arange(n, dtype=np.float32) / sample_rate

        # Rising frequency: a small "I noticed that" chirp.
        freq = np.linspace(760.0, 1180.0, n, dtype=np.float32)
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
        envelope = np.sin(np.pi * t / seconds) ** 0.7
        wave = volume * envelope * np.sin(phase)

        self._push_wave(wave.astype(np.float32), sample_rate)

    def _hard_groan(self, seconds: float, volume: float) -> None:
        sample_rate = self._sample_rate()
        seconds = max(0.20, seconds)
        n = max(1, int(sample_rate * seconds))
        t = np.arange(n, dtype=np.float32) / sample_rate

        # Descending base frequency plus vibrato: more like a disappointed groan
        # than a clean beep.
        base = np.linspace(235.0, 90.0, n, dtype=np.float32)
        vibrato = 10.0 * np.sin(2.0 * np.pi * 6.0 * t)
        freq = base + vibrato
        phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate

        attack = np.minimum(1.0, t / 0.08)
        release = np.minimum(1.0, (seconds - t) / 0.22)
        envelope = np.clip(attack * release, 0.0, 1.0)
        tremolo = 0.68 + 0.32 * np.sin(2.0 * np.pi * 5.0 * t)

        # Add a weak lower harmonic to make it less like a pure sine tone.
        wave = volume * envelope * tremolo * (
            0.85 * np.sin(phase) + 0.25 * np.sin(0.5 * phase)
        )
        wave = np.clip(wave, -0.9, 0.9)

        self._push_wave(wave.astype(np.float32), sample_rate)

    def _push_wave(self, wave: np.ndarray, sample_rate: int) -> None:
        chunk_seconds = 0.02
        chunk_size = max(1, int(sample_rate * chunk_seconds))

        self.mini.media.start_playing()
        try:
            for start in range(0, len(wave), chunk_size):
                chunk = wave[start : start + chunk_size]
                self.mini.media.push_audio_sample(chunk)
                time.sleep(chunk_seconds * 0.75)
        finally:
            self.mini.media.stop_playing()


class ReachyMiniReactor:
    """Rate-limited robot behavior controller.

    The vision loop should remain responsive, so robot motions are executed in a
    single background worker. Audio is separate to allow motion and sound to run
    at the same time.
    """

    def __init__(self, mini: Any, config: RobotReactionConfig) -> None:
        self.mini = mini
        self.config = config

        self._motion_executor = ThreadPoolExecutor(max_workers=1)
        self._motion_future: Optional[Future] = None
        self._audio = TonePlayer(mini, enabled=config.audio_enabled)

        self._last_soft_time = 0.0
        self._last_hard_time = 0.0
        self._last_rest_time = 0.0
        self._last_level = ReactionLevel.NONE
        self._last_mode = "idle"

    def close(self) -> None:
        """Return to neutral and stop workers."""
        try:
            if self._is_motion_idle():
                self._try_submit_motion("rest", self._rest)
            if self._motion_future is not None:
                self._motion_future.result(timeout=3.0)
        except Exception as exc:
            logger.debug("Could not complete final rest motion: %s", exc)
        finally:
            self._audio.close()
            self._motion_executor.shutdown(wait=False, cancel_futures=True)

    def update(self, obs: Observation, session_active: bool, frame_width: int) -> None:
        """Map one vision observation to a robot reaction."""
        now = time.time()
        level = self._classify_observation(obs) if session_active else ReactionLevel.NONE
        side = self._phone_side(obs, frame_width)

        if level == ReactionLevel.HARD:
            is_new_hard = self._last_level != ReactionLevel.HARD
            should_repeat = now - self._last_hard_time >= self.config.hard_repeat_seconds
            if is_new_hard or should_repeat:
                submitted = self._try_submit_motion(
                    "hard_phone_in_hand",
                    lambda: self._hard_distracted(side),
                )
                if submitted:
                    self._last_hard_time = now
            self._last_level = level
            return

        if level == ReactionLevel.SOFT:
            is_new_soft = self._last_level != ReactionLevel.SOFT
            should_repeat = now - self._last_soft_time >= self.config.soft_repeat_seconds
            if is_new_soft or should_repeat:
                submitted = self._try_submit_motion(
                    "soft_hand_near",
                    lambda: self._soft_attention(side),
                )
                if submitted:
                    self._last_soft_time = now
            self._last_level = level
            return

        # No active warning. Return to neutral only on transitions or explicit
        # settling events; do not spam the robot with rest commands every frame.
        should_rest = (
            self._last_level != ReactionLevel.NONE
            or obs.event in {"put_down", "hand_left_phone", "phone_resting"}
            or not session_active
        )
        if should_rest and now - self._last_rest_time >= self.config.rest_repeat_seconds:
            submitted = self._try_submit_motion("rest", self._rest)
            if submitted:
                self._last_rest_time = now
        self._last_level = ReactionLevel.NONE

    @staticmethod
    def _classify_observation(obs: Observation) -> ReactionLevel:
        """Convert vision state into a robot reaction level.

        Hard always wins over soft. This is the important part that makes
        'hand close to phone' and 'phone in hand' behave differently.
        """
        hard = (
            obs.state == PhoneState.IN_HAND
            or obs.phone_in_hand
            or obs.event in {"phone_in_hand", "picked_up"}
        )
        if hard:
            return ReactionLevel.HARD

        soft = (
            obs.state == PhoneState.HAND_NEAR
            or obs.event == "hand_near_phone"
            or bool(obs.hand_near)
        )
        if soft:
            return ReactionLevel.SOFT

        return ReactionLevel.NONE

    @staticmethod
    def _phone_side(obs: Observation, frame_width: int) -> int:
        """Return -1 for left, +1 for right, 0 for center or unknown."""
        if obs.bbox is None or frame_width <= 0:
            return 0
        x, _, w, _ = obs.bbox
        cx = x + w / 2.0
        rel = (cx / frame_width) - 0.5
        if rel < -0.12:
            return -1
        if rel > 0.12:
            return 1
        return 0

    def _is_motion_idle(self) -> bool:
        return self._motion_future is None or self._motion_future.done()

    def _try_submit_motion(self, mode: str, fn: Callable[[], None]) -> bool:
        if not self._is_motion_idle():
            return False
        self._last_mode = mode
        self._motion_future = self._motion_executor.submit(self._safe_motion_run, mode, fn)
        return True

    def _safe_motion_run(self, mode: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            logger.warning("Robot reaction '%s' failed: %s", mode, exc)

    def _base_yaw(self, degrees: float) -> Optional[float]:
        """Return a body/base yaw target in radians, or None to keep yaw unchanged."""
        if not self.config.base_rotation_enabled:
            return None
        return float(np.deg2rad(degrees))

    def _goto(self, antennas: list[float], duration: float, body_yaw: Optional[float]) -> None:
        """Move antennas and optionally the body/base, without commanding the head.

        Newer Reachy Mini SDK versions support body_yaw in goto_target. The
        fallback keeps the code usable on older/simpler simulator builds that
        only accept antennas and duration. In both cases we deliberately do not
        pass a head target, so the head is not moved by this script.
        """
        try:
            self.mini.goto_target(
                antennas=antennas,
                body_yaw=body_yaw,
                duration=duration,
            )
        except TypeError:
            self.mini.goto_target(
                antennas=antennas,
                duration=duration,
            )

    def _rest(self) -> None:
        self._goto(
            antennas=[0.0, 0.0],
            body_yaw=self._base_yaw(0.0),
            duration=self.config.rest_duration,
        )

    def _soft_attention(self, side: int) -> None:
        """Soft reaction: curious warning, not a full alarm.

        Visual language:
        - no head movement;
        - one antenna points more strongly toward the phone side;
        - optional very small body/base turn toward the phone;
        - short chirp.
        """
        amp = self.config.antenna_amplitude
        d = self.config.soft_duration

        print("Reachy Mini: soft warning - hand near phone")
        self._audio.play_soft_chirp(
            seconds=self.config.soft_chirp_seconds,
            volume=self.config.soft_volume,
        )

        if side < 0:
            antenna_focus = [amp, 0.18 * amp]
            base_degrees = -self.config.soft_base_yaw_degrees
        elif side > 0:
            antenna_focus = [0.18 * amp, amp]
            base_degrees = self.config.soft_base_yaw_degrees
        else:
            antenna_focus = [0.72 * amp, 0.72 * amp]
            base_degrees = 0.0

        body_yaw = self._base_yaw(base_degrees)
        self._goto(antennas=antenna_focus, body_yaw=body_yaw, duration=d)
        self._goto(antennas=[0.42 * amp, 0.42 * amp], body_yaw=body_yaw, duration=d)
        self._goto(antennas=antenna_focus, body_yaw=body_yaw, duration=d)

    def _hard_distracted(self, side: int) -> None:
        """Hard reaction: confirmed phone use.

        Visual language:
        - no head movement;
        - antennas go down and alternate asymmetrically;
        - optional body/base yaw shake, like a small "no";
        - low descending groan.
        """
        amp = self.config.antenna_amplitude
        d = self.config.hard_duration
        yaw = self.config.hard_base_yaw_degrees
        direction = 1 if side >= 0 else -1

        print("Reachy Mini: HARD alert - phone in hand")
        self._audio.play_hard_groan(
            seconds=self.config.hard_groan_seconds,
            volume=self.config.hard_volume,
        )

        # This is intentionally much stronger than the soft warning.
        self._goto(
            antennas=[-amp, -amp],
            body_yaw=self._base_yaw(direction * yaw),
            duration=d,
        )
        self._goto(
            antennas=[-0.95 * amp, -0.30 * amp],
            body_yaw=self._base_yaw(-direction * yaw),
            duration=d,
        )
        self._goto(
            antennas=[-0.30 * amp, -0.95 * amp],
            body_yaw=self._base_yaw(direction * 0.75 * yaw),
            duration=d,
        )
        self._goto(
            antennas=[-0.55 * amp, -0.55 * amp],
            body_yaw=self._base_yaw(0.0),
            duration=self.config.rest_duration,
        )


class ConsoleReactor:
    """Fallback used with --no-robot for debugging the vision pipeline."""

    def __init__(self) -> None:
        self._last_message_time = 0.0
        self._last_level = ReactionLevel.NONE

    def update(self, obs: Observation, session_active: bool, frame_width: int) -> None:
        del frame_width
        now = time.time()
        if not session_active:
            return
        if now - self._last_message_time < 0.8:
            return

        level = ReachyMiniReactor._classify_observation(obs)
        if level == ReactionLevel.HARD:
            print("[robot disabled] HARD: phone in hand -> groan + antenna slump + base shake")
            self._last_message_time = now
        elif level == ReactionLevel.SOFT:
            print("[robot disabled] SOFT: hand near phone -> chirp + antenna focus")
            self._last_message_time = now
        self._last_level = level

    def close(self) -> None:
        pass


def build_timestamped_log_path(log_dir: str, explicit_log_path: Optional[str]) -> str:
    if explicit_log_path:
        path = Path(explicit_log_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(log_dir) / f"focus_buddy_events_{ts}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextlib.contextmanager
def robot_context(no_robot: bool):
    if no_robot:
        yield None
        return

    from reachy_mini import ReachyMini

    with ReachyMini() as mini:
        yield mini


def create_monitor(args: argparse.Namespace, log_path: str) -> HybridPhoneMonitor:
    return HybridPhoneMonitor(
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
        event_logger=JsonlEventLogger(log_path),
    )


def run_focus_buddy(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log_path = build_timestamped_log_path(args.log_dir, args.log_path)
    logger.info("Event log: %s", log_path)

    monitor = create_monitor(args, log_path)
    if not monitor.initialize():
        raise RuntimeError("Could not initialize HybridPhoneMonitor")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    session_active = True

    with robot_context(args.no_robot) as mini:
        if mini is None:
            reactor = ConsoleReactor()
            logger.info("Robot disabled. Running with console reactions only.")
        else:
            reactor = ReachyMiniReactor(
                mini,
                RobotReactionConfig(
                    soft_repeat_seconds=args.soft_repeat_seconds,
                    hard_repeat_seconds=args.hard_repeat_seconds,
                    rest_repeat_seconds=args.rest_repeat_seconds,
                    antenna_amplitude=args.antenna_amplitude,
                    base_rotation_enabled=not args.no_base_rotation,
                    soft_base_yaw_degrees=args.soft_base_yaw_degrees,
                    hard_base_yaw_degrees=args.hard_base_yaw_degrees,
                    audio_enabled=not args.no_audio,
                    soft_volume=args.soft_volume,
                    hard_volume=args.hard_volume,
                    soft_chirp_seconds=args.soft_chirp_seconds,
                    hard_groan_seconds=args.hard_groan_seconds,
                ),
            )
            logger.info("Connected to Reachy Mini.")

        try:
            logger.info("Focus session active. Keys: q quit | r reset | space pause/resume")
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Failed to read frame")
                    break

                obs, proc = monitor.process_frame(frame)

                if obs.event is not None:
                    logger.info(
                        "EVENT: %s state=%s speed=%.3f near=%s in_hand=%s",
                        obs.event,
                        obs.state.value,
                        obs.speed,
                        obs.hand_near,
                        obs.phone_in_hand,
                    )

                reactor.update(obs, session_active=session_active, frame_width=proc.shape[1])

                debug = monitor.draw_debug(proc, obs)
                session_text = "SESSION: ACTIVE" if session_active else "SESSION: PAUSED"
                cv2.putText(
                    debug,
                    session_text,
                    (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("Focus Buddy - Reachy Mini", debug)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    monitor.reset()
                    logger.info("Monitor reset")
                if key == ord(" "):
                    session_active = not session_active
                    logger.info("Session active: %s", session_active)

        finally:
            reactor.close()
            cap.release()
            cv2.destroyAllWindows()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="focus_buddy_robot_v3_no_head.py",
        description="Focus Buddy integration: HybridPhoneMonitor + Reachy Mini reactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    camera = parser.add_argument_group("Camera")
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--camera-width", type=int, default=1280)
    camera.add_argument("--camera-height", type=int, default=720)
    camera.add_argument("--camera-fps", type=int, default=30)

    yolo = parser.add_argument_group("YOLO / tracker")
    yolo.add_argument("--model", type=str, default="yolo11s.pt")
    yolo.add_argument("--confidence", type=float, default=0.35)
    yolo.add_argument("--detection-hz", type=float, default=5.0)
    yolo.add_argument("--process-width", type=int, default=640)
    yolo.add_argument("--yolo-imgsz", type=int, default=320)
    yolo.add_argument("--tracker", type=str, default="KCF", choices=["KCF", "CSRT", "MOSSE", "MIL"])

    state = parser.add_argument_group("State machine")
    state.add_argument("--stable-speed-threshold", type=float, default=0.035)
    state.add_argument("--motion-speed-threshold", type=float, default=0.16)
    state.add_argument("--stable-confirm-frames", type=int, default=8)
    state.add_argument("--moving-confirm-frames", type=int, default=8)
    state.add_argument("--missing-confirm-frames", type=int, default=8)
    state.add_argument("--cooldown", type=float, default=8.0)

    hand = parser.add_argument_group("Hand proximity")
    hand.add_argument("--no-hand-proximity", action="store_true")
    hand.add_argument("--hand-landmarker-model", type=str, default="hand_landmarker.task")
    hand.add_argument("--hand-detection-hz", type=float, default=10.0)
    hand.add_argument("--hand-max-age-seconds", type=float, default=0.25)
    hand.add_argument("--hand-phone-distance-threshold", type=float, default=0.12)
    hand.add_argument("--hand-confirm-frames", type=int, default=3)
    hand.add_argument("--hand-release-frames", type=int, default=7)
    hand.add_argument("--in-hand-motion-speed-threshold", type=float, default=0.06)
    hand.add_argument("--in-hand-motion-confirm-frames", type=int, default=2)
    hand.add_argument("--hand-min-detection-confidence", type=float, default=0.45)
    hand.add_argument("--hand-min-tracking-confidence", type=float, default=0.45)

    robot = parser.add_argument_group("Reachy Mini reactions")
    robot.add_argument("--no-robot", action="store_true", help="Run vision only and print robot reactions.")
    robot.add_argument("--soft-repeat-seconds", type=float, default=1.35)
    robot.add_argument("--hard-repeat-seconds", type=float, default=3.8)
    robot.add_argument("--rest-repeat-seconds", type=float, default=1.4)
    robot.add_argument("--antenna-amplitude", type=float, default=0.65)
    robot.add_argument("--no-base-rotation", action="store_true", help="Disable body/base yaw; use antennas and audio only.")
    robot.add_argument("--soft-base-yaw-degrees", type=float, default=4.0)
    robot.add_argument("--hard-base-yaw-degrees", type=float, default=13.0)
    robot.add_argument("--no-audio", action="store_true", help="Disable Reachy Mini audio reactions.")
    robot.add_argument("--soft-volume", type=float, default=0.11)
    robot.add_argument("--hard-volume", type=float, default=0.30)
    robot.add_argument("--soft-chirp-seconds", type=float, default=0.16)
    robot.add_argument("--hard-groan-seconds", type=float, default=1.15)

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument("--log-dir", type=str, default="logs")
    logging_group.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Explicit JSONL path. If omitted, a timestamped filename is created in --log-dir.",
    )

    return parser


if __name__ == "__main__":
    run_focus_buddy(build_argparser().parse_args())
