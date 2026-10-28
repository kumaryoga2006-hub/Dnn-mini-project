"""Webcam and MediaPipe helpers shared by collection and prediction."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

LANDMARK_SHAPE = (42, 3)


def create_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_landmarks(frame: np.ndarray, hands) -> np.ndarray:
    """Return left-then-right hand landmarks, zero-filling hands not in frame."""
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = np.zeros(LANDMARK_SHAPE, dtype=np.float32)
    if not result.multi_hand_landmarks:
        return landmarks

    handedness = result.multi_handedness or []
    for index, hand in enumerate(result.multi_hand_landmarks):
        side = "Right"
        if index < len(handedness):
            side = handedness[index].classification[0].label
        offset = 21 if side == "Right" else 0
        landmarks[offset : offset + 21] = np.asarray(
            [[point.x, point.y, point.z] for point in hand.landmark],
            dtype=np.float32,
        )
    return landmarks


def open_camera(index: int):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {index}. Check that it is connected and not in use."
        )
    return camera

