"""Recognize signs from a live webcam using a trained CNN-LSTM model."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from sign_language.vision import create_hands, extract_landmarks, open_camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/sign_cnn_lstm.keras"))
    parser.add_argument("--labels", type=Path, default=Path("models/labels.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    if not args.model.is_file() or not args.labels.is_file():
        parser.error("Model files not found. Run sign-train before sign-predict.")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    model = keras.models.load_model(args.model)
    with args.labels.open(encoding="utf-8") as file:
        labels = json.load(file)
    if not labels or model.output_shape[-1] != len(labels):
        parser.error("The label file does not match the model output.")
    sequence_length = int(model.input_shape[1])
    sequence: deque[np.ndarray] = deque(maxlen=sequence_length)
    camera = open_camera(args.camera)
    hands = create_hands()
    prediction = "Show a sign"
    confidence = 0.0
    frames_since_prediction = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            frame = cv2.flip(frame, 1)
            sequence.append(extract_landmarks(frame, hands))
            frames_since_prediction += 1
            if len(sequence) == sequence_length and frames_since_prediction >= 5:
                probabilities = model.predict(
                    np.expand_dims(np.asarray(sequence, dtype=np.float32), axis=0),
                    verbose=0,
                )[0]
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])
                prediction = labels[index] if confidence >= args.threshold else "Unknown"
                frames_since_prediction = 0

            cv2.putText(
                frame,
                f"{prediction}  {confidence:.0%}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "0 or Q: quit",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Sign language recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("0")):
                break
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
