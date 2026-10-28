"""Capture labeled sequences of hand landmarks from a webcam."""

from __future__ import annotations

import argparse
import re
import time
import string
from pathlib import Path

import cv2
import numpy as np

from sign_language.vision import create_hands, extract_landmarks, open_camera


def collect_alphabet(
    samples_per_label: int,
    sequence_length: int,
    camera_index: int,
    output: Path,
) -> None:
    counts = {
        letter: len(list((output / letter).glob("*.npy")))
        if (output / letter).is_dir()
        else 0
        for letter in string.ascii_uppercase
    }
    camera = open_camera(camera_index)
    hands = create_hands()
    selected_label: str | None = None
    recording = False
    sequence: list[np.ndarray] = []
    try:
        while any(count < samples_per_label for count in counts.values()):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            frame = cv2.flip(frame, 1)
            landmarks = extract_landmarks(frame, hands)

            if recording:
                sequence.append(landmarks)
                if len(sequence) >= sequence_length:
                    label_dir = output / selected_label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    np.save(
                        label_dir / f"{time.time_ns()}.npy",
                        np.stack(sequence).astype(np.float32),
                    )
                    counts[selected_label] += 1
                    recording = False
                    sequence.clear()

            if recording:
                instruction = "Recording sign..."
            elif selected_label is None:
                instruction = "Press A-Z to select a letter | 0: quit"
            elif counts[selected_label] >= samples_per_label:
                instruction = "Letter complete. Select another A-Z | 0: quit"
            else:
                instruction = f"{selected_label}: SPACE captures sample | A-Z: change | 0: quit"

            completed = sum(count >= samples_per_label for count in counts.values())
            cv2.putText(
                frame,
                f"ASL alphabet: {completed}/26 letters | Selected: {selected_label or '-'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                instruction,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            if selected_label is not None:
                cv2.putText(
                    frame,
                    f"Samples for {selected_label}: {counts[selected_label]}/{samples_per_label}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("ASL alphabet data collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("0")):
                break
            if not recording and ord("a") <= key <= ord("z"):
                candidate = chr(key).upper()
                if counts[candidate] < samples_per_label:
                    selected_label = candidate
            elif key == ord(" ") and not recording and selected_label is not None:
                recording = True
                sequence.clear()
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()

    incomplete = [
        f"{letter} ({count}/{samples_per_label})"
        for letter, count in counts.items()
        if count < samples_per_label
    ]
    if incomplete:
        print(f"Collection stopped early. Incomplete letters: {', '.join(incomplete)}")
    else:
        print(f"Captured {samples_per_label} sample(s) for each ASL letter in {output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="Sign label, e.g. hello")
    parser.add_argument(
        "--alphabet",
        action="store_true",
        help="Collect A-Z using keyboard letter selection",
    )
    parser.add_argument("--samples", type=int, default=30, help="Sequences to capture")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/sequences"))
    args = parser.parse_args()

    if args.samples < 1 or args.sequence_length < 2:
        parser.error("--samples must be positive and --sequence-length must be at least 2")

    if args.alphabet:
        if args.label:
            parser.error("--alphabet cannot be combined with --label")
        collect_alphabet(
            args.samples,
            args.sequence_length,
            args.camera,
            args.output,
        )
        return
    if not args.label:
        parser.error("provide --label for one sign or use --alphabet for A-Z")
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", args.label.strip()).strip("_")
    if not label:
        parser.error("--label must contain at least one letter or number")

    output_dir = args.output / label
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = open_camera(args.camera)
    hands = create_hands()
    captured = 0
    recording = False
    sequence: list[np.ndarray] = []
    try:
        while captured < args.samples:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            frame = cv2.flip(frame, 1)
            landmarks = extract_landmarks(frame, hands)

            if recording:
                sequence.append(landmarks)
                if len(sequence) >= args.sequence_length:
                    filename = output_dir / f"{time.time_ns()}.npy"
                    np.save(filename, np.stack(sequence).astype(np.float32))
                    captured += 1
                    recording = False
                    sequence.clear()

            instruction = (
                "Recording sign..."
                if recording
                else "SPACE: capture next sample | 0 or Q: quit"
            )
            cv2.putText(
                frame,
                f"Sign: {label}  Samples: {captured}/{args.samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                instruction,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Sign language data collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("0")):
                break
            if key == ord(" ") and not recording:
                recording = True
                sequence.clear()
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()

    print(f"Captured {captured} sequence(s) for '{label}' in {output_dir}.")
    if captured < args.samples:
        print("Collection stopped early; run the command again to capture more.")


if __name__ == "__main__":
    main()
