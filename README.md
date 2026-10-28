# Sign Language Recognition

A webcam-based sign recognition project using MediaPipe hand landmarks and a
CNN-LSTM deep neural network. A CNN extracts spatial hand-pose features from
each frame; LSTM layers learn how those poses change over a sequence. The
recognized label is displayed as text.

## Requirements

- Python 3.10 or 3.11
- A webcam
- Windows, macOS, or Linux

## Install

From the project directory on Windows:

```powershell
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
& $python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` for the commands below. On macOS/Linux, create the environment with
`python3 -m venv .venv`, activate with `source .venv/bin/activate`, then run
`python -m pip install -e .`.

## Collect ASL alphabet training data

The alphabet collector handles all 26 labels in one webcam session. Collect at
least 10 sequences per letter for an initial model; 30–40 or more per letter,
with varied hand positions, lighting, and performers, is a better starting
point:

```powershell
sign-collect --alphabet --samples 10
```

In the webcam window, press a letter **A–Z** to select the sign to collect.
Perform that ASL handshape in view and press **SPACE** to record one
30-frame sequence. Select another letter and repeat until the progress shows
all 26 letters complete; press **0** (or **Q**) to stop early. For **J** and **Z**, make
the movement during the capture. Samples are stored in
`data/sequences/<letter>/`. The collector remembers counts across runs, so
re-run the command to add samples. Keep sequence length consistent between
collection and training.

To collect a single sign instead, use `sign-collect --label hello
--samples 40`.

## Train

```powershell
sign-train --epochs 60
```

Training requires at least two label folders and at least five valid sequences
in each. It uses a stratified validation split, early stopping, and learning
rate reduction. The trained model, class names, and validation metrics are
written to `models/`.

## Recognize signs

```powershell
sign-predict
```

The webcam window displays the predicted sign and confidence. Press **0** or
**Q** to
quit. Use `--camera 1` if your webcam is not device 0, or
`--threshold 0.75` to require more confidence before displaying a label.

## Data format and notes

Each `.npy` file is one sequence with shape `(30, 42, 3)`: 30 frames, 21
landmarks per hand, and normalized x/y plus relative z coordinates. Missing
hands are zero-filled. Training stores only landmark coordinates, not webcam
images. This project recognizes the isolated signs represented by the folders
you collect; it is not a continuous sentence translator or a pretrained model.
Recognition quality depends on representative data and should be evaluated
with signers and conditions not used for training.
