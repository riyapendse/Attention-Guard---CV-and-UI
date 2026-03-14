# 🖥️ Attention Guard — CV

A real-time computer vision tool that monitors your attention via webcam. If you look away from the screen (down at your phone, sideways, etc.), a voice alert fires. Available as both a standalone OpenCV window and a full web dashboard.

---

## How It Works

- Uses **MediaPipe FaceMesh** to track 468 facial landmarks + iris landmarks in real time.
- Combines **three signals** to detect inattention:
  - **Head pitch ratio** — nose position between forehead and chin
  - **Eye gaze (iris tracking)** — iris vertical/horizontal position inside the eye socket
  - **Head yaw** — nose offset from face midpoint (sideways detection)
- If inattention is detected for longer than the grace period, `pyttsx3` fires a repeating voice alert.
- Alert stops automatically when you look back at the screen.

---

## Project Structure

```
attention_guard_web/        ← Web dashboard version (recommended)
├── app.py                  ← Flask backend + CV logic
├── requirements.txt
└── templates/
    └── index.html          ← Browser UI

```

---

## Requirements

- Python 3.9 – 3.11 (MediaPipe has limited support for 3.12+)
- A working webcam
- Speakers / headphones (for the voice alert)
- NumPy < 2 (OpenCV 4.9 is incompatible with NumPy 2.x)

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Activate — Windows (Command Prompt)
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users**: if `pyttsx3` produces no sound, also run:
> ```bash
> pip install pypiwin32
> ```

> **Linux users**: install espeak for TTS:
> ```bash
> sudo apt-get install espeak
> ```

---

## Running — Web Dashboard (Recommended)

```bash
cd attention_guard_web
python app.py
```

Then open **http://localhost:5000** in your browser.

### Dashboard Features

| Feature | Description |
|---|---|
| **Start / Stop** | Guard only runs when explicitly started |
| **Live feed** | Camera stream with face bracket overlay in browser |
| **Sensitivity slider** | Adjusts detection threshold live (0.5 = very sensitive, 2.0 = lenient) |
| **Grace period slider** | 0.5s – 4s before alert fires |
| **Alert word** | Customise what gets spoken (default: `FAAAAHHH`) |
| **Session stats** | Session time, focused time, alerts fired, focus score % |
| **Alert log** | Timestamped history of every alert |
| **Signal debug** | Live pitch, gaze, yaw values |

---

## Running — Standalone Window

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause / Resume |

---

## Tuning (standalone `main.py`)

| Constant | Default | Effect |
|---|---|---|
| `GRACE_PERIOD` | `1.5` | Seconds looking away before alert fires |
| `YAW_THRESHOLD` | `0.26` | Sideways head turn sensitivity |
| `PITCH_THRESHOLD` | `0.62` | Downward head tilt sensitivity |
| `GAZE_DOWN_THRESHOLD` | `0.68` | Downward eye gaze sensitivity |
| `GAZE_SIDE_THRESHOLD` | `0.28` | Sideways eye gaze sensitivity |

**To make detection more sensitive:** lower the threshold values.
**To make it more lenient:** raise the threshold values.

---

## Camera Notes

- Uses `cv2.CAP_MSMF` backend on Windows for compatibility.
- Resolution is set to **1280×720** automatically (falls back if webcam doesn't support it).
- If camera fails to open, run this to find your camera index:
  ```bash
  python -c "import cv2; [print(f'Index {i}:', cv2.VideoCapture(i, cv2.CAP_MSMF).isOpened()) for i in range(5)]"
  ```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `mediapipe has no attribute 'solutions'` | `pip install mediapipe==0.10.9` |
| `numpy.core.multiarray failed to import` | `pip install "numpy<2"` |
| Camera won't open | Try `cv2.CAP_MSMF` backend; check Windows camera privacy settings |
| No voice alert | Install `pypiwin32` (Windows) or `espeak` (Linux) |
| PowerShell activation blocked | Switch terminal to Command Prompt in VS Code |
