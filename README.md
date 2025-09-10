# VisionAid: A Multimodal Edge-Device System for Enhancing Environmental Awareness in Visually Impaired Individuals

 ## Features

- **Object Detection (YOLOv11n)** – detects common objects in live camera or still images.
- **Currency Detection** – collects *all* high-confidence bills per frame (with optional NMS).
- **Face Recognition** – register/list known faces (persistent JSON DB), dynamic scaling for any resolution.
- **Audio Feedback** – text-to-speech via `espeak`, priority queue, cooldowns, graceful shutdown/drain.
- **Vibration Feedback** – urgency patterns based on obstacle distance (`HC-SR04` style).
- **GPS + Email Alerts** – stubbed GPS reader, email alerts using `.env` credentials.
- **Cross-platform GPIO** – safe Raspberry Pi GPIO with simulation fallback.
- **Configurable** – all options via CLI flags *and/or* `.env` file.

---

## Project Structure
```
vision_aid/
├── __init__.py
├── audio.py # TTS & audio queue
├── camera.py # Camera abstraction
├── config.py # CLI parser + logging (loads .env)
├── detection.py # YOLO wrapper + currency NMS
├── faces.py # Face recognition + persistence
├── gpio_shim.py # Safe GPIO with simulator
├── gps_email.py # GPS stub + email sender
├── main.py # Entry point (live + image modes)
├── models/ # Pre-trained models (best.onnx, smodel [Vosk], yolo11n_ncnn_model)
├── motor.py # Vibration patterns
├── queues.py # Priority audio queue
├── settings.py # Central .env loader & helpers
├── storage.py # JSON storage for faces
├── threads.py # Stoppable thread manager
└── utils.py # Helpers (resize, throttles, bbox scale)
```

The `resize_with_ratio` helper now safely returns the original image when a
target dimension is zero or negative, preventing runtime errors from invalid
resize requests.
---

## Installation

### 1. Clone repo
```bash
git clone https://github.com/Rajvardhan-Desai/vision_aid.git
cd vision_aid
```
### 2. Setup virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Install system packages
```bash
sudo apt update
sudo apt install espeak-ng cmake libopenblas-dev libatlas-base-dev
```
### 5. Setup `.env`
- Create a `.env` in the project root (copy `.env.example`):
```env
# Email / Alerts
SENDER_EMAIL=you@example.com
SENDER_PASSWORD=app-password-here
EMERGENCY_CONTACT=contact@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Models
CURRENCY_MODEL=models/best.onnx
```
--- 

## Usage
### Single image inference
```bash
python -m vision_aid.main \
  --model runs/detect/train/weights/best.pt \
  --resolution 640x480 \
  --image sample.jpg \
  --thresh 0.5
```
### Live camera mode
```bash
python -m vision_aid.main \
  --model models/yolo11n_ncnn_model \
  --source picamera0 \
  --resolution 640x480 \
  --voice-commands \
  --vosk-model models/smodel \
  --audio-device 5 \
  --enable-gps \
  --currency-model models/best.onnx

```

-  Press `q` in the preview window to quit.
-  Use `--headless` for no preview (saves CPU).
-  Paths like `models/...` are relative to your current working directory;
   if you run the command from elsewhere, provide absolute paths, e.g.
   `--vosk-model /path/to/vision_aid/models/smodel`.

---

## Face Recognition

### List known faces
```bash
python -m vision_aid.main --model best.pt --resolution 640x480 --list-known-faces
```
### Register from image file
```bash
python -m vision_aid.main \
  --model best.pt --resolution 640x480 \
  --register-face "Alice:/path/to/alice.jpg"
```
### Register from camera
```bash
python -m vision_aid.main \
  --model best.pt --source usb0 --resolution 640x480 \
  --register-face Alice
```
- Faces are stored persistently at `data/faces.json`.

---

## Audio System
- Uses `espeak` (system binary) for text-to-speech.
- Priority queue ensures urgent alerts (e.g., obstacles) are spoken before routine detections.
- Cooldowns avoid spamming (3s default, 10s for familiar faces).
- On shutdown, audio thread drains queue before exit.

## Voice Commands
- Continuous microphone capture via `sounddevice` streamed into the `Vosk` speech recognizer.
- Wake word **"assistant"** activates the command listener.
- Supported commands:
  - `help` – list commands
  - `mute` / `speak` – toggle audio announcements
  - `all` / `less` – adjust object announcement verbosity
  - `faces` – report recognized people
  - `scan` – list recently detected objects
  - `distance` – report current range measurement
  - `save face` – begin face registration
  - `stop` – shut down the system (requires confirmation)
  - `currency` – enter currency detection mode
  - `emergency` – send an alert email (requires confirmation)
---

## Vibration Motor

Obstacle proximity → vibration urgency:

- `< 40 cm` → **urgent pulses**
- `40–70 cm` → **warning pulses**
- `70–110 cm` → **gentle pulses**
- `> 110 cm` → no vibration

--- 

## GPS + Email Alerts

- gps_email.get_gps_location() is stubbed — replace with your GPS module integration.
- If available, sends periodic email with current location using .env credentials.
- All email credentials are read securely from .env, never exposed on CLI.

---

## Testing

After installing the dependencies, you can run the unit tests with:

```bash
pytest
```


