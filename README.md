# OpenDay — Exploring Computer Science Through Interactive Projects

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

A collection of interactive projects created for OpenDay — a university open-day
event — to showcase what can be built with computer science, programming, and
technology. The projects explore different areas of computing through hands-on,
visual experiences: computer vision, real-time tracking, sound, games, and
automation.

Visitors at the event did not watch these demos — they played them. They typed
on a keyboard floating in the air, played piano by touching the keys with their
fingertips, caught falling fruit with their face, and rallied a game of pong
with their hands.

## Philosophy

> **Computer science isn't just about writing code — it's about turning ideas
> into things that work.**

## The Projects

| Project | What you do | Run |
| --- | --- | --- |
| [Virtual Keyboard](#1-virtual-keyboard) | Type in the air with finger pinches | `python "Air_Keyboard/Virtual Keyboard.py"` |
| [Virtual Piano](#2-virtual-piano) | Play piano notes with your fingertips | `python Piano/piano.py` |
| [Catch The Apples](#3-catch-the-apples) | Catch falling fruit with your face | `python CTA/CTA.py` |
| [Air Pong](#4-air-pong) | Play pong against a friend with your hands | `python PingPong/PingPong.py` |

---

### 1. Virtual Keyboard

**Folder:** `Air_Keyboard/`

An on-screen keyboard drawn over the live camera feed. Move your index finger
over a key to highlight it, then pinch your index finger and thumb together to
"press" it.

- Full QWERTY layout, with **Space**, **delete** (backspace), **language
  toggle**, and **Enter** keys.
- Bilingual output: press **language** to switch between English and Persian
  (Persian letters are shown on the keys and rendered correctly in the typed
  text line).
- Pressing **Enter** appends the typed line to `Air_Keyboard/typed_text.txt`
  (created automatically next to the script).
- The fingertip position is smoothed and key presses are debounced, so one
  pinch types exactly one key — no accidental repeated letters.
- Only the first detected hand controls the keyboard, so a second hand or a
  bystander in frame cannot type by accident.

**How it works:** MediaPipe Hands tracks the index-finger tip and thumb tip in
each camera frame; the index tip selects the key under it and the
thumb-to-index distance (scaled to the frame size) triggers a press.

**Requirements:** webcam. No speakers needed.

```bash
python "Air_Keyboard/Virtual Keyboard.py"
```

Press `ESC` or `Q` to quit.

### 2. Virtual Piano

**Folder:** `Piano/`

Five piano keys (C1–G1) are drawn over the bottom of the camera window. Touch
a key zone with any fingertip and the note plays through your speakers.

- All ten fingertips of two hands can play — one note per fingertip.
- A note is triggered when a fingertip *enters* a key zone; it repeats only
  after the fingertip leaves and re-enters, so holding a finger down doesn't
  drone.
- Hand skeleton, colored fingertip markers, and note labels are drawn on the
  live video for instant feedback.

**How it works:** MediaPipe Hands tracks up to two hands; each of the five
fingertip landmarks (thumb, index, middle, ring, pinky) is mapped to the key
zone under its position, and pygame plays the matching WAV file from
`Piano/sounds/`.

**Requirements:** webcam and speakers or headphones.

```bash
python Piano/piano.py
```

Press `Q` or `ESC` to quit.

### 3. Catch The Apples

**Folder:** `CTA/`

A falling-fruit arcade game in a pygame window. Your face is the basket: move
your head left and right and the basket follows you. Catch fruit for points —
red = 1, orange = 2, yellow = 3 — but every fruit that hits the ground costs
the same points.

- New fruit falls every couple of seconds; the game gets busy fast.
- If no face is found for a couple of seconds, the basket drifts gently back
  to the center so the demo recovers on its own.
- The last five caught fruits are shown inside the basket as a trail.

**How it works:** OpenCV's Haar-cascade face detector finds the face in each
camera frame; its horizontal center is smoothed (exponential average) and used
as the basket position.

**Requirements:** webcam. No speakers needed.

```bash
python CTA/CTA.py
```

Close the window to quit.

### 4. Air Pong

**Folder:** `PingPong/`

Two-player pong where the paddles are controlled by your hands. Each player
holds up a hand and moves it up and down; the vertical position of the index
fingertip steers the paddle on their side of the screen.

- The leftmost hand controls the left paddle, the rightmost hand the right
  paddle. Playing alone? A single hand simply controls the paddle on its own
  side of the screen.
- First player to **5 points** wins; press `ENTER` to restart.
- The ball serves after a short countdown and gets faster with every point
  (up to a cap), so long rallies stay exciting.

**How it works:** MediaPipe Hands tracks up to two hands per frame; each
index-fingertip position is mapped (with smoothing) to a paddle. Collision
handling mirrors the ball's horizontal angle off paddles so it never tunnels
through them.

**Requirements:** webcam. No speakers needed.

```bash
python PingPong/PingPong.py
```

Close the window to quit.

---

## Technologies

- **Python** (verified on 3.12; MediaPipe supports 3.9–3.12)
- **OpenCV** — camera capture, image conversion, face detection (Haar cascade)
- **MediaPipe** — hand landmark tracking for the keyboard, piano, and pong
- **pygame** — game loops and rendering (pong, fruit catching) and audio
  playback (piano WAV files)
- **Pillow, arabic-reshaper, python-bidi** — correct Persian text shaping and
  rendering in the virtual keyboard
- **NumPy** — image buffer conversion behind the keyboard's text rendering

## Installation

You need **Python 3.9–3.12** (MediaPipe does not yet provide wheels for
Python 3.13+), a webcam, and pip.

```bash
git clone https://github.com/sinasaberri/OpenDay-Exploring-Computer-Science-Through-Interactive-Projects.git
cd OpenDay-Exploring-Computer-Science-Through-Interactive-Projects
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Hardware & Permissions

- **Webcam** — all four projects use the default camera (`index 0`). Make sure
  no other application is using it when you launch a demo.
- **Speakers or headphones** — only the Virtual Piano plays audio.
- **Camera permission** — on macOS, the first launch shows a system prompt to
  grant the terminal/Python access to the camera (System Settings → Privacy &
  Security → Camera). On Windows, camera access for desktop apps must be
  enabled under Settings → Privacy & security → Camera.

No microphone or physical keyboard/mouse is used by the demos themselves.

## Project Structure

```text
OpenDay/
├── Air_Keyboard/          # Virtual Keyboard (MediaPipe + OpenCV)
│   └── "Virtual Keyboard.py"
├── Piano/                 # Virtual Piano (MediaPipe + OpenCV + pygame)
│   ├── piano.py
│   └── sounds/            # one WAV file per note
├── CTA/                   # Catch The Apples (OpenCV face detection + pygame)
│   └── CTA.py
├── PingPong/              # Air Pong (MediaPipe + pygame)
│   └── PingPong.py
└── requirements.txt
```

## Known Limitations

- Detection quality depends on lighting and camera quality; MediaPipe runs on
  CPU, so very old machines may see a lower frame rate.
- The keyboard's pinch threshold is scaled to the frame height and tuned for
  720p input; other resolutions work but may need `PINCH_RATIO` adjustment.
- The Persian fallback font search looks at system fonts (Arial, Tahoma,
  Segoe UI); if none is available, Persian text falls back to PIL's default
  font, which cannot render it.
- Demos assume a single webcam at index 0; selecting a different camera is not
  exposed in the UI.

## Open Source

All four projects are publicly available so students, developers, and curious
visitors can explore the code, learn from the implementations, run the demos
themselves, and build their own ideas on top of them. Contributions and
remixes are welcome.

Released under the [MIT License](LICENSE).
