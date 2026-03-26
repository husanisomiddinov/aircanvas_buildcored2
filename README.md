# AirCanvas (Day 2 - Easy)

Draw in the air with your hand using a webcam.

When your thumb and index fingertip are pinched together, the index fingertip acts like a stylus.  
When you separate your fingers, drawing stops.

## Features

- Live webcam feed
- Hand tracking with MediaPipe Hands
- Pinch-to-draw behavior
- Stop drawing when fingers separate
- Multiple colors (Blue, Green, Red, Yellow)
- Keep strokes on screen until cleared

## Tech Stack

- Python 3
- MediaPipe Hands
- OpenCV
- NumPy

## Setup

1. Create and activate a virtual environment (recommended):

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   python air_canvas.py
   ```

## Controls

- Pinch thumb + index finger: start drawing
- Release fingers: stop drawing
- Hover over top color circles to select color
- `1`, `2`, `3`, `4`: quick color switch
- `C`: clear canvas
- `Q`: quit

## Hardware Concept Connection

This mirrors a basic touchscreen input model:

- **Coordinate mapping**: hand landmark coordinates are mapped to screen pixels
- **Input sampling rate**: webcam frames act like periodic sensor sampling (similar to ADC reads)

In a resistive touchscreen + ADC system, you read X/Y values each sample cycle.  
Here, each frame gives X/Y fingertip position used to draw in real time.

## Common Fixes

- **Hand not detected**
  - Use better lighting (face a window or lamp)
  - Keep your hand fully visible in the frame

- **Drawing flickers**
  - Increase `self.pinch_threshold` in `air_canvas.py` (example: from `38` to `45`)

- **Canvas clears when moving**
  - Keep strokes stored in a list (`self.strokes`) instead of drawing only on the current frame

## Notes

- If your webcam does not open, close apps that may already be using the camera.
- For smoother drawing, keep your hand at a moderate distance from the camera.
