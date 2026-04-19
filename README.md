# Hand Gesture to Cursor Controller

Control your computer mouse using hand gestures! This application uses MediaPipe for real-time hand and face detection via your webcam, allowing you to move your cursor, click, and scroll using intuitive hand gestures.

## Features

✋ **Hand Gesture Recognition**
- **Cursor Movement**: Move your mouse cursor by pointing with your index finger
- **Click**: Pinch your thumb and index finger together to perform a mouse click
- **Scroll**: While pinching, move your hand up or down to scroll
- **Gun Gesture**: Point with index and ring fingers up while other fingers are down for a visual "gun" effect

👁️ **Face Landmarks**
- Real-time detection and visualization of face landmarks
- Eye closure detection for potential future interactions

🎯 **Smooth Interaction**
- Exponential smoothing for cursor movement to reduce jitter
- Configurable gesture detection thresholds
- Multi-hand support

## Requirements

- Python 3.7+
- Webcam/Camera
- Windows, macOS, or Linux

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Handgesturetocursor.git
cd Handgesturetocursor
```

### 2. Create and activate a virtual environment (recommended)
```bash
# Windows
python -m venv haat
haat\Scripts\activate

# macOS/Linux
python3 -m venv haat
source haat/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### Required Packages
- **opencv-python** - For video capture and image processing
- **mediapipe** - For hand and face landmark detection
- **pyautogui** - For mouse control
- **pynput** - For keyboard input
- **numpy** - For numerical operations

## Usage

### Basic Usage
```bash
python main.py
```

The application will:
1. Open your webcam
2. Display real-time hand and face landmarks
3. Track your hand position and gestures
4. Control your mouse based on detected gestures

### Gesture Controls

| Gesture | Action |
|---------|--------|
| **Index finger pointing** | Move cursor to finger position |
| **Pinch (thumb + index)** | Click mouse button |
| **Pinch + move up/down** | Scroll up/down |
| **Index + ring up, other fingers down** | Gun gesture (visual effect) |

### Using MediaPipe Tasks API (Optional)

If you want to use MediaPipe's Tasks API instead of the Solutions API:

1. Download the hand landmarker model:
   - Visit [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
   - Download the `.task` file

2. Place it in the project root as `hand_landmarker.task`

3. Optionally download the face landmarker model and place it as `face_landmarker.task`

Alternatively, set environment variables:
```bash
# Windows
set HAND_LANDMARKER_MODEL=path/to/hand_landmarker.task
set FACE_LANDMARKER_MODEL=path/to/face_landmarker.task

# macOS/Linux
export HAND_LANDMARKER_MODEL=path/to/hand_landmarker.task
export FACE_LANDMARKER_MODEL=path/to/face_landmarker.task

python main.py
```

## Configuration

You can adjust the following parameters in `main.py` to customize the behavior:

```python
click_cooldown = 0.3              # Seconds between mouse clicks
scroll_cooldown = 0.1             # Seconds between scroll events
pinch_scroll_threshold = 0.05     # Distance threshold for pinch detection
scroll_smoothing = 0.7            # Smoothing factor for scroll (0-1)
mouse_smoothing = 0.6             # Smoothing factor for cursor movement (0-1)
```

### Hand Detection Parameters
- `max_num_hands`: Number of hands to detect (default: 2)
- `min_detection_confidence`: Confidence threshold for hand detection (default: 0.8)

### Face Detection Parameters
- `max_num_faces`: Number of faces to detect (default: 1)
- `min_detection_confidence`: Confidence threshold for face detection (default: 0.6)

## How It Works

### 1. Hand Detection
The application uses MediaPipe's hand landmarker to detect 21 landmarks on each hand, including:
- Palm
- Fingers (thumb, index, middle, ring, pinky)
- Knuckle positions

### 2. Gesture Recognition
Multiple hand gestures are recognized by analyzing the relative positions of finger landmarks:
- **Pinch**: Distance between thumb and index finger tips < threshold
- **Gun Gesture**: Index and ring fingers extended, other fingers folded
- **Finger Pointing**: Index finger up, other fingers down

### 3. Mouse Control
- **Movement**: The index finger tip position is mapped to screen coordinates
- **Smoothing**: Exponential smoothing reduces jitter in cursor movement
- **Click**: Pinch gesture triggers mouse click events
- **Scroll**: Vertical movement while pinching triggers scroll events

### 4. Face Detection
Face landmarks are detected and visualized in real-time. The system can detect:
- Eye positions and openness
- Mouth shape
- Face contours

## Troubleshooting

### Camera Not Opening
- Check if your camera is properly connected
- Verify camera permissions in your OS settings
- Try a different camera/USB port

### Poor Gesture Recognition
- Ensure adequate lighting
- Keep your hand within the camera frame
- Adjust the `min_detection_confidence` parameter to be more/less strict
- Make sure your gestures are clear and deliberate

### High CPU Usage
- Reduce the resolution of your webcam
- Lower the `max_num_hands` or `max_num_faces` values
- Disable face tracking by not providing a face_landmarker.task file

### Mouse Movement is Jerky
- Increase the `mouse_smoothing` value (closer to 1.0)
- Ensure your webcam has good frame rate (30+ FPS)

## File Structure

```
Handgesturetocursor/
├── main.py                      # Main application script
├── requirement.txt              # Python dependencies
├── README.md                    # This file
├── hand_landmarker.task         # Hand landmarker model (optional)
├── face_landmarker.task         # Face landmarker model (optional)
└── haat/                        # Virtual environment directory
```

## Performance Tips

1. **Lighting**: Ensure good lighting conditions for better hand detection
2. **Distance**: Position your hand 30-100cm from the camera for best results
3. **Background**: Use a plain background for optimal detection
4. **Smoothing**: Adjust smoothing factors based on your PC performance
5. **Resolution**: Reduce camera resolution if CPU usage is high

## Future Improvements

- [ ] Add more gesture types (fist, thumbs up, peace sign, etc.)
- [ ] Implement gesture recording and playback
- [ ] Add voice control integration
- [ ] Create a GUI settings panel
- [ ] Support for multiple monitor setups
- [ ] Gesture customization and binding to keyboard shortcuts
- [ ] Real-time gesture statistics and performance metrics

## Limitations

- Requires good lighting conditions
- Single-hand gestures may conflict when multiple hands are detected
- Gesture recognition accuracy depends on hand orientation and camera angle
- Works best with a stationary camera

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - For hand and face detection models
- [OpenCV](https://opencv.org/) - For computer vision processing
- [PyAutoGUI](https://pyautogui.readthedocs.io/) - For mouse control

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

---

**Enjoy controlling your computer with hand gestures!** 🖐️💻
