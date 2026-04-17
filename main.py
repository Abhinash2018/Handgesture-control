import cv2 as cv
import mediapipe as mp
import math
import numpy as np
import pyautogui


use_solutions = hasattr(mp, "solutions")
if use_solutions:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
    # Face mesh (solutions API)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6)
else:
    # mediapipe Tasks API is installed (no mp.solutions). To use the
    # hand-landmarker task you must provide a model file (Hand Landmarker
    # TFLite/task bundle). Place it at `hand_landmarker.task` in the project
    # root or set the HAND_LANDMARKER_MODEL environment variable.
    import os
    model_path = os.environ.get("HAND_LANDMARKER_MODEL", "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("mediapipe 'solutions' API not available.")
        print("To run with the installed MediaPipe Tasks API, download a hand landmarker model and save it as 'hand_landmarker.task' in the project root, or set HAND_LANDMARKER_MODEL to its path.")
        print("See: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker for model details.")
        exit(1)
    # Lazy import of Tasks API classes
    from mediapipe.tasks.python.vision import hand_landmarker as hl
    from mediapipe.tasks.python.core import base_options
    BaseOptions = base_options.BaseOptions
    HandLandmarker = hl.HandLandmarker
    HandLandmarkerOptions = hl.HandLandmarkerOptions
    # Create hand landmarker from provided model
    options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_hands=2)
    task_landmarker = HandLandmarker.create_from_options(options)
    # Note: later in the loop we'll use `task_landmarker.detect` to get landmarks
    # Optionally create a face landmarker if a task model is provided
    face_model_path = os.environ.get("FACE_LANDMARKER_MODEL", "face_landmarker.task")
    face_task_landmarker = None
    if os.path.exists(face_model_path):
        from mediapipe.tasks.python.vision import face_landmarker as fl
        FaceLandmarker = fl.FaceLandmarker
        FaceLandmarkerOptions = fl.FaceLandmarkerOptions
        face_options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=face_model_path), num_faces=1)
        try:
            face_task_landmarker = FaceLandmarker.create_from_options(face_options)
        except Exception:
            face_task_landmarker = None


def is_finger_up(hand_landmarks, finger_tip_idx, finger_pip_idx):
    """Check if a finger is pointing up"""
    return hand_landmarks[finger_tip_idx].y < hand_landmarks[finger_pip_idx].y


def is_gun_gesture(hand_landmarks):
    """
    Detect gun gesture: index and ring fingers up, other fingers down
    Finger indices: 
    - Thumb: 4 (tip), 3 (PIP)
    - Index: 8 (tip), 6 (PIP)
    - Middle: 12 (tip), 10 (PIP)
    - Ring: 16 (tip), 14 (PIP)
    - Pinky: 20 (tip), 18 (PIP)
    """
    lm = hand_landmarks.landmark
    # Use index + ring extended as the "gun pointing" gesture.
    index_up = is_finger_up(lm, 8, 6)
    ring_up = is_finger_up(lm, 16, 14)
    middle_down = not is_finger_up(lm, 12, 10)
    pinky_down = not is_finger_up(lm, 20, 18)

    return index_up and ring_up and middle_down and pinky_down


def is_index_only_pointing(hand_landmarks):
    """
    Detect index finger only pointing: index up, middle/ring/pinky down
    Used as drawing trigger
    """
    lm = hand_landmarks.landmark
    index_up = is_finger_up(lm, 8, 6)
    middle_down = not is_finger_up(lm, 12, 10)
    ring_down = not is_finger_up(lm, 16, 14)
    pinky_down = not is_finger_up(lm, 20, 18)
    
    return index_up and middle_down and ring_down and pinky_down


def is_all_fingers_open(hand_landmarks):
    """
    Detect open hand: all fingers extended (thumb, index, middle, ring, pinky all up)
    Used as eraser trigger
    """
    lm = hand_landmarks.landmark
    thumb_up = is_finger_up(lm, 4, 3)
    index_up = is_finger_up(lm, 8, 6)
    middle_up = is_finger_up(lm, 12, 10)
    ring_up = is_finger_up(lm, 16, 14)
    pinky_up = is_finger_up(lm, 20, 18)
    
    return thumb_up and index_up and middle_up and ring_up and pinky_up


def is_pinch_gesture(hand_landmarks, threshold=0.05):
    """
    Detect pinch: thumb and index finger tips are close together
    Used for mouse click and scrolling
    Returns distance between thumb and index tips
    """
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    index_tip = lm[8]
    
    # Calculate distance between thumb and index tips
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    
    return distance < threshold, distance


def draw_gun(frame, hand_landmarks, handedness):
    """Draw a gun at the hand position"""
    lm = hand_landmarks.landmark
    
    # Get index finger tip position (barrel of gun)
    index_tip = lm[8]
    x = int(index_tip.x * frame.shape[1])
    y = int(index_tip.y * frame.shape[0])
    
    # Get wrist position (handle of gun)
    wrist = lm[0]
    wrist_x = int(wrist.x * frame.shape[1])
    wrist_y = int(wrist.y * frame.shape[0])
    
    # Draw a line from wrist to index as a guide (thin)
    cv.line(frame, (wrist_x, wrist_y), (x, y), (150, 150, 150), 2)

    # Compute muzzle direction (from wrist -> index) and extend for flash
    dx = x - wrist_x
    dy = y - wrist_y
    length = math.hypot(dx, dy)
    if length == 0:
        ux, uy = 0, -1
    else:
        ux, uy = dx / length, dy / length

    muzzle_dist = 35
    muzzle_x = int(x + ux * muzzle_dist)
    muzzle_y = int(y + uy * muzzle_dist)

    # Muzzle flash: bright circle + small radial lines
    cv.circle(frame, (muzzle_x, muzzle_y), 12, (0, 200, 255), -1)
    cv.circle(frame, (muzzle_x, muzzle_y), 6, (0, 255, 255), -1)
    for a in range(0, 360, 45):
        ra = math.radians(a)
        sx = int(muzzle_x + math.cos(ra) * 18)
        sy = int(muzzle_y + math.sin(ra) * 18)
        cv.line(frame, (muzzle_x, muzzle_y), (sx, sy), (0, 180, 255), 2)

    # Short bullet trail from index tip outward
    cv.line(frame, (x, y), (muzzle_x, muzzle_y), (0, 220, 220), 4)

    # Crosshair at index tip
    ch_size = 8
    cv.line(frame, (x - ch_size, y), (x + ch_size, y), (0, 255, 0), 2)
    cv.line(frame, (x, y - ch_size), (x, y + ch_size), (0, 255, 0), 2)

    # Add label
    cv.putText(frame, "POINT", (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def is_mouth_o_shape(face_landmarks_list):
    """
    Detect if one eye is closed.
    face_landmarks_list: list of NormalizedLandmark or similar with x, y attributes
    Returns True if either left or right eye is closed
    """
    if not face_landmarks_list or len(face_landmarks_list) < 386:
        return False
    
    try:
        # Eye landmark indices (MediaPipe FaceMesh):
        # Left eye: top=159, bottom=145
        # Right eye: top=386, bottom=374
        
        left_eye_top = face_landmarks_list[159]
        left_eye_bottom = face_landmarks_list[145]
        right_eye_top = face_landmarks_list[386]
        right_eye_bottom = face_landmarks_list[374]
        
        # Calculate vertical eye openness (distance between top and bottom)
        left_eye_openness = abs(left_eye_bottom.y - left_eye_top.y)
        right_eye_openness = abs(right_eye_bottom.y - right_eye_top.y)
        
        # Threshold for eye closure
        eye_closure_threshold = 0.015  # if openness < this, eye is considered closed
        
        left_eye_closed = left_eye_openness < eye_closure_threshold
        right_eye_closed = right_eye_openness < eye_closure_threshold
        
        # Return True if one (and only one) eye is closed
        return (left_eye_closed and not right_eye_closed) or (right_eye_closed and not left_eye_closed)
    except (IndexError, AttributeError):
        return False


def get_hand_index_tip_pos(hand_landmarks, frame_shape):
    """Get index finger tip position from hand landmarks, scaled to frame"""
    if not hand_landmarks or not hasattr(hand_landmarks, 'landmark'):
        return None
    lm = hand_landmarks.landmark
    if len(lm) < 9:
        return None
    index_tip = lm[8]
    x = int(index_tip.x * frame_shape[1])
    y = int(index_tip.y * frame_shape[0])
    return (x, y)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("cam not opended")
    exit()

# Drawing trail for mouth "O" feature (disabled)
# drawing_trail = []

# Mouse control variables
prev_index_y = None
mouse_state = {'last_click_time': 0, 'smoothed_x': None, 'smoothed_y': None}
click_cooldown = 0.3  # seconds between clicks
scroll_cooldown = 0.1  # seconds between scroll events
pinch_scroll_threshold = 0.05  # pinch distance threshold for scroll
scroll_smoothing = 0.7  # exponential smoothing factor (0-1, higher = more smoothing)
mouse_smoothing = 0.6  # exponential smoothing factor for mouse position

# Get screen size for mouse mapping
screen_width, screen_height = pyautogui.size()

def smooth_mouse_position(new_x, new_y, smoothed_x, smoothed_y):
    """Apply exponential smoothing to mouse coordinates"""
    if smoothed_x is None:
        return new_x, new_y
    smoothed_x = smoothed_x * mouse_smoothing + new_x * (1 - mouse_smoothing)
    smoothed_y = smoothed_y * mouse_smoothing + new_y * (1 - mouse_smoothing)
    return int(smoothed_x), int(smoothed_y)
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("can't recieve frame")
        break
    frame = cv.flip(frame, 1)
    rgb =  cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if use_solutions:
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for gun gesture
                if is_gun_gesture(hand_landmarks):
                    draw_gun(frame, hand_landmarks, result.multi_handedness[hand_idx])
                
                # Mouse control: move cursor to index finger position
                lm = hand_landmarks.landmark
                index_tip = lm[8]
                mouse_x = int(index_tip.x * screen_width)
                mouse_y = int(index_tip.y * screen_height)
                
                # Apply smoothing to mouse position
                smoothed_x, smoothed_y = smooth_mouse_position(
                    mouse_x, mouse_y, 
                    mouse_state['smoothed_x'], 
                    mouse_state['smoothed_y']
                )
                mouse_state['smoothed_x'] = smoothed_x
                mouse_state['smoothed_y'] = smoothed_y
                
                # Move mouse to smoothed index finger position
                try:
                    pyautogui.moveTo(smoothed_x, smoothed_y)
                except Exception:
                    pass
                
                # Detect pinch gesture
                is_pinching, pinch_distance = is_pinch_gesture(hand_landmarks)
                
                if is_pinching:
                    import time
                    current_time = time.time()
                    
                    # Detect click on pinch
                    if current_time - mouse_state['last_click_time'] > click_cooldown:
                        try:
                            pyautogui.click(button='left')
                            mouse_state['last_click_time'] = current_time
                        except Exception:
                            pass
                    
                    # Detect vertical movement for scroll while pinching
                    current_index_y = index_tip.y
                    if prev_index_y is not None:
                        y_diff = (prev_index_y - current_index_y) * screen_height
                        if abs(y_diff) > 15:  # threshold to avoid noise
                            try:
                                # Positive y_diff = finger moved up = scroll up
                                scroll_amount = int(y_diff / 25)
                                if abs(scroll_amount) > 0:
                                    pyautogui.scroll(scroll_amount)
                            except Exception:
                                pass
                    prev_index_y = current_index_y
                else:
                    # Reset scroll tracking when not pinching
                    prev_index_y = None
        # Face tracking (solutions API)
        try:
            face_results = face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                for f_landmarks in face_results.multi_face_landmarks:
                    # Draw face landmark dots
                    H, W = frame.shape[:2]
                    pts = np.array([[int(p.x * W), int(p.y * H)] for p in f_landmarks.landmark], dtype=np.int32)
                    if pts.size:
                        # draw small landmark dots (like hand tracker)
                        for p in pts:
                            cv.circle(frame, (int(p[0]), int(p[1])), 2, (255, 200, 0), -1)
        except Exception:
            pass
    else:
        # Tasks API path (use file-backed Image for compatibility)
        from mediapipe.tasks.python.vision.core import image as mp_image
        # Write a temporary file (slower) and load via Tasks Image.create_from_file
        import tempfile
        tmpf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp_name = tmpf.name
        tmpf.close()
        cv.imwrite(tmp_name, cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
        mp_img = mp_image.Image.create_from_file(tmp_name)
        task_result = task_landmarker.detect(mp_img)

        # task_result.hand_landmarks is List[List[NormalizedLandmark]]
        if task_result and task_result.hand_landmarks:
            for hand_idx, lm_list in enumerate(task_result.hand_landmarks):
                # Create simple wrapper with .landmark attribute to reuse existing functions
                class SimpleHand:
                    def __init__(self, landmarks):
                        self.landmark = landmarks

                simple_hand = SimpleHand(lm_list)

                # Draw simple landmarks for visibility
                for lm in lm_list:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv.circle(frame, (x, y), 2, (255, 200, 0), -1)

                handedness = None
                try:
                    handedness = task_result.handedness[hand_idx]
                except Exception:
                    handedness = None

                if is_gun_gesture(simple_hand):
                    draw_gun(frame, simple_hand, handedness)
                
                # Mouse control: move cursor to index finger position
                index_tip = lm_list[8]
                mouse_x = int(index_tip.x * screen_width)
                mouse_y = int(index_tip.y * screen_height)
                
                # Apply smoothing to mouse position
                smoothed_x, smoothed_y = smooth_mouse_position(
                    mouse_x, mouse_y,
                    mouse_state['smoothed_x'],
                    mouse_state['smoothed_y']
                )
                mouse_state['smoothed_x'] = smoothed_x
                mouse_state['smoothed_y'] = smoothed_y
                
                # Move mouse to smoothed index finger position
                try:
                    pyautogui.moveTo(smoothed_x, smoothed_y)
                except Exception:
                    pass
                
                # Detect pinch gesture
                is_pinching, pinch_distance = is_pinch_gesture(simple_hand)
                
                if is_pinching:
                    import time
                    current_time = time.time()
                    
                    # Detect click on pinch
                    if current_time - mouse_state['last_click_time'] > click_cooldown:
                        try:
                            pyautogui.click(button='left')
                            mouse_state['last_click_time'] = current_time
                        except Exception:
                            pass
                    
                    # Detect vertical movement for scroll while pinching
                    current_index_y = index_tip.y
                    if prev_index_y is not None:
                        y_diff = (prev_index_y - current_index_y) * screen_height
                        if abs(y_diff) > 15:  # threshold to avoid noise
                            try:
                                # Positive y_diff = finger moved up = scroll up
                                scroll_amount = int(y_diff / 25)
                                if abs(scroll_amount) > 0:
                                    pyautogui.scroll(scroll_amount)
                            except Exception:
                                pass
                    prev_index_y = current_index_y
                else:
                    # Reset scroll tracking when not pinching
                    prev_index_y = None

        # Face tracking (Tasks API) if face task landmarker exists
        mouth_is_o = False
        if 'face_task_landmarker' in globals() and face_task_landmarker is not None:
            try:
                mp_img = mp_image.Image.create_from_file(tmp_name)
                face_result = face_task_landmarker.detect(mp_img)
                if face_result and face_result.face_landmarks:
                    for flm in face_result.face_landmarks:
                        # Check if mouth is "O" shaped
                        mouth_is_o = is_mouth_o_shape(flm)
                        
                        H, W = frame.shape[:2]
                        pts = np.array([[int(p.x * W), int(p.y * H)] for p in flm], dtype=np.int32)
                        if pts.size:
                            # draw small landmark dots (like hand tracker)
                            for p in pts:
                                cv.circle(frame, (int(p[0]), int(p[1])), 2, (255, 200, 0), -1)
            except Exception:
                pass
        
        # Face tracking (Tasks API) if face task landmarker exists
        if 'face_task_landmarker' in globals() and face_task_landmarker is not None:
            try:
                mp_img = mp_image.Image.create_from_file(tmp_name)
                face_result = face_task_landmarker.detect(mp_img)
                if face_result and face_result.face_landmarks:
                    for flm in face_result.face_landmarks:
                        H, W = frame.shape[:2]
                        pts = np.array([[int(p.x * W), int(p.y * H)] for p in flm], dtype=np.int32)
                        if pts.size:
                            # draw small landmark dots (like hand tracker)
                            for p in pts:
                                cv.circle(frame, (int(p[0]), int(p[1])), 2, (255, 200, 0), -1)
            except Exception:
                pass
        
        # Show mouse mode indicator
        cv.putText(frame, "MOUSE: Active", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        # remove the temp file after both detections
        try:
            import os
            os.remove(tmp_name)
        except Exception:
            pass

    cv.imshow("live video" , frame)
    if cv.waitKey(2)== ord('q'):
        break

cap.release()
cv.destroyAllWindows()