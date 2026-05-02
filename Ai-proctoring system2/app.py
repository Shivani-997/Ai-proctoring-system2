import cv2
import numpy as np
from datetime import datetime
from database_connection import create_proctoring_session, insert_proctoring_frame, insert_suspicious_activity
import threading

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print("✅ Face detection initialized with OpenCV")

# Simple object detection function
def detect_objects(frame):
    """Detect objects in hand (simplified)"""
    h, w = frame.shape[:2]
    objects_detected = []
    
    try:
        # Define region of interest (lower half - where hands usually are)
        roi = frame[h//2:h, 0:w]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common objects
        # Phone/Electronic - Black, Silver, Blue
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        
        # Book/Paper - White, Brown
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        
        # Skin color for hands
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        # Create masks
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Combine masks (exclude skin)
        mask = cv2.bitwise_or(mask_black, mask_brown)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_skin))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum area to filter noise
                x, y, w_box, h_box = cv2.boundingRect(contour)
                # Adjust y coordinate back to original frame
                y += h//2
                objects_detected.append({
                    'bbox': (x, y, w_box, h_box),
                    'area': area
                })
    except Exception as e:
        print(f"❌ Object detection error: {e}")
    
    return objects_detected

def get_head_direction(face_x, face_y, face_w, face_h, frame_w, frame_h):
    """Get head direction based on face position"""
    face_center_x = face_x + face_w/2
    face_center_y = face_y + face_h/2
    frame_center_x = frame_w/2
    frame_center_y = frame_h/2
    
    # Calculate offset
    x_offset = face_center_x - frame_center_x
    y_offset = face_center_y - frame_center_y
    
    # Normalize
    x_offset_norm = x_offset / (frame_w/2)
    y_offset_norm = y_offset / (frame_h/2)
    
    # Determine direction
    direction = "FORWARD"
    
    if x_offset_norm < -0.1:
        direction = "LOOKING RIGHT"
    elif x_offset_norm > 0.1:
        direction = "LOOKING LEFT"
    elif y_offset_norm < -0.1:
        direction = "LOOKING UP"
    elif y_offset_norm > 0.1:
        direction = "LOOKING DOWN"
    
    return direction, x_offset_norm, y_offset_norm

def detect_eyes(face_roi, x, y):
    """Detect eyes in face region"""
    eyes = eye_cascade.detectMultiScale(face_roi)
    eye_count = len(eyes)
    
    # Draw eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
    
    return eye_count

def proctoringAlgo(user_email=None):
    """Proctoring with face and object detection"""
    
    print(f"\n{'='*50}")
    print(f"🔥 PROCTORING STARTED FOR: {user_email}")
    print(f"✅ Features: Face Detection | Head Movement | Eye Detection | Object Detection")
    print(f"{'='*50}\n")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Camera not available")
        return
    
    # Create session
    session_id = create_proctoring_session(user_email or 'unknown')
    print(f"✅ Session ID: {session_id}")
    
    # Tracking variables
    frame_count = 0
    no_face_counter = 0
    looking_away_counter = 0
    eye_blink_counter = 0
    object_detected_counter = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
        face_count = len(faces)
        face_detected = face_count > 0
        
        # Default values
        head_direction = "NO FACE"
        eye_count = 0
        x_offset = 0
        y_offset = 0
        
        # Process faces
        if face_detected:
            # Get the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = largest_face
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 3)
            
            # Get head direction
            head_direction, x_offset, y_offset = get_head_direction(x, y, fw, fh, w, h)
            
            # Extract face ROI for eye detection
            face_roi = gray[y:y+fh, x:x+fw]
            if face_roi.size > 0:
                eye_count = detect_eyes(face_roi, x, y)
            
            # Track looking away
            if head_direction != "FORWARD":
                looking_away_counter += 1
                if looking_away_counter > 30:
                    print(f"⚠️ Looking away: {head_direction}")
                    insert_suspicious_activity(session_id, None, 'looking_away', f'Head: {head_direction}')
                    looking_away_counter = 0
            else:
                looking_away_counter = max(0, looking_away_counter - 1)
            
            # Track eye blinks
            if eye_count < 2:
                eye_blink_counter += 1
                if eye_blink_counter > 10:
                    print(f"👁️ Possible blink detected")
                    threading.Thread(target=insert_suspicious_activity, args=(
                        session_id, None, 'blink', 'Eye blink detected'
                    )).start()
                    eye_blink_counter = 0
            else:
                eye_blink_counter = max(0, eye_blink_counter - 1)
            
            # Multiple faces alert
            if face_count > 1:
                print(f"⚠️ MULTIPLE FACES: {face_count}")
                threading.Thread(target=insert_suspicious_activity, args=(
                    session_id, None, 'multiple_faces', f'{face_count} faces detected'
                )).start()
        
        else:
            no_face_counter += 1
            if no_face_counter > 30:
                print(f"⚠️ No face detected")
                threading.Thread(target=insert_suspicious_activity, args=(
                    session_id, None, 'no_face', 'No face in frame'
                )).start()
                no_face_counter = 0
        
        # ===== OBJECT DETECTION =====
        objects = []  
        if frame_count % 15 == 0:  # Har 15 frame pe check
            objects = detect_objects(frame)
            if objects:
                object_detected_counter += 1
                for obj in objects:
                    x, y, w_box, h_box = obj['bbox']
                    cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)
                    cv2.putText(frame, "⚠️ OBJECT", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Log suspicious activity (every 5 detections)
                if object_detected_counter % 5 == 0:
                    threading.Thread(target=insert_suspicious_activity, args=(
                        session_id, None, 'object_detected', f'Object detected in hand'
                    )).start()
        
        # ===== DRAW ON FRAME =====
        # Black background for text
        cv2.rectangle(frame, (0, 0), (500, 280), (0, 0, 0), -1)
        
        # Draw text
        y_pos = 30
        line_height = 30
        
        cv2.putText(frame, f"USER: {user_email}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_pos += line_height
        
        cv2.putText(frame, f"FRAME: {frame_count}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_pos += line_height
        
        # Face count
        if face_count == 0:
            color = (255, 165, 0)
            text = f"FACES: {face_count} (NO FACE!)"
        elif face_count == 1:
            color = (0, 255, 0)
            text = f"FACES: {face_count}"
        else:
            color = (0, 0, 255)
            text = f"FACES: {face_count} ⚠️"
        
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_pos += line_height
        
        cv2.putText(frame, f"HEAD: {head_direction}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_pos += line_height
        
        cv2.putText(frame, f"EYES: {eye_count}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += line_height
        
        # Object detection status
        obj_status = f"OBJECTS: {len(objects)} detected" if objects else "OBJECTS: None"
        cv2.putText(frame, obj_status, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255) if objects else (200, 200, 200), 2)
        y_pos += line_height
        
        cv2.putText(frame, f"OFFSET: X:{x_offset:.2f} Y:{y_offset:.2f}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Timestamp
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp_str, (w-150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Alert banners
        if face_count > 1:
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
            cv2.putText(frame, f"⚠️ MULTIPLE FACES ({face_count}) ⚠️", 
                       (w//2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        if face_count == 0 and frame_count > 30:
            cv2.rectangle(frame, (0, 0), (w, 60), (255, 165, 0), -1)
            cv2.putText(frame, "⚠️ NO FACE DETECTED ⚠️", 
                       (w//2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        if objects:
            cv2.rectangle(frame, (0, h-60), (w, h), (255, 0, 0), -1)
            cv2.putText(frame, f"⚠️ OBJECT DETECTED IN HAND ⚠️", 
                       (w//2 - 200, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        # Save frame every 30 frames
        if frame_count % 30 == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            frame_to_save = cv2.resize(frame.copy(), (640, 480))
            
            face_data = {
                'face_detected': face_detected,
                'face_count': face_count,
                'head_pose': head_direction,
                'objects_detected': len(objects)
            }
            
            frame_id = insert_proctoring_frame(session_id, timestamp, frame_to_save, face_data)
            if frame_id:
                obj_msg = f" Objects: {len(objects)}" if objects else ""
                print(f"✅ Frame {frame_count} saved | Faces: {face_count} | Head: {head_direction}{obj_msg}")
        
        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()