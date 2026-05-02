import cv2
import numpy as np

class ObjectDetector:
    """YOLO-based object detection for proctoring"""
    
    def __init__(self):
        self.model = None
        self.classes = []
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            # Try to load YOLOv4-tiny (fast for real-time)
            self.net = cv2.dnn.readNet(
                'yolov4-tiny.weights', 
                'yolov4-tiny.cfg'
            )
            
            # Load class names
            with open('coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            print("✅ YOLO model loaded successfully")
            
        except Exception as e:
            print(f"❌ YOLO model load failed: {e}")
            print("📌 Using fallback detection (simulated)")
            self.net = None
    
    def detect(self, frame):
        """Detect objects in frame"""
        detections = []
        
        if self.net is not None:
            # YOLO detection
            height, width = frame.shape[:2]
            
            # Prepare blob
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            # Process detections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:  # Threshold
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Filter relevant objects for proctoring
            proctoring_classes = [
                'cell phone', 'book', 'laptop', 'tv', 'remote',
                'person', 'bottle', 'cup', 'chair', 'couch'
            ]
            
            if len(indexes) > 0:
                for i in indexes.flatten():
                    class_name = self.classes[class_ids[i]]
                    if class_name in proctoring_classes:
                        x, y, w, h = boxes[i]
                        detections.append({
                            'class': class_name,
                            'confidence': confidences[i],
                            'box': (x, y, x+w, y+h)
                        })
        
        else:
            # Simulated detection for testing
            if np.random.random() > 0.95:  # 5% chance
                h, w = frame.shape[:2]
                detections.append({
                    'class': 'cell phone',
                    'confidence': 0.85,
                    'box': (w//2-50, h//2-30, w//2+50, h//2+30)
                })
        
        return detections