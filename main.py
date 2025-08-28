import cv2
import numpy as np
import time
from datetime import datetime

class ObjectDetectionUI:
    def __init__(self):
        self.thres = 0.5
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Increased resolution
        self.cap.set(4, 720)
        
        # Load class names
        self.classNames = []
        classFile = 'coco.names'
        try:
            with open(classFile, 'rt') as f:
                self.classNames = f.read().rstrip('\n').split('\n')
        except FileNotFoundError:
            print("Warning: coco.names file not found. Using default classes.")
            self.classNames = ["person", "bicycle", "car", "motorcycle", "airplane"]
        
        # Load model
        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'
        
        try:
            self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
            self.net.setInputSize(320, 320)
            self.net.setInputScale(1.0/127.5)
            self.net.setInputMean((127.5, 127.5, 127.5))
            self.net.setInputSwapRB(True)
            self.model_loaded = True
        except:
            print("Warning: Model files not found. UI will show without detection.")
            self.model_loaded = False
        
        # UI Settings
        self.colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (150, 200, 100), (100, 150, 200)
        ]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection statistics
        self.detection_history = []
        self.show_stats = True
        self.paused = False
        
    def draw_header(self, img):
        """Draw header with title and controls info"""
        overlay = img.copy()
        height, width = img.shape[:2]
        
        # Semi-transparent header background
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(img, "AngelEyes - Object Detection", (10, 30), 
                   self.font, 0.8, (255, 255, 255), 2)
        
        # Current settings
        settings = f"Threshold: {self.thres:.1f} | FPS: {self.current_fps:.1f}"
        cv2.putText(img, settings, (10, 55), 
                   self.font, 0.5, (150, 255, 150), 1)
        
        return img
    
    def draw_footer_stats(self, img, detections):
        """Draw footer with detection statistics"""
        if not self.show_stats:
            return img
            
        height, width = img.shape[:2]
        overlay = img.copy()
        
        # Semi-transparent footer background
        footer_height = 80
        cv2.rectangle(overlay, (0, height - footer_height), (width, height), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Detection count and time
        detection_count = len(detections) if detections else 0
        current_time = datetime.now().strftime("%H:%M:%S")
        
        cv2.putText(img, f"Objects: {detection_count} | Time: {current_time}", 
                   (10, height - 50), self.font, 0.6, (100, 255, 100), 1)
        
        # Object list if any detected
        if detections and len(detections) > 0:
            detected_objects = []
            for detection in detections:
                if 'class_name' in detection:
                    detected_objects.append(detection['class_name'])
            
            if detected_objects:
                unique_objects = list(set(detected_objects))
                objects_text = "Detected: " + ", ".join(unique_objects[:4])
                if len(unique_objects) > 4:
                    objects_text += "..."
                cv2.putText(img, objects_text, 
                           (10, height - 20), self.font, 0.5, (255, 255, 100), 1)
        
        return img
    
    def draw_enhanced_bounding_boxes(self, img, detections):
        """Draw enhanced bounding boxes with better styling"""
        if not detections:
            return img
            
        for i, detection in enumerate(detections):
            box = detection['box']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x, y, w, h = box
            color = self.colors[i % len(self.colors)]
            
            # Main bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Corner decorations
            corner_length = 20
            corner_thickness = 3
            # Top-left corner
            cv2.line(img, (x, y), (x + corner_length, y), color, corner_thickness)
            cv2.line(img, (x, y), (x, y + corner_length), color, corner_thickness)
            # Top-right corner
            cv2.line(img, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
            cv2.line(img, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
            # Bottom-left corner
            cv2.line(img, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
            cv2.line(img, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
            # Bottom-right corner
            cv2.line(img, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
            cv2.line(img, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)
            
            # Label background
            label = f"{class_name.upper()}"
            confidence_text = f"{confidence*100:.1f}%"
            
            # Calculate label dimensions
            (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, self.font, 0.5, 1)
            
            # Adjust label position if too close to header
            label_y = max(y, 90)  # Keep labels below header
            
            # Draw label background
            label_bg_height = max(label_h, conf_h) + 15
            cv2.rectangle(img, (x, label_y - label_bg_height), (x + max(label_w, conf_w) + 15, label_y), color, -1)
            cv2.rectangle(img, (x, label_y - label_bg_height), (x + max(label_w, conf_w) + 15, label_y), (255, 255, 255), 1)
            
            # Draw text
            cv2.putText(img, label, (x + 8, label_y - label_bg_height + label_h + 3),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
            cv2.putText(img, confidence_text, (x + 8, label_y - 3),
                       self.font, 0.5, (255, 255, 255), 1)
        
        return img
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            end_time = time.time()
            self.current_fps = self.fps_counter / (end_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """Main application loop"""
        print("AngelEyes Object Detection Started!")
        print("Controls:")
        print("  Q - Quit")
        print("  S - Toggle Statistics")
        print("  P - Pause/Resume")
        print("  R - Reset Detection History")
        print("  + - Increase Threshold")
        print("  - - Decrease Threshold")
        print("\nPress any key to start...")
        
        while True:
            if not self.paused:
                success, img = self.cap.read()
                if not success:
                    print("Failed to read from camera")
                    break
            
            # Object detection
            detections = []
            if self.model_loaded and not self.paused:
                try:
                    classIds, confs, bbox = self.net.detect(img, confThreshold=self.thres)
                    
                    if len(classIds) != 0:
                        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                            if 0 <= classId-1 < len(self.classNames):
                                detections.append({
                                    'box': box,
                                    'class_name': self.classNames[classId-1],
                                    'confidence': confidence,
                                    'class_id': classId
                                })
                except Exception as e:
                    print(f"Detection error: {e}")
            
            # Update detection history
            if detections:
                self.detection_history.append({
                    'time': datetime.now(),
                    'count': len(detections),
                    'objects': [d['class_name'] for d in detections]
                })
                # Keep only last 100 detections
                if len(self.detection_history) > 100:
                    self.detection_history.pop(0)
            
            # Draw UI elements
            img = self.draw_header(img)
            img = self.draw_enhanced_bounding_boxes(img, detections)
            img = self.draw_footer_stats(img, detections)
            
            # Show pause indicator
            if self.paused:
                height, width = img.shape[:2]
                cv2.putText(img, "PAUSED - Press P to resume", 
                           (width//2 - 150, height//2), 
                           self.font, 1, (0, 0, 255), 2)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display
            cv2.imshow("AngelEyes - Object Detection", img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                self.show_stats = not self.show_stats
                print(f"Statistics: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('p') or key == ord('P'):
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Resumed'}")
            elif key == ord('r') or key == ord('R'):
                self.detection_history.clear()
                print("Detection history cleared")
            elif key == ord('+') or key == ord('='):
                self.thres = min(1.0, self.thres + 0.05)
                print(f"Threshold increased to {self.thres:.2f}")
            elif key == ord('-'):
                self.thres = max(0.1, self.thres - 0.05)
                print(f"Threshold decreased to {self.thres:.2f}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("AngelEyes Object Detection Stopped!")

# Run the application
if __name__ == "__main__":
    app = ObjectDetectionUI()
    app.run()