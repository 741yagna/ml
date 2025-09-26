"""
Real-Time Fitness Trainer using BlazePose
Run this script in VS Code with webcam connected
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebcamFitnessTrainer:
    def __init__(self):
        logger.info("Initializing Fitness Trainer...")
        
        try:
            # Initialize MediaPipe Pose (BlazePose)
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Configure BlazePose model
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            self.current_exercise = "squat"
            logger.info("Fitness Trainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Fitness Trainer: {e}")
            sys.exit(1)
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1, 1))
            return np.degrees(angle)
        except Exception as e:
            logger.warning(f"Angle calculation error: {e}")
            return 0
    
    def analyze_squat(self, landmarks):
        """Analyze squat form"""
        feedback = []
        metrics = {}
        
        try:
            # Get landmarks
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            
            # Calculate angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            metrics['knee_angle'] = avg_knee_angle
            
            # Squat depth assessment
            if avg_knee_angle < 80:
                feedback.append("✅ Excellent depth!")
            elif avg_knee_angle < 100:
                feedback.append("✅ Good depth")
            else:
                feedback.append("⚠️ Go deeper - aim for 90°")
            
            # Knee alignment
            knee_distance = abs(left_knee.x - right_knee.x)
            if knee_distance > 0.15:
                feedback.append("❌ Knees caving in!")
            else:
                feedback.append("✅ Good knee alignment")
            
            # Back posture
            back_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            if back_angle < 150:
                feedback.append("❌ Keep back straight!")
            else:
                feedback.append("✅ Good back posture")
                
        except Exception as e:
            logger.warning(f"Squat analysis error: {e}")
            feedback = ["❌ Cannot detect full body - step back"]
            metrics = {'error': True}
        
        return feedback, metrics

    def process_webcam_frame(self, frame):
        """Process each webcam frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process with BlazePose
            results = self.pose.process(rgb_frame)
            
            rgb_frame.flags.writeable = True
            output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            feedback = []
            metrics = {}
            
            if results.pose_landmarks:
                feedback, metrics = self.analyze_squat(results.pose_landmarks.landmark)
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            return output_frame, feedback, metrics
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, ["❌ Processing error"], {}

def check_webcam():
    """Check if webcam is accessible"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot access webcam. Please check:")
        logger.error("1. Webcam is connected")
        logger.error("2. No other app is using the webcam")
        logger.error("3. Permissions are granted")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Webcam accessed but cannot read frames")
        return False
    
    logger.info("Webcam check passed")
    return True

def main():
    """Main function to run the fitness trainer"""
    logger.info("Starting Fitness Trainer...")
    
    # Check webcam first
    if not check_webcam():
        sys.exit(1)
    
    # Initialize fitness trainer
    trainer = WebcamFitnessTrainer()
    
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        logger.info("Webcam initialized successfully")
        
    except Exception as e:
        logger.error(f"Webcam initialization failed: {e}")
        sys.exit(1)
    
    print("\n=== Real-Time Fitness Trainer ===")
    print("Press 'q' to Quit")
    print("Make sure you have enough space and good lighting!")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Cannot read frame from webcam")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, feedback, metrics = trainer.process_webcam_frame(frame)
            
            # Display exercise info
            cv2.putText(processed_frame, f"Exercise: {trainer.current_exercise.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Display metrics if available
            if 'knee_angle' in metrics:
                cv2.putText(processed_frame, f"Knee Angle: {metrics['knee_angle']:.1f}°", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display feedback messages
            y_offset = 110
            for i, message in enumerate(feedback):
                color = (0, 255, 0) if message.startswith("✅") else (0, 165, 255) if message.startswith("⚠️") else (0, 0, 255)
                cv2.putText(processed_frame, message, (10, y_offset + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show frame
            cv2.imshow('Real-Time Fitness Trainer - BlazePose', processed_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                break
                
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Fitness trainer closed successfully")

if __name__ == "__main__":
    main()