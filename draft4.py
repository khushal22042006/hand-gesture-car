import cv2 
import numpy as np
import mediapipe as mp
import serial 
import time


# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set up serial communication for Bluetooth (ensure the port is correct)
arduino = serial.Serial('COM4', 9600, timeout=1)  # Replace 'COM5' with your HC-05 Bluetooth port
time.sleep(2)  # Allow some time for the connection to establish

# Define a function to classify gestures
def get_gesture(landmarks):
    """
    Determines the gesture based on finger landmarks.
    """
    wrist = landmarks[0]
    index_finger_tip = landmarks[8]
    middle_finger_tip = landmarks[12]
    ring_finger_tip = landmarks[16]
    pinky_finger_tip = landmarks[20]
    thumb_tip = landmarks[4]
    
    # Check for fist
    distances = [
        np.linalg.norm([index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y]),
        np.linalg.norm([middle_finger_tip.x - wrist.x, middle_finger_tip.y - wrist.y]),
        np.linalg.norm([ring_finger_tip.x - wrist.x, ring_finger_tip.y - wrist.y]),
        np.linalg.norm([pinky_finger_tip.x - wrist.x, pinky_finger_tip.y - wrist.y]),
        np.linalg.norm([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y]),
    ]
    
    if all(distance < 0.28 for distance in distances):  # for "closed fist"
        return "Stop"
    if thumb_tip.y>wrist.y :
        if thumb_tip.x == wrist.x :
         return "back"
    if index_finger_tip.y < wrist.y:  # Fingers are pointing
        if index_finger_tip.x < wrist.x - 0.2:  # Pointing left
            return "Left"
        elif index_finger_tip.x > wrist.x + 0.2:  # Pointing right
            return "Right"
        else:  # Fingers pointing straight
            return "Straight"
        
    return "Unknown"

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        print("Unable to access the camera.")
        break
    
    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get gesture based on landmarks
            gesture = get_gesture(hand_landmarks.landmark)
            
            # Display the gesture on the frame
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Send the gesture to the Arduino via Bluetooth
            if gesture == "Stop":
                arduino.write(b'S')  # Send 'S' for stop
            elif gesture == "Left":
                arduino.write(b'L')  # Send 'L' for left
            elif gesture == "Right":
                arduino.write(b'R')  # Send 'R' for right
            elif gesture == "Straight":
                arduino.write(b'F')  # Send 'F' for forward
            elif gesture == "back":
                arduino.write(b'B')  # Send 'B' for forward
            else:
                arduino.write(b'S')  # Default to stop if gesture is unknown
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
