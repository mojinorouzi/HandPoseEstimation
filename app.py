from cv2 import LINE_AA
import mediapipe as mp 
import cv2
import numpy as np 
import uuid 
import os 
from utiles import get_label

mp_drawing = mp.solutions.drawing_utils 
mp_hands = mp.solutions.hands




cap = cv2.VideoCapture(0)

""" 
 ** mp_hands.Hands Args
 min_detection_confidence: threshold for the initial detection to be successful.
 min_tracking_confidence : threshold for tracking after initial detection.
 max_num_hands ; maximum number of hands to detect. Default to 2.
"""

with mp_hands.Hands(min_detection_confidence=0.8 , min_tracking_confidence=0.5 , max_num_hands=10) as hands:
    while cap.isOpened():
        ret , frame = cap.read()
        
        # BGR to RGB
        image = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        
        # SET FLAGS
        image.flags.writeable = False
        
        # DETECTION
        results = hands.process(image)
        
        # SET FLAGS
        image.flags.writeable = True
        
        # RGB To BGR
        image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
        
        # DETECTION
        # Rendering results 
        if results.multi_hand_landmarks:
            for num , hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand , mp_hands.HAND_CONNECTIONS , 
                                          # DrawingSpec is a mediapipe class that allows you yo customize the look of your detection
                                          mp_drawing.DrawingSpec(color=(121 , 22 , 76) , thickness=2 , circle_radius=3) , 
                                          mp_drawing.DrawingSpec(color=(180 , 22 , 120) , thickness=2 , circle_radius=2)
                                          )
                if get_label(num , hand , results, mp_hands):
                    text , coord = get_label(num , hand , results , mp_hands)
                    cv2.putText(image, text, coord , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 2 , cv2.LINE_AA)
        cv2.imshow("Hand Tracking" , image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    
cap.release()
cv2.destroyAllWindows()
