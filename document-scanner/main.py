import cv2
import numpy as numpy
from scanner import Scanner
import utils

# Webcam
cap = cv2.VideoCapture('https://192.168.68.57:8080/video') # 0 for webcam, else, specify path of video
# Scanner
scanner = Scanner()

while True: 
    
    ret, frame = cap.read(0) 
    
    original = frame
    edged = scanner.apply_filters(frame.copy())
    contours = scanner.draw_contours(edged.copy(), original)

    original = cv2.putText(original, 'Original', (100, 100), cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,0), 2);
    edged = cv2.putText(edged, 'Edged', (100, 100), cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,0), 2);
    contours = cv2.putText(contours, 'Contours', (100, 100), cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,0), 2);

    images = utils.stackImages(0.2, [original, edged, contours])
    cv2.imshow("Scanner ", images)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release() 
cv2.destroyAllWindows()