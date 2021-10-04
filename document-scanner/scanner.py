import cv2
import numpy as np
import imutils

class Scanner(object):
    def apply_filters(self, image):
        # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # Test 1
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # edged = cv2.Canny(gray, 75, 200)
        # Test 2
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edged = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # edged = cv2.Canny(thres, 200, 200)
        # # Test 3
        # imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        # imgCanny = cv2.Canny(imgBlur, 200, 200)
        # kernel = np.ones((5, 5))
        # imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
        # imgThres = cv2.erode(imgDial, kernel, iterations=1)
        # edged = imgThres
        # Test 4
        # gray = cv2.GaussianBlur(image,(3,3),2)
        # threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        # threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
        # edged = cv2.Canny(threshold,50,150,apertureSize = 7)
        return edged
    
    def draw_contours(self, edged, original):
        edged_copy = edged.copy()
        original_copy = original.copy()
        # find the contours in the edged image
        cnts = cv2.findContours(edged_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found the document
            cv2.drawContours(original_copy, [approx], -1, (255, 0, 0), 2)

            if len(approx) == 4:
                screenCnt = approx
                break

        try:
            cv2.drawContours(original_copy, [screenCnt], -1, (0, 255, 0), 2)
        except:
            print("Failed to find rectangle")

        # for immutability
        return original_copy




