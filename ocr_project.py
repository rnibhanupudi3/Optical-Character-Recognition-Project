from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2
import time

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--image", required=True, help="path to OCR input image")
args = vars(argparser.parse_args())

im = cv2.imread(args["image"])
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
(height, width) = gray.shape
th, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
gray = cv.fastNlMeansDenoising(gray,None,10,10,7,21)

#resize for debugging
"""
if height > 1080 or width > 1920:
    height = int(height * .5)
    width = int(width * .5) 
"""
#gray = cv2.resize(gray, (height, width), interpolation = cv2.INTER_AREA)

cv2.imshow("Processed Grayscale Image", gray)

rect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)) #change
sq = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)) #change

gray = cv2.GaussianBlur(gray, (3,3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)
cv2.imshow("Blackhat Image", blackhat)

gradient = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradient = np.absolute(gradient)
(min_val, max_val) = (np.min(gradient), np.max(gradient))
gradient = (gradient - min_val) / (max_val - min_val)
gradient = (gradient * 255).astype("uint8")
cv2.imshow("Scharr Gradient", gradient)

gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, rect)
threshold = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Rectangular Closing Operation", threshold)

threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, sq)
threshold = cv2.erode(threshold, None, iterations = 3)
cv2.imshow("Square Closing Operation", threshold)

contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, method="bottom-to-top")[0]

cv2.waitKey()

mrz = None
mrz2 = None

for c in contours:
    (x,y,w,h) = cv2.boundingRect(c)
    percentW = w/float(width)
    percentH = h/float(height)
    print(percentW)
    
    if percentW > .8 and percentH > .04:
        mrz = (x,y,w,h)
        break
    elif percentW > .8 and mrz is None:
        mrz = (x,y,w,h)
    elif percentW > .8:
        mrz2 = (x,y,w,h)
        break
    print(mrz2)

if mrz is None:
    print("MRZ not found")
    sys.exit(0)
    
(x,y,w,h) = mrz

pad_x = int((x + w) * .03)
pad_y = int((y + h) * .03)
(x,y) = (max(0, x - pad_x), y - pad_y)
(w,h) = (min(width - x, w + pad_x*2), min(height - y, h + pad_y*2))

mrz = im[y:y+h, x:x+w]

if mrz2:
    (x2, y2, w2, h2) = mrz2
    pad_x = int((x2 + w2) * .03)
    pad_y = int((y2 + h2) * .03)
    (x2,y2) = (max(0, x2 - pad_x), y2 - pad_y)
    (w2,h2) = (min(width - x2, w2 + pad_x*2), min(height - y2, h2 + pad_y*2))
    mrz2 = im[y2:y2+h2, x2:x2+w2]

if mrz2 is None:
    mrzText = pytesseract.image_to_string(mrz)
else:
    mrzText = pytesseract.image_to_string(mrz2) + pytesseract.image_to_string(mrz)

mrzText = mrzText.replace(" ", "")
print(mrzText)

cv2.imshow("MRZ", mrz)

if mrz2 is not None:
    cv2.imshow("MRZ2", mrz2)

cv2.waitKey()
cv2.destroyAllWindows()
sys.exit(0)


    