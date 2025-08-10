import cv2
import numpy as np
import pytesseract
import os
image_path = '../data/extracted_images/extracted_images1/image5.png'
output_path='../data/ocr_results/ocr_results1/raw_extraction/cnops2.txt'
img = cv2.imread(image_path)
grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(grayed, 45, 255, cv2.THRESH_BINARY_INV)[1]
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

cropped = img[y:y+h, x:x+w]
cropped_resized = cv2.resize(cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)



extracted_text= pytesseract.image_to_string(cropped_resized, lang='fra',config='--psm 6')



with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_text)