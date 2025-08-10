import cv2
import numpy as np
import pytesseract
import os

image_path = '../data/extracted_images/extracted_images2/image1rot.png'
output_path='../data/ocr_results/ocr_results2/tresorerie_generale_royaume.txt'
img = cv2.imread(image_path)

def preprocess(image):
  rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
  grayed = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
  binary = cv2.threshold(grayed, 90,255,cv2.THRESH_BINARY_INV)[1]
  return binary


preprocessed_img=preprocess(img)

extracted_text = pytesseract.image_to_string(preprocessed_img, lang='fra')

with open(output_path, 'w', encoding='utf-8') as f:
  f.write(extracted_text)
