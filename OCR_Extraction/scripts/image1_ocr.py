import cv2
import pytesseract
import os

image_path = '../data/extracted_images/extracted_images1/image1.jpeg'
output_path='../data/ocr_results/ocr_results1/raw_extraction/ticket_reservation.txt'
img = cv2.imread(image_path)
height,weight,_=img.shape
new_height=int(height/3)
top_part=img[0:new_height+10,:]
middle_part=img[new_height+10:2*new_height+10,:]
last_part=img[2*new_height+10:,:]

def preprocess(image):
   grayed=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   thresh=cv2.threshold(grayed,205,255,cv2.THRESH_BINARY_INV)[1]
   return thresh

preprocessed_top_part=preprocess(top_part)

top_part_extraction=pytesseract.image_to_string(preprocessed_top_part,lang='fra')
middle_part_extraction=pytesseract.image_to_string(middle_part,lang='fra+ara')
last_part_extraction=pytesseract.image_to_string(last_part,lang='fra+ara')

cv2.imshow('top',preprocessed_top_part)
cv2.imshow('middle',middle_part)
cv2.imshow('last',last_part)
cv2.waitKey(0)
cv2.destroyAllWindows()

with open(output_path, 'w', encoding='utf-8') as f:
   f.write(top_part_extraction)
   f.write(middle_part_extraction)
   f.write(last_part_extraction)