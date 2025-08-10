import cv2
import numpy as np
import pytesseract
import os 

input_path='../data/extracted_images/extracted_images1/image7.png'
output_path='../data/ocr_results/ocr_results1/raw_extraction/certificat_scolarite_corusRayan.txt'

img=cv2.imread(input_path)

def resize(image,dim):
    resized=cv2.resize(image,None,fx=dim,fy=dim,interpolation=cv2.INTER_CUBIC)
    return resized
def extract(image,language,confi=None):
    if confi:
      text=pytesseract.image_to_string(image,lang=language,config=confi)
    else:
       text=pytesseract.image_to_string(image,language)
    return text
 
img=img[:430,:570]
resized=resize(img,2)

top_left=img[30:110,:180]
top_left=resize(top_left,2)
extracted_top_left=extract(top_left,'fra')


top_right=img[30:110,440:]
top_right=resize(top_right,3)
extracted_top_right=extract(top_right,'fra')


top_middle=img[30:110,220:370]
top_middle=resize(top_middle,3)
extracted_top_middle=extract(top_middle,'fra')


title=img[115:140,140:400]
title=resize(title,2)
extracted_title=pytesseract.image_to_string(title,lang='fra')

center=img[145:300]
grayed=cv2.cvtColor(center,cv2.COLOR_BGR2GRAY)
center=resize(grayed,3)
extracted_center=extract(center,'fra','--psm 6')


bottom=img[320:410]

hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)

# Define a more accurate range for blue (you can tune this!)
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create blue mask
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Replace blue pixels with white
bottom[blue_mask > 0] = [255, 255, 255]

gray_bottom= cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
imv=cv2.bitwise_not(gray_bottom)
imv=resize(imv,2)
extracted_bottom=extract(imv,'fra','--psm 6')


with open(output_path, 'w', encoding='utf-8') as f:
   f.write(extracted_top_middle)
   f.write(extracted_top_left)
   f.write(extracted_top_right)
   f.write(extracted_title)
   f.write(extracted_center)
   f.write(extracted_bottom)




