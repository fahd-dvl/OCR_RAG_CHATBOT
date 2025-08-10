import cv2
import pytesseract
import os

image_path = '../data/extracted_images/extracted_images1/image10.png'
output_path='../data/ocr_results/ocr_results1/raw_extraction/erreur_technique.txt'

img=cv2.imread(image_path)
height,width,_=img.shape
new_height=height//3
new_width=width//3
resized_img = cv2.resize(img, (new_width, new_height))
error_message=resized_img[250:550,250:1000]
grayed=cv2.cvtColor(error_message,cv2.COLOR_BGR2GRAY)
res=cv2.resize(grayed,(grayed.shape[1]*2,grayed.shape[0]*2))
extracted_text=pytesseract.image_to_string(res,lang='fra')

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_text)