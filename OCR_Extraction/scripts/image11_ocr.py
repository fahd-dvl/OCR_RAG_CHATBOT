import cv2
import numpy as np
import pytesseract
import os

input_path='../data/extracted_images/extracted_images1/image11.png'
output_path='../data/ocr_results/ocr_results1/raw_extraction/attestation_interets.txt'

img=cv2.imread(input_path)


logo=img[70:105,:]
resized_logo=cv2.resize(logo,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_logo=pytesseract.image_to_string(resized_logo,lang='fra+ara')

title=img[150:240,:]
resized_title=cv2.resize(title,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
extracted_title=pytesseract.image_to_string(resized_title,lang='fra')

text=img[260:400,:]
grayed=cv2.cvtColor(text,cv2.COLOR_BGR2GRAY)
thresh=cv2.threshold(grayed,210,255,cv2.THRESH_BINARY_INV)[1]
kernel=np.ones((2,2),np.uint8)
resized_text=cv2.resize(thresh,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
dilated=cv2.dilate(resized_text,kernel,iterations=1)
eroded=cv2.erode(dilated,kernel,iterations=1)
extracted_text=pytesseract.image_to_string(eroded,lang='fra')


interets=img[400:420,:]
resized_interets=cv2.resize(interets,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
extracted_interets=pytesseract.image_to_string(resized_interets,config=custom_config)

gestionnaire=img[540:570,:280]
resized_gestionnaire=cv2.resize(gestionnaire,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_gestionnaire=pytesseract.image_to_string(resized_gestionnaire)


signature_directeur=img[540:650,300:]
grayed_signature=cv2.cvtColor(signature_directeur,cv2.COLOR_BGR2GRAY)
resized_signature=cv2.resize(grayed_signature,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_signature=pytesseract.image_to_string(resized_signature)



date_lieu=img[430:520,:]
resized_date_lieu=cv2.resize(date_lieu,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_date_lieu=pytesseract.image_to_string(resized_date_lieu,lang='fra')


with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_logo)
    f.write(extracted_title)
    f.write(extracted_text)
    f.write(extracted_interets)
    f.write(extracted_date_lieu)
    f.write(extracted_gestionnaire)
    f.write(extracted_signature)
