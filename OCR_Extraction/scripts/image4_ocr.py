import cv2
import numpy as np
import pytesseract
import os

input_path='../data/extracted_images/extracted_images1/image4.png'
output_path='../data/ocr_results/ocr_results1/raw_extraction/decompte_remboursement.txt'

img=cv2.imread(input_path)

sanlam_logo=img[270:340,150:]
grayed_logo=cv2.cvtColor(sanlam_logo,cv2.COLOR_BGR2GRAY)
extracted_logo=pytesseract.image_to_string(grayed_logo,lang='fra')

title=img[350:420,200:]
extracted_title=pytesseract.image_to_string(title,lang='fra')

data=img[430:493,350:]
grayed=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
resized_data=cv2.resize(grayed,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_data=pytesseract.image_to_string(resized_data,lang='fra',config="--psm 6")

table_row_1=img[530:565]
resized_table1=cv2.resize(table_row_1,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_table_row_1=pytesseract.image_to_string(resized_table1,lang='fra')

table_row_2=img[565:595,120:]
resized_table2=cv2.resize(table_row_2,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_table_row_2=pytesseract.image_to_string(resized_table2,lang='fra')

table_row_3=img[590:650,:]
resized_table3=cv2.resize(table_row_3,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
extracted_table_row_3=pytesseract.image_to_string(resized_table3,lang='fra',config="--psm 6")

 
reglement_compagnie=img[650:670,430:700]
grayed_reglement=cv2.cvtColor(reglement_compagnie,cv2.COLOR_BGR2GRAY)
invert=cv2.bitwise_not(grayed_reglement)
resized_reglement_compagnie=cv2.resize(invert,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
extracted_reglement_compagnie=pytesseract.image_to_string(resized_reglement_compagnie,lang='fra')



date_lieu=img[670:900,:]
resized_date_lieu=cv2.resize(date_lieu,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
inv=cv2.bitwise_not(resized_date_lieu)
extracted_date_lieu=pytesseract.image_to_string(inv,lang='fra',)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_logo)
    f.write(extracted_title)
    f.write(extracted_data)
    f.write(extracted_table_row_1)
    f.write(extracted_table_row_2)
    f.write(extracted_table_row_3)
    f.write(extracted_reglement_compagnie)
    f.write(extracted_date_lieu)

