import cv2
import pytesseract
import os

image_path = '../data/extracted_images/extracted_images1/image3.jpeg'
output_path='../data/ocr_results/ocr_results1/raw_extraction/remboursement_cnops1.txt'
img=cv2.imread(image_path)
height,width,_=img.shape

header=img[50:100,:]
extracted_text_header=pytesseract.image_to_string(header)

remboursement=img[320:360,270:340]
extracted_remboursement=pytesseract.image_to_string(remboursement,lang='fra')


state=img[430:480,90:190]
bigger_state=cv2.resize(state,(state.shape[1]*4,state.shape[0]*4))
extracted_state=pytesseract.image_to_string(bigger_state,lang='fra')

table=img[480:600,:]
grayed_table=cv2.cvtColor(table,cv2.COLOR_BGR2GRAY)
bigger_table=cv2.resize(grayed_table,(table.shape[1]*2,table.shape[0]*2))
inv=cv2.bitwise_not(bigger_table)

extracted_table=pytesseract.image_to_string(inv,lang='fra',config='--psm 6')


with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_text_header)
    f.write(extracted_remboursement)
    f.write(extracted_state)
    f.write(extracted_table)



