import cv2
import pytesseract
import os

image_path = '../data/extracted_images/extracted_images1/image6.png'
output_path='../data/ocr_results/ocr_results1/raw_extraction/certificat_scolarite_arabe.txt'

img=cv2.imread(image_path)

def resize(image,dim):
    resized=cv2.resize(image,None,fx=dim,fy=dim,interpolation=cv2.INTER_CUBIC)
    return resized
def extract(image,language,confi=None):
    if confi:
      text=pytesseract.image_to_string(image,lang=language,config=confi)
    else:
       text=pytesseract.image_to_string(image,language)
    return text

ministere=img[:100,430:540]
inv=cv2.bitwise_not(ministere)
resized_ministere=cv2.resize(inv,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
extracted_ministere=pytesseract.image_to_string(resized_ministere,lang='ara')


top_left=img[:110,:290]
resized_top_left=resize(top_left,2)
extracted_top_left=extract(resized_top_left,'ara')

top_right=img[7:110,550:]
resized_top_right=resize(top_right,3)
extracted_top_right=extract(resized_top_right,'ara')


title=img[115:180,:515]
extracted_title=extract(title,'ara')


middle=img[220:400,125:]
resized_middle=resize(middle,2)
extracted_middle=extract(resized_middle,'ara')


bottom_right=img[410:550,350:]
resized_bottom_right=resize(bottom_right,2)
extracted_bottom_right=extract(resized_bottom_right,'ara')


bottom_left=img[410:550,:340]
gray=cv2.cvtColor(bottom_left,cv2.COLOR_BGR2GRAY)
thresh=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)[1]
resized_bottom_left=resize(bottom_left,2)
extracted_bottom_left=extract(thresh,'ara')

with open(output_path, 'w', encoding='utf-8') as f:
   f.write(extracted_ministere)
   f.write(extracted_top_left)
   f.write(extracted_top_right)
   f.write(extracted_title)
   f.write(extracted_middle)
   f.write(extracted_bottom_right)
   f.write(extracted_bottom_left)



