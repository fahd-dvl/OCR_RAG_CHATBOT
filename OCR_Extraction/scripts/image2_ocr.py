import cv2
import numpy as np
import pytesseract

image_path = '../data/extracted_images/extracted_images1/image2.png'
img = cv2.imread(image_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = (90, 50, 50)
upper_blue = (130, 255, 255)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.dilate(mask, kernel, iterations=1)
result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
grayed = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(grayed)
thresh = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY_INV)[1]
height, width = thresh.shape
resized = cv2.resize(thresh, (width*2, height*2), interpolation=cv2.INTER_CUBIC)


cv2.imshow('Resized', resized)
cv2.imshow('thresh', thresh)
cv2.imshow('Enhanced enhanced', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

extracted_text = pytesseract.image_to_string(resized, lang= 'fra')
print(extracted_text)