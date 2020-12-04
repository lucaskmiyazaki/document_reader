import pytesseract as ocr
import cv2 

img = cv2.imread("imgs/test.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h, w = gray.shape
resized = cv2.resize(gray, (w, h))
text = ocr.image_to_string(resized, lang='por_tax_doc')
cv2.imshow("test", resized)
cv2.waitKey(0)
print(text)

