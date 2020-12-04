Tax Reader 

This algorithm tries to extract informations from a brazilian service tax document in a image format (.pdf).
1. Preprocessing: Segmentation and filter applications
2. Training: Segmentation of each field in the image (training set) and data crossing with the real info (in csv file)
3. OCR: use of tesseract or microsoft azure to identify words
4. Parser: use of regular expressions to extract fields of interest


Trainning
create a csv file (csv/notas.csv) with all tax doc real info (id, verification-code, number, ...)
add all documents in pdf to /pdfs
run train_segme.py to generate .tif and .gt.txt from /pdfs and csv/notas.csv
clone git@github.com:tesseract-ocr/tesstrain.git
create data folder and add por_tax_doc-ground-truth with all .tif and gt.txt files
from git@github.com:tesseract-ocr/tessdata_best.git add start model .traineddata file
make training MODEL_NAME=por_tax_doc START_MODEL=por PSM=7 
export TESSDATA_PREFIX="tessdata" (add your traineddata file to /tessdata inside this project)
pytesseract.to_string(text, lang="por_tax_doc") (change this line in houghOCR.py)

Identification
add all documents in pdf to /pdfs
run parser.py
use compare.py to check what is the accuracy (use mode='microsoft' or 'tesseract')
