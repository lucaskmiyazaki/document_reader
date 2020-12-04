from pdf2image import convert_from_path
import sys
import requests
import cv2
import re

def sort_words(line_infos):
    full_text = ''
    text = []
    for line in line_infos:
        for word_metadata in line:
            words = []
            for word_info in word_metadata["words"]:
                print(word_info)
                words.append(word_info["text"])
            text = ' '.join(words)
            full_text += text
            full_text += '\n'
    return full_text


def microsoftOCR(frame, debug):
    subscription_key = "1dasd663b7asdadsd3379053450a5"
    endpoint = "https://v360testingocr.cognitiveservices.azure.com/"
    ocr_url = endpoint + "vision/v3.0/ocr"
    #image_path = "test.jpg"
    #image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'language': 'pt', 'detectOrientation': 'true'}

    _, encoded_image = cv2.imencode('.jpg', frame)
    buf = encoded_image.tobytes()
    response = requests.post(ocr_url, headers=headers, params=params, data=buf)
    response.raise_for_status()
    analysis = response.json()
    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    full_text = sort_words(line_infos)

    if debug:
        print(full_text)

    return full_text

def ocr_manager(file_path):
    #file_path = "pdfs/2NGT-PCJ8.pdf"
    pages=convert_from_path(file_path)
    #for page in pages:
    pages[0].save("tax.jpeg","jpeg")
    img = cv2.imread("tax.jpeg")
    full_text = microsoftOCR(img, True)
    return full_text
#file1 = open("tax.txt", "w")
#file1.write(full_text)
#file1.close()

