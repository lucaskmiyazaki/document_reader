import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract as ocr
import sys
import time
import requests
import re
import glob
import pandas as pd
from fuzzywuzzy import fuzz
# parametrizar variaveis
# enviar imagem com filtro para tesseract

def region_growing(pts, map_pts, img, n):
    h, w = img.shape
    max_x = 0
    max_y = 0
    min_x = w
    min_y = h
    for pt in pts:
        pt = (pt[0], pt[1])
        x, y = pt
        if img[y][x] == 0 and map_pts[y][x] == 0:
            stack = []
            stack.append(pt)
            map_pts[y][x] = 255
            while len(stack) > 0:
                x, y = stack.pop()
                if x > max_x: max_x = x
                if y > max_y: max_y = y
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if y+1 < h  and img[y+1][x] == 0 and map_pts[y+1][x] == 0:
                    stack.append((x, y+1))
                    map_pts[y+1][x] = 255
                if y-1 >= 0 and img[y-1][x] == 0 and map_pts[y-1][x] == 0:
                    stack.append((x, y-1))
                    map_pts[y-1][x] = 255
                if x+1 < w  and img[y][x+1] == 0 and map_pts[y][x+1] == 0:
                    stack.append((x+1, y))
                    map_pts[y][x+1] = 255
                if x-1 >= 0 and img[y][x-1] == 0 and map_pts[y][x-1] == 0:
                    stack.append((x-1, y))
                    map_pts[y][x-1] = 255
    #cv2.imwrite("map%d.jpg"%n, map_pts)
    #cv2.imshow("oi",map_pts)
    #cv2.waitKey(0)
    return map_pts, max_x, max_y, min_x, min_y

def rescale_frame(frame, percent=0.75):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def find_seed(frame):
    h, w = frame.shape
    half = h//2
    seed = -1
    thickness = 0
    for i in range(w):
        if frame[half][i] == 0:
            seed = i
            thickness += 1
        elif seed != -1:
            return seed, half, thickness*2
    raise Exception("seed not found")

def find_seed2(frame, min_length=100, max_gap=0):
    edges = (255 - frame)#cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges, rho = 1,theta = np.pi/2,threshold = 100, maxLineGap= max_gap,minLineLength= min_length)
    min_thick = 100
    if lines is not None:
        d1, d2, d3 = lines.shape
        lines = lines.reshape((int(d1*d2*d3/2), 2))
        lines = lines.tolist()
        for i in range(1, len(lines), 2):
            x1 = lines[i-1][0]
            y1 = lines[i-1][1]
            x2 = lines[i][0]
            y2 = lines[i][1]
            thickness = find_thickness(edges, x1, y2, x2, y2)
            if thickness < min_thick: min_thick = thickness
            #cv2.line(edges, (x1, y1), (x2, y2), 255, 20)
        #cv2.imshow("oi", edges)
        #cv2.waitKey(0)
        #print(min_thick)
    return lines, min_thick

def find_thickness(frame, x1, y1, x2, y2):
    if abs(x1-x2) > abs(y1 - y2):
        thickness = 1
        count = 0
        while frame[count + (y1+y2)//2][(x1+x2)//2] == 255:
            thickness += 1
            count += 1
        count = 0
        while frame[-count + (y1+y2)//2][(x1+x2)//2] == 255:
            thickness += 1
            count += 1
        return thickness
    else:
        thickness = 1
        count = 0
        while frame[(y1+y2)//2][count + (x1+x2)//2] == 255:
            thickness += 1
            count += 1
        count = 0
        while frame[(y1+y2)//2][-count + (x1+x2)//2] == 255:
            thickness += 1
            count += 1
        return thickness

def find_values(frame, min_gap, axis):
    h, w = frame.shape
    if   axis == 'x': l = w
    elif axis == 'y': l = h
    crop_values = []
    i = 0
    while i < l:
        if (axis == 'y' and frame[i][w - w//10] == 255) or (axis == 'x' and frame[h//2][i]):
            crop_values.append(i)
            i += min_gap
        else: i += 1
    return crop_values
    
def crop_image(frame, values, scale, axis, thickness=1):
    values = np.array(values)
    values = values // scale
    values = values.astype(int)
    if axis == "full":
        max_x = values[0]
        max_y = values[1]
        min_x = values[2]
        min_y = values[3]
        cv2.line(frame, (min_x, min_y), (max_x, min_y), 255, thickness)
        cv2.line(frame, (min_x, max_y), (max_x, max_y), 255, thickness)
        cv2.line(frame, (min_x, min_y), (min_x, max_y), 255, thickness)
        cv2.line(frame, (max_x, min_y), (max_x, max_y), 255, thickness)
        return frame[min_y:max_y, min_x:max_x]
    if axis == 'y':
        frames = []
        for i in range(1, len(values)):
            frames.append(frame[values[i-1]:values[i]])
        return frames
    if axis == 'x':
        frames = []
        for i in range(1, len(values)):
            frames.append(frame[:, values[i-1]:values[i]])
            #cv2.imshow("oi", frame[:, values[i-1]:values[i]])
            #cv2.waitKey(0)
        return frames

def gambiarra(text, frame):
    if "Data e Hora" in text:
        if "/" not in text:
            print("aqui")
            h, w = frame.shape
            resized = cv2.resize(frame, (2*w, 2*h))
            text = ocr.image_to_string(resized, lang='por')
    return text

def parser(text):
    name = ["issue_date", "number", "verification_code"]
    regEx = ['Data\se\sHora\sde\sEmiss.o\n.*(?P<value>../../....)', 'N.mero\sda\sNota(\n|.)*?(?P<value>\d{5,8})', 'C.digo\sde\s.erifica..o(\n|.)*?(?P<value>.{4,9})']
    match = None
    for i in range(len(regEx)):
        match = re.search(regEx[i], text)
        if match: 
            index = name[i]
            return match.group('value'), index
    return None, None


def tesseractOCR(frames, debug):
    tiff = []
    info = []
    for frame in frames:
        text = ocr.image_to_string(frame, lang='por')
        text = gambiarra(text, frame)
        if text: 
            value, index = parser(text)
            if value:
                tiff.append(frame)
                info.append({"value": value, "index": index})
        #full_text += text
        if debug:
            print(text)
            #cv2.imwrite("test.jpg", frame)
            cv2.imshow("cropped image", frame)
            cv2.waitKey(0)
    #return full_text
    return tiff, info

def microsoftOCR(frame, debug):
    subscription_key = "8a884e663b744b78a50337908d6e60a8" 
    endpoint = "https://v360testingocr.cognitiveservices.azure.com/"
    ocr_url = endpoint + "vision/v3.0/ocr"
    image_path = "test.jpg"
    image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'language': 'pt', 'detectOrientation': 'true'}

    full_text = ''
    _, encoded_image = cv2.imencode('.jpg', frame)
    buf = encoded_image.tobytes()
    response = requests.post(ocr_url, headers=headers, params=params, data=buf)
    response.raise_for_status()
    analysis = response.json()
    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    text = []
    for line in line_infos:
        for word_metadata in line:
            words = []
            for word_info in word_metadata["words"]:
                words.append(word_info["text"])
            text = ' '.join(words)
            full_text += text
            full_text += '\n'

    if debug:
        print(full_text)
    
    return full_text

def ocr_manager(file_path, mode='tesseract', debug=False, th_filter=False):
    # preprocess document
    time0 = time.time()
    res = 0.80
    max_gap = 0
    pages=convert_from_path(file_path)
    pages[0].save("tax.jpeg","jpeg")
    img = cv2.imread("tax.jpeg")
    gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = rescale_frame(gr, res)

    # find doc structure lines 
    h, w = gray.shape
    min_length = w//8
    _ret, th_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    pts, thickness = find_seed2(th_img, min_length, max_gap)
    min_gap = thickness*4
    points = np.zeros(th_img.shape)
    points, max_x, max_y, min_x, min_y = region_growing(pts, points, th_img, 0)

    # find segmentation values
    values0 = [max_x, max_y, min_x, min_y]
    cropped = crop_image(points, values0, 1, "full", thickness)
    values1 = find_values(cropped, min_gap, 'y')
    frames  = crop_image(cropped, values1, 1, 'y')
    values2 = []
    for f in frames:
        values2.append(find_values(f, min_gap, 'x'))

    # crop images
    if th_filter: _ret, gr = cv2.threshold(gr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gr = crop_image(gr, values0, res, "full")
    images = crop_image(gr, values1, res, 'y')
    crops = []
    for i in range(len(images)):
        gr = crop_image(images[i], values2[i], res, 'x')
        for f in gr:
            crops.append(f)
    time1 = time.time()
    
    # debug
    if debug:
        print("segmentation time: %fs"%(time1-time0))
        cv2.imshow("original image", img)
        cv2.waitKey(0)
        cv2.imshow("color filter application", th_img)
        cv2.waitKey(0)
        cv2.imshow("document skeleton", points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ocr
    if mode == 'tesseract':
        tiff, info = tesseractOCR(crops, debug=debug)
        cv2.destroyAllWindows()
        
    return tiff, info
    #file1 = open("tax.txt", "w")
    #file1.write(full_text)
    #file1.close()

def process_values(v2, mode):
    if mode in ["number"]:
        v2 = str(int(v2))
        for i in range(8 - len(v2)): v2 = '0' + v2
        return v2
    elif mode in ["total_value"]:
        integer = int(v2)
        decimal = int((v2%1)*100)
        return "%d,%d"%(integer, decimal)
    elif mode == "issue_date":
        year = v2[:4]
        mon  = v2[5:7]
        day  = v2[8:10]
        return "%s/%s/%s"%(day, mon, year)
    elif mode in ["verification_code", "supplier_legal_name"]:
        v2 = v2.upper()
        return v2

def create_samples():
    i = 0
    for file in glob.glob("pdfs/*.pdf"):
        try:
            tiff, info = ocr_manager(file)
            for j in range(len(tiff)):
                index = info[j]["index"]
                value = info[j]["value"]
                cv2.imwrite("data/%d%s.tif"%(i, index), tiff[j])
                txt = open("data/%d%s.gt.txt"%(i, index), "w+")
                txt.write(value)
                txt.close()
            i+=1
        except: print("erro")
    return i

def sort_samples(n):
    names = ["issue_date", "number", "verification_code"]
    format = ["%s", "%s", "%s"] #["Data e Hora de Emissão\n%s", "Número da Nota\n%s", "Código de Verificação\n%s"]
    cat_format = ["Data e Hora de Emissão", "Número da Nota", "Código de Verificação"]
    notas_df = pd.read_csv("csv/notas.csv", sep=',')
    for i in range(n):
        try:
            txt = open("data/%dverification_code.gt.txt"%i, "r")
            code = txt.read()
            notas_codes = notas_df["verification_code"].values
            if code not in notas_codes:
                max_ratio = 0
                for c in notas_codes:
                    ratio = fuzz.partial_ratio(code.lower(), c.lower())
                    if ratio > max_ratio:
                        max_ratio = ratio
                        code = c
            for j in range(len(names)):
                #file = open("data/%d%s.gt.txt"%(i, names[j]), "r")
                #value = file.read()
                img = cv2.imread("data/%d%s.tif"%(i, names[j]))
                h, w, _c = img.shape
                if names[j] == "issue_date": data = img[h//2:h, :11*w//20]
                else: data = img[h//2:h, :]
                categ = img[:h//2, :]
                cv2.imwrite("sorted_data/%s%s.tif"%(code, names[j]), data)
                cv2.imwrite("sorted_data/cat%s%s.tif"%(code, names[j]), categ)
                value = notas_df.loc[notas_df["verification_code"] == code][names[j]].values[0]
                value = process_values(value, names[j])
                new_file = open("sorted_data/%s%s.gt.txt"%(code, names[j]), "w+")
                new_file.write(format[j]%value)
                cat_file = open("sorted_data/cat%s%s.gt.txt"%(code, names[j]), "w+")
                cat_file.write(cat_format[j])
                #file.close()
                new_file.close()
                cat_file.close()
        except: print("erro")

create_samples()
sort_samples(91)
