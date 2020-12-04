import re
import houghOCR as ocr 
import microsoft
import sys
#import cv2
import time
import glob
import pandas as pd

def process_value(value):
    new = ''
    for c in value:
        if c.isnumeric():
            new += c
    return new[:-2]+','+new[-2:]

def parser(v):
    table = {}
    strings1 = ["net_value", "inss_value", "ir_value", "csll_value", "cofins_value", "pis_value", "iss_value", "total_value"]
    for key in v:
        if v[key] != None:
            value = v[key].group('value')
            if key in strings1:
                value = process_value(value)
            table[key] = [value]
        else: 
            if key in strings1:
                table[key] = ["0,00"]
            else:
                table[key] = [""]
    return table


def tesseract_parser(file_path):
    time0 = time.time()
    full_text = ocr.ocr_manager(file_path, mode='tesseract')

    v = {}
    v["verification_code"]               = re.search('C.digo\sde\s.erifica..o(\n|.)*?(?P<value>.{4,9})', full_text)
    v["supplier_legal_name"]             = re.search('PRESTADOR(\n|.)*?Raz.o\sSocial:\s(?P<value>.*)', full_text)
    v["supplier_identification_number"]  = re.search('PRESTADOR\sDE\sSERVIÇOS(\n|.)*?CPF.CNPJ.\s(?P<value>(\d|\.|/|-)*)\s', full_text)
    v["supplier_city"]                   = re.search('PRESTADOR(\n|.)*?Munic.pio:\s(?P<value>.*)\sUF', full_text)
    v["issue_date"]                      = re.search('Data\se\sHora\sde\sEmiss.o\n.*(?P<value>../../....)', full_text)
    if v["issue_date"] == None: v["issue_date"]  = re.search('emitido\sem(\s|\n)(?P<value>../../....)', full_text)
    v["net_due_date"]                    = re.search('Data\sde\svencimento(.|\n)*?:(\s|\n)(?P<value>../../....)', full_text)
    v["source_state_abbreviation"]       = re.search('PRESTADOR(\n|.)*?UF:\s(?P<value>..)', full_text)
    v["target_state_abbreviation"]       = re.search('TOMADOR(\n|.)*?UF:\s(?P<value>..)', full_text)
    v["customer_legal_name"]             = re.search('TOMADOR(\n|.)*?Raz.o\sSocial:\s(?P<value>.*)', full_text)
    v["customer_email"]                  = re.search('TOMADOR(\n|.)*?E-mail:\s(?P<value>.*)', full_text)
    v["supplier_municipal_registration"] = re.search('PRESTADOR(\n|.)*?Inscri..o\sMunicipal:\s(?P<value>.*)', full_text)
    v["complementary_law_116"]           = re.search('C.+igo\sdo\sS.r.+\n(?P<value>\d{5})\s-', full_text)
    v["cnae"]                            = re.search('CNAE:\s(?P<value>\d+)', full_text)#
    v["city_hall"]                       = re.search('(?P<value>PREFEITURA MUNICIPAL DE BARUERI)', full_text)
    v["customer_identification_number"]  = re.search('TOMADOR\sDE\sSERVIÇOS(\n|.)*?CPF.CNPJ.\s(?P<value>(\d|\.|/|-)*)\s', full_text)
    v["number"]                          = re.search('N.mero\sda\sNota(\n|.)*?(?P<value>\d{5,8})', full_text)
    v["net_value"]                       = re.search('PREFEITURA\sDO\sMUNICÍPIO\sDE\sS.O\sPAULO(\n|.)*?(?P<value>\d{5, 8})', full_text)#
    #v["ir_tax_rate"]                     = re.match('asasda', full_text)#
    #v["pis_tax_rate"]                    = re.match('dasda', full_text)#
    #v["cofins_tax_rate"]                 = re.match('asdasd', full_text)#
    #v["csll_tax_rate"]                   = re.match('asdad', full_text)#
    #v["inss_tax_rate"]                   = re.match('asdasd', full_text)#
    #v["iss_tax_rate"]                    = re.match('asdasda', full_text)#
    v["inss_value"]                      = re.search('INSS\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["ir_value"]                        = re.search('IRRF\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["pis_value"]                       = re.search('PIS.?PASEP\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["cofins_value"]                    = re.search('COFINS\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["csll_value"]                      = re.search('CSLL\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["iss_value"]                       = re.search('Valor\sdo\sISS\s?\(R.\)\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["total_value"]                     = re.search('VALOR\sTOTAL\sDO\sSERVI.O\s=\sR.\s(?P<value>(\d|\.)+,\d+)', full_text)

    table = parser(v)
    time1 = time.time()
    table["time"] = time1 - time0
    return pd.DataFrame(table)

def microsoft_parser(file_path):
    time0 = time.time()
    #file1 = open(file_path, "r")
    #full_text = file1.read()
    full_text = microsoft.ocr_manager(file_path)
    v = {}

    v["verification_code"]               = re.search('C.digo\sde\s.erifica..o(\n|.)*?(?P<value>.{4,9})', full_text)
    v["supplier_legal_name"]             = re.search('PRESTADOR(\n|.)*?Raz.o\sSocial:\s(?P<value>.*)', full_text)
    v["supplier_identification_number"]  = re.search('CPF.CNPJ.\s(?P<value>(\d|\.|/|-)*)', full_text)
    v["supplier_city"]                   = re.search('PRESTADOR(\n|.)*?Munic.pio:\s(?P<value>.*)\sUF', full_text)
    v["issue_date"]                      = re.search('Data\se\sHora\sde\sEmiss.o\n.*(?P<value>../../....)', full_text)
    v["net_due_date"]                    = re.search('Data\sde\svencimento(.|\n)*?:(\s|\n)(?P<value>../../....)', full_text)
    v["source_state_abbreviation"]       = re.search('PRESTADOR(\n|.)*?UF.\s(?P<value>..)', full_text)
    v["target_state_abbreviation"]       = re.search('TOMADOR(\n|.)*?UF.\s(?P<value>..)', full_text)
    v["customer_legal_name"]             = re.search('TOMADOR(\n|.)*?Raz.o\sSocial:\s(?P<value>.*)', full_text)
    v["customer_email"]                  = re.search('TOMADOR(\n|.)*?E-mail:\s(?P<value>.*)', full_text)
    v["supplier_municipal_registration"] = re.search('PRESTADOR(\n|.)*?Inscri..o\sMunicipal:\s(?P<value>.*)', full_text)
    v["complementary_law_116"]           = re.search('C.+igo\sdo\sS.r.+\n(?P<value>\d{5})\s-', full_text)
    v["cnae"]                            = re.search('CNAE:\s(?P<value>\d+)', full_text)#
    v["city_hall"]                       = re.search('(?P<value>PREFEITURA MUNICIPAL DE BARUERI)', full_text)
    v["customer_identification_number"]  = re.search('TOMADOR\sDE\sSERVIÇOS(\n|.)*?CPF.CNPJ.\s(?P<value>(\d|\.|/|-)*)', full_text)
    v["number"]                          = re.search('N.mero\sda\sNota(\n|.)*?(?P<value>\d{5,8})', full_text)
    v["net_value"]                       = re.search('PREFEITURA\sDO\sMUNICÍPIO\sDE\sS.O\sPAULO(\n|.)*?(?P<value>\d{5, 8})', full_text)#
    v["inss_value"]                      = re.search('INSS\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["ir_value"]                        = re.search('IRRF\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["pis_value"]                       = re.search('PIS.?PASEP\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["cofins_value"]                    = re.search('COFINS\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["csll_value"]                      = re.search('CSLL\s\(R.\).*\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["iss_value"]                       = re.search('Valor\sdo\sISS\s?\(R.\)\n(?P<value>(\d|\.)+,?\d+)', full_text)
    v["total_value"]                     = re.search('VALOR\sTOTAL\sDO\sSERVI.O\s=\sR.\s(?P<value>(\d|\.)+,\d+)', full_text)

    table = parser(v)
    time1 = time.time()
    table["time"] = time1 - time0
    return pd.DataFrame(table)


def test1():
    table = tesseract_parser("pdfs/2NGT-PCJ8.pdf")
    print(table)

def test():
    table = pd.DataFrame()
    for file in glob.glob("pdfs/*.pdf"):
        try:
            line = tesseract_parser(file)
            #line = microsoft_parser(file)
            print(line)
            table = table.append(line)
            #time.sleep(4) # prevents from using more than the free tier
        except:
            print("erro")
    print(table)
    table.to_csv("tesseract.csv")

#full = microsoft_parser("pdfs/K6XE-RYQA.pdf")
#full = ocr.ocr_manager("pdfs/3cxv-sn9h.pdf", mode="tesseract")
#print(full)
test()
