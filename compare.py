import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np

def process_values(v1, v2, mode):
    if mode in ["number"]:
        v1 = str(int(v1))
        v2 = str(int(v2))
        return v1, v2
    elif mode in ["total_value"]:
        vtemp = ''
        for c in v1:
            if c == ',': vtemp += '.'
            else: vtemp += c
        v1 = str(float(vtemp))
        v2 = str(float(v2))
        return v1, v2
    elif mode == "issue_date":
        year = v2[:4]
        mon  = v2[5:7]
        day  = v2[8:10]
        return v1, "%s/%s/%s"%(day, mon, year)
    elif mode in ["verification_code", "supplier_legal_name"]:
        v1 = v1.upper()
        v2 = v2.upper()
        return v1, v2
    elif mode in ["supplier_identification_number", "customer_identification_number"]:
        vtemp = ''
        for c in v1:
            if c.isnumeric():
                vtemp += c
        return vtemp, v2
    else:
        return v1, v2



my_df = pd.read_csv("tesseract.csv", sep=',')
hinode_df = pd.read_csv("csv/hinode.csv", sep=',')
my_df['internal_identifier'] = my_df['verification_code']
hinode_df['internal_identifier'] = hinode_df['verification_code']
fail = 0
found = 0
success = 0
identifier = 0
for code in my_df["verification_code"]:
    if pd.isnull(code):
        fail += 1
    elif code in hinode_df["verification_code"].values: #[x.upper() for x in hinode_df["verification_code"].values]:
        success += 1
        my_df["internal_identifier"] = my_df["internal_identifier"].replace([code], identifier)
        hinode_df["internal_identifier"] = hinode_df["internal_identifier"].replace([code], identifier)
        id = hinode_df.loc[hinode_df["verification_code"] == code]["internal_identifier"].values[0]
        if type(id) != int: print(type(id))
        identifier += 1
    else: 
        found += 1
        number = my_df.loc[my_df["verification_code"] == code]["number"].values
        if not pd.isnull(number): number = int(number[0])
        if number in hinode_df["number"].values:
            possible_codes = hinode_df.loc[hinode_df["number"] == number]["verification_code"].values
            max_ratio = 0
            for c in possible_codes:
                lower = fuzz.partial_ratio(code.lower(), c.lower())
                if lower > max_ratio: 
                    partial = fuzz.partial_ratio(code, c)
                    ratio = fuzz.ratio(code, c)
                    max_ratio = lower
                    hinode_code = c
            lower = max_ratio
            my_df["internal_identifier"] = my_df["internal_identifier"].replace([code], identifier)
            hinode_df["internal_identifier"] = hinode_df["internal_identifier"].replace([hinode_code], identifier)
            id = hinode_df.loc[hinode_df["verification_code"] == hinode_code]["internal_identifier"].values[0]
            if type(id) != int: print(type(id))
            identifier += 1
            #print(number)
            #print(code)
            #print(hinode_code)
            #print("ratio: %f"%ratio)
            #print("partial: %f"%partial)
            #print("lower: %f\n"%lower)
        #else: print("not found")

#my_df["internal_identifier"] = my_df["internal_identifier"].replace([np.nan], identifier)
#hinode_df["internal_identifier"] = hinode_df["internal_identifier"].replace([np.nan], identifier)
print("fail: %d, found: %d, success: %d" %(fail, found, success))
total = found+success
#print(my_df["internal_identifier"].values)
#print(hinode_df["internal_identifier"].values)

for col in my_df.columns:
    if col not in ["Unnamed: 0", "internal_identifier"] and col in hinode_df.columns and col in ["number", "issue_date", "verification_code", "supplier_legal_name", "supplier_identification_number", "customer_identification_number", "total_value"]:
        print(col)
        precise = 0
        similar = 0
        different = 0
        not_found = 0
        total = 0
        for i in range(identifier):
            total += 1
            my_value = my_df.loc[my_df["internal_identifier"] == i][col].values[0]
            hinode_value = hinode_df.loc[hinode_df["internal_identifier"] == i][col].values[0]
            if pd.isnull(my_value):
                not_found += 1
            else:
                my_value, hinode_value = process_values(my_value, hinode_value, col)
                ratio = fuzz.ratio(str(my_value), str(hinode_value))
                if ratio == 100: 
                    precise += 1
                elif ratio > 60: 
                    if col == "issue_date": print("mine: %s, hinode: %s"%(my_value, hinode_value))
                    similar += 1
                else: 
                    different += 1
                    if col == "verification_code":
                        print("mine: %s, hinode: %s"%(my_value, hinode_value))
        print("not found = %.2f%%" %(100*not_found/total))
        print("different = %.2f%%" %(100*different/total))
        print("similar = %.2f%%" %(100*similar/total))
        print("identical = %.2f%%\n" %(100*precise/total))

time = my_df["time"].values
mean_time = np.mean(time)
print("mean time = %fs"%mean_time)
            

