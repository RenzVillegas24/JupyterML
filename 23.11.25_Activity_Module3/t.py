
p = '''
[
    {"key":"Recommended Use","value":["NOT STATED"]},
    {"key":"Primary Color","value":["Multi-Color"]},
    {"key":"Battery Type","value":["Does Not Contain a Battery"]},
    {"key":"Heel Height","value":["Flat/Low Heel (Under 1\\")"]},
    {"key":"Material","value":["GEL"]},
    {"key":"Product in Inches (L x W x H)","value":["6.0 x 6.0 x 1.0"]},
    {"key":"Gender","value":["Women"]},
    {"key":"Model No.","value":["46105"]},
    {"key":"Shoe Size","value":["OS"]},
    {"key":"Size","value":["OS"]},
    {"key":"Shoe Category","value":["Women's Shoes"]},
    {"key":"Container Type","value":["Box"]},
    {"key":"Multi Pack Indicator","value":["No"]},
    {"key":"Color","value":["Multicolor ,�� MULTI"]},
    {"key":"Model","value":["46105"]},
    {"key":"Shipping Weight (in pounds)","value":["1.0"]},
    {"key":"Walmart No.","value":["552675364"]},
    {"key":"Manufacturer Part Number","value":["46105"]},
    {"key":"Brand","value":["BEAUTIFEET"]},
    {"key":"Recommended Surface","value":["NOT STATED"]}]
'''

import json

print(json.loads(p))